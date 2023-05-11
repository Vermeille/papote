import torch
import torch.nn.functional as F
import random
import torchelie as tch
from discordloader import discord_load
from model import make_transformer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torchelie.callbacks as tcb
from bpe import BPE, Text
import metrics
import os
from contextlib import suppress
import data_utils as data
from torchvision.transforms import Compose


def sample(model,
           bpe,
           seed,
           ctx_len,
           num_tokens=100,
           top_k=40,
           top_p=0.95,
           temperature=1.0,
           typical_p=None,
           callback_seed=None,
           callback_stream=None):
    rank = next(model.parameters()).device
    model.eval()
    encoded = bpe.encode_text(seed)
    if callback_seed is not None:
        callback_seed(encoded)
    with suppress(KeyboardInterrupt):
        for _ in range(num_tokens):
            t = Text('')
            t.set_tokens(encoded)
            t.tokenize(bpe.merges)
            encoded = t.as_tokens()
            encoded_t = torch.tensor([encoded[-ctx_len:]],
                                     dtype=torch.long).to(rank)
            logits = model(encoded_t)[0][-1].cpu()
            logits[torch.tensor([bpe.STX, bpe.DC1, bpe.DC2,
                                 bpe.DC3])] = -float('inf')
            logits = logits / temperature

            #top k sampling
            top_indices = torch.topk(logits, min(top_k, logits.shape[-1]))[1]
            logits = logits[top_indices]

            #top p sampling
            probs = F.softmax(logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)
            cumulative_probs[0] = 0
            logits = logits[cumulative_probs < top_p]
            top_indices = top_indices[cumulative_probs < top_p]

            # typical sampling
            if typical_p is not None:
                normalized = F.log_softmax(logits, dim=-1)
                entropy = -torch.sum(normalized.exp() * normalized, dim=-1)
                shifted = torch.abs(-normalized - entropy)
                typical_sorted = torch.argsort(shifted, dim=-1)
                top_indices = top_indices[typical_sorted]
                typical_probs = F.softmax(logits[typical_sorted],
                                          dim=-1).cumsum(dim=-1)

                typical_probs[0] = 0
                typical_sorted = typical_sorted[typical_probs < typical_p]
                logits = logits[typical_sorted]

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            next_token = top_indices[next_token]
            if callback_stream:
                callback_stream(next_token)
            encoded.append(next_token)
        return bpe.decode_text(encoded, sep.encode())


def train(bpe: BPE, datapath, lr, epochs, model_size, pretrained, *, rank,
          world_size):
    FULL_BS = 500_000
    LOCAL_BS = {
        'xxs': 48,
        'xs': 32,
        's': 16,
        'm': 8,
        'l': 4,
        'xl': 2,
        'xxl': 1
    }[model_size]
    CTX = 640
    ACCUMULATION = FULL_BS // (LOCAL_BS * world_size * CTX)

    sampler = data.TextDirSampler(
        datapath, CTX + 1, bpe.unicode_for_special('<|SOH|>'),
        Compose([
            data.RandomCase(bpe.unicode_for_special('<|DC2|>'),
                            bpe.unicode_for_special('<|DC3|>'),
                            uppercase_p=0.0,
                            lowercase_p=0.05),
            data.RandomPromptSplit(bpe.unicode_for_special('<|STX|>'), p=0.05),
            data.Tokenize(bpe,
                          bpe.unicode_for_special('<|DC1|>'),
                          dropout=0.1,
                          dropout_p=0.05)
        ]))

    test_sampler = data.EvalDirSampler('test', CTX + 1, bpe)
    train_loader = DataLoader(sampler,
                              LOCAL_BS,
                              shuffle=True,
                              num_workers=4,
                              drop_last=True,
                              pin_memory=True,
                              persistent_workers=True)
    print('#batches', len(train_loader), '#dataset', len(train_loader.dataset),
          'bs', LOCAL_BS, 'accum', ACCUMULATION)

    basem = make_transformer(model_size, len(bpe.vocab), CTX,
                             dropout=0.1).to(rank)

    print(basem.num_parameters() / 1e6, 'M params')
    print(basem.num_parameters_without_embeddings() / 1e6,
          'M params without embeddings')

    m = torch.compile(basem)
    if pretrained is not None:
        tch.utils.load_state_dict_forgiving(
            m,
            torch.load(pretrained, map_location='cpu')['model'])

    if world_size > 1:
        m = DDP(m, device_ids=[rank], output_device=rank)

    def train_fun(batch):
        logits = m(batch[:, :-1])
        loss = F.cross_entropy(logits.transpose(2, 1), batch[:, 1:])
        loss.backward()
        return {'loss': loss.item(), 'ppl': metrics.perplexity(loss).item()}

    def test_fun():
        basem.eval()
        outs = [
            sample(basem, bpe, chr(bpe.SOH), CTX, num_tokens=CTX)
            for _ in range(10)
        ]

        for out in outs:
            print('-', out)

        test_loss = 0
        for x in test_sampler:
            x = x.to(rank)
            logits = m(x[None, :-1])
            loss = F.cross_entropy(logits.transpose(2, 1), x[None, 1:])
            test_loss += loss
        test_loss /= len(test_sampler)

        basem.train()
        return {
            'outs': '<hr/>'.join(outs).replace('\n', '<br/>'),
            'loss': test_loss,
            'ppl': metrics.perplexity(test_loss).item()
        }

    if True:
        optimizer = AdamW([{
            'params': params,
            'lr': lr * (1 if fan_in == 1 else 1),
            'weight_decay': 0.1 if decayable else 0.0
        } for (decayable,
               fan_in), params in basem.mu_parametrization().items()],
                          betas=(0.9, 0.99))
    else:
        optimizer = AdamW([{
            'params': basem.decayable_parameters(),
            'weight_decay': 0.01,
        }, {
            'params': basem.undecayable_parameters(),
            'weight_decay': 0.0
        }],
                          lr=lr,
                          betas=(0.9, 0.99))

    recipe = tch.recipes.TrainAndCall(
        m,
        train_fun,
        test_fun,
        train_loader,
        log_every=10,
        test_every=1000,
        checkpoint=f"model_{model_size}" if rank == 0 else None,
        visdom_env=f'mylm_{model_size}-lr={lr}-reinit-fan_in'
        if rank == 0 else None)
    recipe.callbacks.add_callbacks([
        tcb.LRSched(tch.lr_scheduler.CosineDecay(optimizer,
                                                 len(train_loader) * epochs,
                                                 0.05),
                    metric=None,
                    step_each_batch=True),
        tcb.Optimizer(optimizer,
                      log_lr=True,
                      clip_grad_norm=1,
                      accumulation=ACCUMULATION),
        tcb.Log('loss', 'loss'),
        tcb.Log('ppl', 'ppl'),
    ])
    recipe.test_loop.callbacks.add_callbacks([
        tcb.Log('outs', 'outs'),
        tcb.Log('loss', 'loss'),
        tcb.Log('ppl', 'ppl'),
    ])
    recipe.to(rank)
    recipe.run(epochs)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--model', default='xxs')
    parser.add_argument('--pretrained')
    args = parser.parse_args()

    bpe = BPE.load('bpe.json')

    tch.utils.parallel_run(
        train,
        bpe,
        'data/',
        args.lr,
        args.epochs,
        args.model,
        args.pretrained,
    )
