import math
import torch
import torch.nn.functional as F
import random
import torchelie as tch
from model import make_transformer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torchelie.callbacks as tcb
from bpe import BPE, Text, normalize_text
import metrics
import os
from contextlib import suppress
import data_utils as data
from torchvision.transforms import Compose
import math


@torch.no_grad()
def sample(model,
           bpe,
           seed,
           ctx_len,
           num_tokens=100,
           top_k=100,
           top_p=0.9,
           temperature=1.0,
           typical_p=None,
           callback_seed=None,
           callback_stream=None):
    from sampler import (ForbiddenTokens, FixedRepetitionPenalty, TopK, TopP,
                         Temperature, Typical)
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
            prompt = torch.tensor(encoded[-ctx_len:], dtype=torch.long)

            logits = model(prompt[None].to(rank))[0][-1].float().cpu()
            idx = torch.arange(logits.shape[-1])
            logits, idx = ForbiddenTokens([bpe.STX, bpe.DC1, bpe.DC2,
                                           bpe.DC3])(logits, idx, prompt)
            #logits, idx = FixedRepetitionPenalty()(logits, idx, prompt)

            #top k sampling
            logits, idx = TopK(top_k)(logits, idx, prompt)

            #top p sampling
            logits, idx = TopP(top_p)(logits, idx, prompt)

            # typical sampling
            if typical_p is not None:
                logits, idx = Typical(typical_p)(logits, idx, prompt)
            logits, idx = Temperature(temperature)(logits, idx, prompt)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            next_token = idx[next_token]
            if callback_stream:
                callback_stream(next_token)
            encoded.append(next_token)
        return bpe.decode_text(encoded)


class ThinkObjective:

    def __init__(self, think_token):
        self.think_token = think_token

    def __call__(self, x):
        x, y = data.NextTokenObjective()(x)
        N = x.shape[0]
        num_think_tokens = random.randint(0, 5)
        pos_think_tokens = random.randint(1, len(x) - num_think_tokens - 1)
        x = torch.cat([
            x[:pos_think_tokens],
            torch.tensor([self.think_token] * num_think_tokens,
                         dtype=torch.long), x[pos_think_tokens:]
        ])
        y = torch.cat([
            y[:pos_think_tokens - 1],
            torch.tensor([y[pos_think_tokens - 1]] * num_think_tokens,
                         dtype=torch.long), y[pos_think_tokens - 1:]
        ])
        return x[:N], y[:N]


class MLMObjective:

    def __init__(self, mask_token):
        self.mask_token = mask_token

    def __call__(self, x):
        x, y = data.NextTokenObjective()(x)
        # randomly mask 15% of the input tokens
        mask = torch.rand_like(x, dtype=torch.float) < 0.15
        x[mask] = self.mask_token
        return x, y


class RandomPad:

    def __init__(self, pad_token):
        self.pad_token = pad_token

    def __call__(self, x):
        x, y = data.NextTokenObjective()(x)
        N = x.shape[0]
        for _ in range(5):
            num_pad_tokens = random.randint(0, 5)
            pos_pad_tokens = random.randint(1, len(x) - num_pad_tokens - 1)
            x = torch.cat([
                x[:pos_pad_tokens],
                torch.tensor([self.pad_token] * num_pad_tokens,
                             dtype=torch.long), x[pos_pad_tokens:]
            ])
            y = torch.cat([
                y[:pos_pad_tokens - 1],
                torch.tensor([y[pos_pad_tokens - 1]] * num_pad_tokens,
                             dtype=torch.long), y[pos_pad_tokens - 1:]
            ])
        return x[:N], y[:N]


def think_gain(x, y, loss, bpe):
    # Compare the loss of the token preceding <|THINK|> to the loss of the token following <|THINK|>
    # This is a proxy for the gain in perplexity that we get from thinking
    think_mask = (x == bpe.specials['<|THINK|>'])
    first_think = think_mask.int().argmax(dim=1) - 1
    num_think = (first_think >= 0).sum()
    last_think = first_think + think_mask.sum(1)
    think_gain = torch.sum(loss[torch.arange(len(x)), last_think] -
                           loss[torch.arange(len(x)), first_think]) / num_think

    assert (y[torch.arange(len(x)), last_think] == y[torch.arange(len(x)),
                                                     first_think]).all()
    return think_gain


def unthink(inputs):
    think_mask = (inputs == 5)
    think_offset = think_mask.cumsum(1) * think_mask
    inputs = inputs.view(-1)[torch.arange(inputs.numel(), device=inputs.device)
                             - think_offset.to(inputs.device).view(-1)].view(
                                 *inputs.shape)
    think_offset = think_mask[..., None]
    return inputs, think_mask


def train(*, datapath, lr, epochs, model_size, pretrained, bpe_path,
          batch_size, global_batch_size, rank, world_size):
    FULL_BS = global_batch_size
    LOCAL_BS = batch_size
    CTX = 512
    ACCUMULATION = int(round(FULL_BS / (LOCAL_BS * CTX * world_size)))

    if pretrained is not None:
        checkpoint = torch.load(pretrained, map_location='cpu')

    if bpe_path is not None:
        print('loading BPE from', bpe_path)
        bpe = BPE.load(bpe_path)
    else:
        print('Using BPE from checkpoint')
        bpe = BPE()
        bpe.load_state_dict(checkpoint['bpe'])
    bpe.add_special('<|THINK|>', 5)

    sampler = data.TextDirSampler(
        datapath,
        CTX + 1,
        bpe.unicode_for_special('<|SOH|>'),
        Compose([
            normalize_text,
            data.RandomCase(bpe.specials['<|DC1|>'],
                            bpe.specials['<|DC2|>'],
                            uppercase_p=0.0,
                            lowercase_p=0.00),
            data.RandomPromptSplit(bpe.specials['<|STX|>'], p=0.00),
            data.Tokenize(bpe,
                          bpe.specials['<|DC3|>'],
                          dropout=0.1,
                          dropout_p=0.00)
        ]),
        to_input_and_target=data.NextTokenObjective())

    test_sampler = data.EvalDirSampler('test', CTX + 1, bpe)
    train_loader = DataLoader(sampler,
                              LOCAL_BS,
                              shuffle=True,
                              num_workers=16,
                              drop_last=True,
                              pin_memory=True,
                              persistent_workers=True)
    print('#batches', len(train_loader), '#dataset', len(train_loader.dataset),
          'bs', LOCAL_BS, 'accum', ACCUMULATION)

    basem = make_transformer(model_size, len(bpe.vocab), CTX,
                             dropout=0.).to(rank)

    print(basem.num_parameters() / 1e6, 'M params')
    print(basem.num_parameters_without_embeddings() / 1e6,
          'M params without embeddings')

    m = torch.compile(basem)
    if pretrained is not None:
        tch.utils.load_state_dict_forgiving(m,
                                            checkpoint['model'],
                                            fit_dst_size=True)

        ckpt_ctx = checkpoint['model']['_orig_mod.positional_embedding'].shape[
            1]
        m.positional_embedding.data[:, :min(CTX, ckpt_ctx)] = checkpoint[
            'model']['_orig_mod.positional_embedding'][:, :min(CTX, ckpt_ctx)]

        ckpt_vocab_size = checkpoint['model'][
            '_orig_mod.token_embedding.weight'].shape[0]
        m.token_embedding.weight.data[:min(
            len(bpe.vocab), ckpt_vocab_size
        )] = checkpoint['model']['_orig_mod.token_embedding.weight'][:min(
            len(bpe.vocab), ckpt_vocab_size)]

        m.unembed.weight.data[:min(len(bpe.vocab), ckpt_vocab_size
                                   )] = checkpoint['model'][
                                       '_orig_mod.unembed.weight'][:min(
                                           len(bpe.vocab), ckpt_vocab_size)]

    if world_size > 1:
        m = DDP(m, device_ids=[rank], output_device=rank)

    # weight decay from Cramming paper: 0.01
    # weight decay from LLaMA: 0.1
    # betas from LLaMA / nanoGPT
    optimizer = AdamW([{
        'params': params,
        'lr': lr,
        'weight_decay': 0.01 if decayable else 0.0
    } for (decayable, fan_in), params in basem.mu_parametrization().items()],
                      betas=(0.9, 0.95))

    scaler = torch.cuda.amp.GradScaler()

    def train_fun(batch):
        x, y = batch
        with torch.autocast('cuda'):
            logits = m(x).float()
        loss = F.cross_entropy(logits.transpose(2, 1), y, reduction='none')
        scaler.scale(loss.mean() / ACCUMULATION).backward()
        loss_per_char = torch.mean(
            loss.sum(dim=1).cpu() /
            torch.tensor([len(bpe.decode_text(xx)) for xx in x.cpu()]))
        return {
            'loss': loss_per_char.item(),
            'ppl': metrics.perplexity(loss_per_char).item(),
        }

    @torch.no_grad()
    def test_fun():
        basem.eval()
        with torch.autocast('cuda'):
            outs = [
                sample(basem, bpe, chr(bpe.SOH), CTX, num_tokens=CTX)
                for _ in range(10)
            ]

        for out in outs:
            print('-', out)

        test_loss = 0
        for x in test_sampler:
            xgpu = x.to(rank)
            with torch.autocast('cuda'):
                logits = m(xgpu[None, :-1]).float()
            loss = F.cross_entropy(logits.transpose(2, 1), xgpu[None, 1:])
            # get loss per char to account for different tokenizations
            loss *= x.shape[0] / len(bpe.decode_text(x))
            del xgpu
            test_loss += loss
        test_loss /= len(test_sampler)

        basem.train()
        return {
            'outs': '<hr/>'.join(outs).replace('\n', '<br/>'),
            'loss': test_loss,
            'ppl': metrics.perplexity(test_loss).item()
        }

    recipe = tch.recipes.TrainAndCall(
        m,
        train_fun,
        test_fun,
        train_loader,
        log_every=10,
        test_every=1000,
        checkpoint=f"model_{model_size}" if rank == 0 else None,
        visdom_env=f'mylm_{model_size}-lr={lr}'
        f'{"-finetune" if pretrained is not None else ""}'
        if rank == 0 else None)
    recipe.callbacks.add_callbacks([
        tcb.LRSched(tch.lr_scheduler.CosineDecay(
            optimizer,
            len(train_loader) * epochs,
            warmup_ratio=0.05 if pretrained is None else 0.0),
                    metric=None,
                    step_each_batch=True),
        tcb.Optimizer(optimizer,
                      log_lr=True,
                      clip_grad_norm=2,
                      scaler=scaler,
                      accumulation=ACCUMULATION,
                      grad_multiplier=ACCUMULATION),
        tcb.Log('loss', 'loss'),
        tcb.Log('ppl', 'ppl'),
    ])
    recipe.test_loop.callbacks.add_callbacks([
        tcb.Log('outs', 'outs'),
        tcb.Log('loss', 'loss'),
        tcb.Log('ppl', 'ppl'),
    ])
    recipe.register('model_type', model_size)
    recipe.register('bpe', bpe)
    recipe.to(rank)
    recipe.run(epochs)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--model', default='xxs')
    parser.add_argument('--pretrained')
    parser.add_argument('--bpe')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--global-batch-size', type=int, default=500_000)
    args = parser.parse_args()

    if not args.bpe and not args.pretrained:
        raise ValueError('Either --bpe or --pretrained must be specified')

    tch.utils.parallel_run(
        train,
        datapath='data/raw/x',
        lr=args.lr,
        epochs=args.epochs,
        model_size=args.model,
        pretrained=args.pretrained,
        bpe_path=args.bpe,
        batch_size=args.batch_size,
        global_batch_size=args.global_batch_size,
    )
