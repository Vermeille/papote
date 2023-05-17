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
           top_k=40,
           top_p=0.95,
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


def train(datapath, lr, epochs, model_size, pretrained, bpe_path, batch_size,
          *, rank, world_size):
    FULL_BS = 500_000
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

    sampler = data.TextDirSampler(
        datapath, CTX + 1, bpe.unicode_for_special('<|SOH|>'),
        Compose([
            data.RandomCase(bpe.unicode_for_special('<|DC1|>'),
                            bpe.unicode_for_special('<|DC2|>'),
                            uppercase_p=0.0,
                            lowercase_p=0.00),
            data.RandomPromptSplit(bpe.unicode_for_special('<|STX|>'), p=0.00),
            data.Tokenize(bpe,
                          bpe.unicode_for_special('<|DC3|>'),
                          dropout=0.1,
                          dropout_p=0.00)
        ]))

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
        with torch.autocast('cuda'):
            logits = m(batch[:, :-1]).float()
        loss = F.cross_entropy(logits.transpose(2, 1),
                               batch[:, 1:],
                               reduction='none')
        scaler.scale(loss.mean() / ACCUMULATION).backward()
        loss_per_char = torch.mean(
            loss.sum(dim=1).cpu() /
            torch.tensor([len(bpe.decode_text(x)) for x in batch.cpu()]))
        return {
            'loss': loss_per_char.item(),
            'ppl': metrics.perplexity(loss_per_char).item()
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
    args = parser.parse_args()

    if not args.bpe and not args.pretrained:
        raise ValueError('Either --bpe or --pretrained must be specified')

    tch.utils.parallel_run(
        train,
        'data/',
        args.lr,
        args.epochs,
        args.model,
        args.pretrained,
        args.bpe,
        args.batch_size,
    )
