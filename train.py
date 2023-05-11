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
from bpe import BPE, Text
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
        return bpe.decode_text(encoded)


class ArcFace:

    def __init__(self, margin_m=0.5):
        self.m = margin_m

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def __call__(self, cosine, label):
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        print(cosine.shape, label.shape)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(2,
                         label.view(label.shape[0], label.shape[1], 1).long(),
                         1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output


def train(bpe: BPE, datapath, lr, epochs, model_size, pretrained, *, rank,
          world_size):
    FULL_BS = 500_000
    LOCAL_BS = {
        'xxs': 36,
        'xs': 32,
        's': 16,
        'm': 8,
        'l': 4,
        'xl': 2,
        'xxl': 1
    }[model_size] * 2
    CTX = 640
    ACCUMULATION = int(round(FULL_BS / (LOCAL_BS * CTX * world_size)))

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
        weights = torch.load(pretrained, map_location='cpu')
        tch.utils.load_state_dict_forgiving(m, weights['model'])

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
        loss = F.cross_entropy(logits.transpose(2, 1), batch[:, 1:])
        scaler.scale(loss / ACCUMULATION).backward()
        return {'loss': loss.item(), 'ppl': metrics.perplexity(loss).item()}

    @torch.no_grad()
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

    recipe = tch.recipes.TrainAndCall(
        m,
        train_fun,
        test_fun,
        train_loader,
        log_every=10,
        test_every=1000,
        checkpoint=f"model_{model_size}" if rank == 0 else None,
        visdom_env=f'mylm_{model_size}-lr={lr}-reinit'
        f'{"-finetune" if pretrained is not None else ""}'
        if rank == 0 else None)
    recipe.callbacks.add_callbacks([
        tcb.LRSched(tch.lr_scheduler.CosineDecay(
            optimizer,
            len(train_loader) * epochs,
            warmup_ratio=0.0 if pretrained is None else 0.05),
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
