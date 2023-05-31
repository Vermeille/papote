import math
import torch
import torch.nn.functional as F
import random
import torchelie as tch
from torch.optim import AdamW, SparseAdam
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torchelie.callbacks as tcb
import os
from contextlib import suppress
from torchvision.transforms import Compose
from papote.sampler import default_sampler
from papote.model import make_transformer
import papote.data_utils as data
from papote.bpe import BPE, Text
import papote.metrics as metrics


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
        x, y = data.NextTokenObjective()(torch.tensor(x, dtype=torch.long))
        N = len(x)
        for _ in range(random.randrange(5)):
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


def train(*, datapath, lr, chinchilla_factor, model_size, pretrained, bpe_path,
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
    bpe.add_special('<|SUFFIX|>', bpe.specials['<|NAK|>'])
    bpe.add_special('<|PREFIX|>', bpe.specials['<|SYN|>'])
    bpe.add_special('<|WRAP|>', bpe.specials['<|ETB|>'])

    basem = make_transformer(model_size, len(bpe.vocab), CTX,
                             dropout=0.1).to(rank)

    print(basem.num_parameters() / 1e6, 'M params')
    print(basem.num_parameters_without_embeddings() / 1e6,
          'M params without embeddings')

    print('computing chinchilla optimal training time:',
          (basem.num_parameters() * 20) / 1e6, 'M tokens')

    m = torch.compile(basem)
    if pretrained is not None:
        tch.utils.load_state_dict_forgiving(m,
                                            checkpoint['model'],
                                            fit_dst_size=True)

    if world_size > 1:
        m = DDP(m, device_ids=[rank], output_device=rank)
    sampler = data.TextDirSampler(
        datapath,
        CTX + 1,
        bpe.unicode_for_special('<|SOH|>'),
        Compose([
            data.CleanPrivateUnicode(),
            data.Tokenize(bpe, bpe.specials['<|DC3|>'], dropout_p=0.00),
            data.Crop(CTX + 1),
            data.FillInTheMiddle(bpe.specials['<|SUFFIX|>'],
                                 bpe.specials['<|PREFIX|>'],
                                 bpe.specials['<|WRAP|>'],
                                 p=0.5),
        ]),
        to_input_and_target=data.NextTokenObjective())

    random_sampler = torch.utils.data.WeightedRandomSampler(
        [1] * len(sampler),
        LOCAL_BS * 1000,
        generator=torch.Generator().manual_seed(42 + rank))
    test_sampler = data.EvalDirSampler('test', CTX + 1, bpe)
    train_loader = DataLoader(sampler,
                              LOCAL_BS,
                              sampler=random_sampler,
                              num_workers=16,
                              drop_last=True,
                              pin_memory=True,
                              persistent_workers=True)

    epochs = round(basem.num_parameters() * 20 * chinchilla_factor /
                   (len(train_loader) * CTX * LOCAL_BS * world_size) + 1)
    print('#iter',
          len(train_loader) * epochs, 'len(dataset)',
          len(train_loader.dataset), 'bs', LOCAL_BS, 'accum', ACCUMULATION,
          '#tokens',
          len(train_loader) * epochs * CTX * LOCAL_BS * world_size / 1e6, 'M')

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
        sample = default_sampler(basem, bpe, length=CTX)
        with torch.autocast('cuda'):
            outs = [sample.sample(chr(bpe.SOH)) for _ in range(10)]

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
            len(train_loader) * round(epochs),
            warmup_ratio=0.05 if pretrained is None else 0.0),
                    metric=None,
                    step_each_batch=True),
        tcb.Optimizer(optimizer,
                      log_lr=True,
                      clip_grad_norm=0.5,
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
    recipe.run(round(epochs))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--chinchilla-factor', type=float, default=1.0)
    parser.add_argument('--model', default='xxs')
    parser.add_argument('--pretrained')
    parser.add_argument('--bpe')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--global-batch-size', type=int, default=500_000)
    parser.add_argument('--data', default='data/')
    args = parser.parse_args()

    if not args.bpe and not args.pretrained:
        raise ValueError('Either --bpe or --pretrained must be specified')

    tch.utils.parallel_run(
        train,
        datapath=args.data,
        lr=args.lr,
        chinchilla_factor=args.chinchilla_factor,
        model_size=args.model,
        pretrained=args.pretrained,
        bpe_path=args.bpe,
        batch_size=args.batch_size,
        global_batch_size=args.global_batch_size,
    )
