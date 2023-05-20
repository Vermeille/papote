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
from sampler import default_sampler


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
        for _ in range(random.randint(5)):
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
    bpe.add_special('<|SUFFIX|>', bpe.specials['<|NAK|>'])
    bpe.add_special('<|PREFIX|>', bpe.specials['<|SYN|>'])
    bpe.add_special('<|WRAP|>', bpe.specials['<|ETB|>'])

    sampler = data.TextDirSampler(
        datapath,
        CTX + 1,
        bpe.unicode_for_special('<|SOH|>'),
        Compose([
            normalize_text,
            data.CleanPrivateUnicode(),
            data.Tokenize(bpe, bpe.specials['<|DC3|>'], dropout_p=0.00),
            data.Crop(CTX + 1),
            data.FillInTheMiddle(bpe.specials['<|SUFFIX|>'],
                                 bpe.specials['<|PREFIX|>'],
                                 bpe.specials['<|WRAP|>'],
                                 p=0.5),
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
        sample = default_sampler(basem, bpe)
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
