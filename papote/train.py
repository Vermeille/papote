import html
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
from papote.bpe import BPE
import papote.metrics as metrics


class BestAndWorst:
    def __init__(self, bpe, k=5):
        super().__init__()
        self.bpe = bpe
        self.k = k
        self.worst = []
        self.best = []

    def on_epoch_start(self, state):
        self.worst = []
        self.best = []

    @torch.no_grad()
    def on_batch_end(self, state):
        all = self.best + self.worst + list(
            zip(state['loss_per_sentence'],
                (self.bpe.decode_text(xx) for xx in state['batch'][0])))
        all.sort(key=lambda x: -x[0])
        self.worst = all[:self.k]
        self.best = all[-self.k:]

        if state['iters'] % 100 == 0:
            e = html.escape
            state['metrics']['inspect'] = f"""<h3>Best</h3>
            <table>
            <tr><th>Loss</th><th>Text</th></tr>
            {''.join(f'<tr><td>{x[0]}</td><td>{e(repr(x[1]))}</td></tr>' for x
                in reversed(self.best))}
            </table>
            <h3>Worst</h3>
            <table>
            <tr><th>Loss</th><th>Text</th></tr>
            {''.join(f'<tr><td>{x[0]}</td><td>{e(repr(x[1]))}</td></tr>' for x in self.worst)}
            </table>
            """


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


class LogCtxLoss:

    def __init__(self, loss):
        self.loss = loss

    def to_visdom(self, vis, name):
        vis.line(X=list(range(len(self.loss))),
                 Y=self.loss,
                 win=name,
                 name=name,
                 opts=dict(title=name))


def train(*, datapath, lr, chinchilla_factor, model_size, pretrained, bpe_path,
          batch_size, global_batch_size, rank, world_size):
    torch.cuda.set_device(rank)
    FULL_BS = global_batch_size
    LOCAL_BS = batch_size
    CTX = 256
    ACCUMULATION = int(max(1, round(FULL_BS / (LOCAL_BS * CTX * world_size))))

    if pretrained is not None:
        checkpoint = torch.load(pretrained, map_location='cpu')

    if bpe_path is not None:
        print('loading BPE from', bpe_path)
        bpe = BPE.load(bpe_path)
        print(bpe.vocab)
    else:
        print('Using BPE from checkpoint')
        bpe = BPE()
        bpe.load_state_dict(checkpoint['bpe'])
    bpe.add_special('<|THINK|>', 5)
    bpe.add_special('<|SUFFIX|>', bpe.specials['<|NAK|>'])
    bpe.add_special('<|PREFIX|>', bpe.specials['<|SYN|>'])
    bpe.add_special('<|WRAP|>', bpe.specials['<|ETB|>'])

    basem = make_transformer(model_size, len(bpe.vocab), CTX).to(rank)

    print(basem.num_parameters() / 1e6, 'M params')
    print(basem.num_parameters_without_embeddings() / 1e6,
          'M params without embeddings')

    print('computing chinchilla optimal training time:',
          (basem.num_parameters() * 20) / 1e6, 'M tokens')

    if pretrained is not None:
        tch.utils.load_state_dict_forgiving(m,
                                            checkpoint['model'],
                                            fit_dst_size=True)
    if world_size > 1:
        m = basem  #torch.compile(basem)
        m = DDP(m, device_ids=[rank], output_device=rank)
        #m = FSDP(basem)
    else:
        m = torch.compile(basem)
    print(m)
    sampler = data.ChunkSampler(
        datapath,
        CTX + 1,
        '<|SOH|>',
        Compose([
            data.NFKC(),
            data.CleanPrivateUnicode(),
            data.Tokenize(bpe, bpe.specials['<|DC3|>'], dropout_p=0.00),
            data.Align(CTX + 1, bpe.specials['<|EOT|>']),
            #data.FillInTheMiddle(bpe.specials['<|SUFFIX|>'], bpe.specials['<|PREFIX|>'], bpe.specials['<|WRAP|>'], p=0.5),
        ]),
        to_input_and_target=data.NextTokenObjective())

    test_sampler = data.EvalDirSampler('test', CTX + 1, bpe)
    train_loader = DataLoader(sampler,
                              LOCAL_BS,
                              num_workers=1,
                              drop_last=True,
                              pin_memory=True,
                              persistent_workers=True)

    # chinchilla advises 20 tokens per parameter, without counting the
    # embeddings
    num_iters = round(basem.num_parameters_without_embeddings() * 20 *
                      chinchilla_factor / (CTX * LOCAL_BS * world_size) + 1)
    print('#iter', num_iters, 'bs', LOCAL_BS, 'accum', ACCUMULATION, '#tokens',
          num_iters * CTX * LOCAL_BS * world_size / 1e6, 'M')
    print('bs', LOCAL_BS, 'accum', ACCUMULATION)

    # weight decay from Cramming paper: 0.01
    # weight decay from LLaMA: 0.1
    # betas from LLaMA / nanoGPT
    optimizer = AdamW([{
        'params': params,
        'lr': lr,
        'weight_decay': 0.1 if decayable else 0.0
    } for (decayable, fan_in), params in basem.mu_parametrization().items()],
                      betas=(0.9, 0.95))  # transformer++ betas

    scaler = torch.cuda.amp.GradScaler()
    loss_fn = data.SeqWeightedLoss(0.99, loss_fn=data.binary_entropy)

    def train_fun(batch):
        x, y = batch
        with torch.autocast('cuda'):
            pred = m(x).float()
        loss = loss_fn(pred.transpose(1, 2), y)
        scaler.scale(loss.mean() / ACCUMULATION).backward()
        loss_per_char = torch.mean(
            loss.sum(dim=1).cpu() /
            torch.tensor([len(bpe.decode_text(xx)) for xx in x.cpu()]))
        return {
            'pred': pred.detach(),
            'loss_at_pos': LogCtxLoss(loss.mean(dim=0)),
            'loss_per_sentence': loss.mean(dim=1),
            'pos_weight': LogCtxLoss(loss_fn.weight),
            'loss': loss_per_char.item(),
            'ppl': metrics.perplexity(loss_per_char).item(),
        }

    @torch.no_grad()
    def test_fun():
        basem.eval()
        with torch.autocast('cuda'):
            outs = []
            for _ in range(10):
                sample = default_sampler(basem, bpe, length=CTX)
                outs.append(sample.sample('<|SOH|>'))

        state = {'metrics': {}}
        test_loss = 0
        topk = tcb.TopkAccAvg(15, False, 'running')
        topk.on_epoch_start(state)
        total_acc = 0
        for x in test_sampler:
            xgpu = x.to(rank)
            with torch.autocast('cuda'):
                preds = m(xgpu[None, :-1]).float()
            loss = F.cross_entropy(preds.transpose(1, 2),
                                   xgpu[None, 1:],
                                   reduction='none').mean()
            # get loss per char to account for different tokenizations
            loss *= x.shape[0] / len(bpe.decode_text(x))
            state['pred'] = preds
            state['batch'] = (None, xgpu[None, 1:])
            topk.on_batch_end(state)
            del xgpu
            test_loss += loss
        test_loss /= len(test_sampler)
        topk.on_epoch_end(state)

        basem.train()
        return {
            'outs': '<hr/>'.join(outs).replace('\n', '<br/>'),
            'loss': test_loss,
            'ppl': metrics.perplexity(test_loss).item(),
            **state['metrics'],
        }

    recipe = tch.recipes.TrainAndCall(
        m,
        train_fun,
        test_fun,
        train_loader,
        log_every=10,
        test_every=500,
        checkpoint=f"model_{model_size}" if rank == 0 else None,
        visdom_env=f'mylm_{model_size}-lr={lr}'
        f'{"-finetune" if pretrained is not None else ""}'
        if rank == 0 else None)
    recipe.callbacks.add_callbacks([
        tcb.LRSched(tch.lr_scheduler.CosineDecay(
            optimizer,
            num_iters,
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
        tcb.Log('loss_at_pos', 'loss_at_pos'),
        tcb.Log('pos_weight', 'pos_weight'),
        tcb.TopkAccAvg(k=15, post_each_batch=True),
        BestAndWorst(bpe),
    ])
    recipe.test_loop.callbacks.add_callbacks([
        tcb.Log('outs', 'outs'),
        tcb.Log('loss', 'loss'),
        tcb.Log('ppl', 'ppl'),
        tcb.Log('topk15_acc', 'topk15_acc'),
    ])
    recipe.register('loss_fn', loss_fn)
    recipe.register('model_type', model_size)
    recipe.register('bpe', bpe)
    recipe.to(rank)
    recipe.run(1)


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
