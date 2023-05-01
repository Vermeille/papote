import torch
from bpe import BPE
from data_utils import EvalDirSampler
from model import make_transformer
import torch.nn.functional as F
import metrics

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    CTX = 640

    rank = 'cpu'
    bpe = BPE.load('bpe.json')
    test_sampler = EvalDirSampler('test', CTX + 1, bpe)
    print(len(test_sampler))
    m = make_transformer('xxs', len(bpe.vocab), CTX).to(rank)
    m.eval()

    m = torch.compile(m)
    m.load_state_dict(torch.load('good.pth', map_location='cpu')['model'])

    test_loss = 0
    for x in test_sampler:
        x = x.to(rank)
        logits = m(x[None, :-1])
        loss = F.cross_entropy(logits.transpose(2, 1), x[None, 1:])
        test_loss += loss
    test_loss /= len(test_sampler)

    print({
        'loss': test_loss.item(),
        'ppl': metrics.perplexity(test_loss).item()
    })
