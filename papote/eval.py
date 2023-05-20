import torch
import torch.nn.functional as F
from papote.bpe import BPE
from papote.data_utils import EvalDirSampler
from papote.model import make_transformer
import papote.metrics as metrics

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
