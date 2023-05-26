from typing import Optional
import readline
import torch
import sys
from types import SimpleNamespace
from colored import fg, bg, attr
from papote.bpe import BPE
from papote.model import Transformer, transformer_from_checkpoint
import papote.sampler as S
from papote.utils import OptionsBase


class Printer:

    colors = [
        'red',
        'orange_red_1',
        'dark_orange',
        'orange_1',
        'yellow',
        'light_green_2',
    ]

    def __init__(self, bpe, separator='', print_prompt=True):
        self.bpe = bpe
        self.separator = separator
        self.print_prompt = print_prompt

    def __call__(self, prompt, next_token, prob, logit):
        if not next_token:
            if self.print_prompt:
                print(self.bpe.decode_text(prompt, self.separator.encode()),
                      end=self.separator,
                      flush=True)
        else:
            color = self.colors[int(min(prob, 0.99) * len(self.colors))]
            print(fg(color) +
                  self.bpe.vocab[next_token].decode('utf-8', 'ignore') +
                  attr('reset'),
                  end=self.separator,
                  flush=True)


class Options(OptionsBase):
    sep: Optional[str] = ''
    temperature: float = 0.7
    top_k: int = 100
    top_p: float = 0.95
    typical_p: Optional[float] = None
    cfg: Optional[float] = 5
    repeat_penalty: float = 0.9
    repeat_window: int = 16
    length: int

    def __init__(self, length):
        super().__init__()
        self.length = length


def build_sampler(model, bpe, **kwargs):
    sep = kwargs.pop('sep', '')
    sampler = S.default_sampler(model, bpe, **kwargs)
    sampler.event_handler = Printer(bpe, sep)
    return sampler


if __name__ == '__main__':
    # Load the BPE
    checkpoint = torch.load(sys.argv[1], map_location='cpu')
    CTX = checkpoint['model']['_orig_mod.positional_embedding'].shape[1]

    bpe = BPE()
    bpe.load_state_dict(checkpoint['bpe'])

    # Load the model
    model = transformer_from_checkpoint(checkpoint)
    model.eval()
    modelc = torch.compile(model)
    modelc.load_state_dict(checkpoint['model'])
    del checkpoint

    opts = Options(1024)
    sampler = build_sampler(model, bpe, **opts.__dict__)
    # Sample from the model
    with torch.inference_mode():
        while True:
            print(opts.__dict__)
            try:
                text = input('>>> ')
            except EOFError:
                sys.exit(0)

            if not text:
                continue
            text = text.replace('\\n', '\n')

            if text.startswith('!'):
                try:
                    opts.parse(text[1:])
                except Exception as e:
                    continue
                sampler = build_sampler(model, bpe, **opts.__dict__)
                continue

            out = sampler.sample(text)
            print()
