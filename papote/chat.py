from typing import Optional
import readline
import torch
import sys
from types import SimpleNamespace
from papote.bpe import BPE
from papote.model import Transformer, transformer_from_checkpoint
from papote.sampler import default_sampler
from papote.interactive import build_sampler as interactive_sampler, Printer
from papote.utils import OptionsBase


class ReversePrompt:

    def __init__(self, bpe, reverse_prompt):
        self.bpe = bpe
        self.reverse_prompt = reverse_prompt

    def __call__(self, encoded):
        text = self.bpe.decode_text(encoded)
        return text.endswith(self.reverse_prompt)


class Options(OptionsBase):
    sep: Optional[str] = ''
    temperature: float = 0.7
    top_k: int = 30
    top_p: float = 0.9
    typical_p: Optional[float] = None
    cfg: Optional[float] = 3
    repeat_penalty: float = 0.8
    repeat_window: int = 32
    length: int
    from_: str

    def __init__(self, from_, length):
        super().__init__()
        self.from_ = from_
        self.length = length


def build_sampler(model, bpe, opts):
    reverse_prompt = ReversePrompt(bpe, opts.pop('from_') + '>')
    sampler = interactive_sampler(model, bpe, **opts)
    sampler.stopping_criterion = reverse_prompt
    sampler.event_handler.print_prompt = False
    return sampler


if __name__ == '__main__':
    # Load the BPE
    checkpoint = torch.load(sys.argv[1], map_location='cpu')
    CTX = checkpoint['model']['_orig_mod.positional_embedding'].shape[1]

    bpe = BPE()
    bpe.load_state_dict(checkpoint['bpe'])

    # Load the model
    model = transformer_from_checkpoint(checkpoint)
    modelc = torch.compile(model)
    modelc.load_state_dict(checkpoint['model'])
    model.eval()
    del checkpoint

    opts = Options(sys.argv[2], 1024)
    # Sample from the model
    sampler = build_sampler(model, bpe, dict(opts.__dict__))
    history = f'{opts.from_}>'
    print(f'{opts.from_}> ', end='')
    to_ = 'moi' if len(sys.argv) < 4 else sys.argv[3]
    with torch.inference_mode():
        while True:
            text = input(' ').replace('\\n', '\n')
            if not text:
                break

            if text.startswith('!'):
                if text == '!exit':
                    sys.exit(0)
                elif text == '!reset':
                    history = ''
                elif command == '!history':
                    print('\n````')
                    print(history)
                    print('````\n')
                else:
                    try:
                        opts.parse(text[1:])
                    except Exception as e:
                        continue
                print(f'{opts.from_}> ', end='')
                sampler = build_sampler(model, bpe, dict(opts.__dict__))
                continue
            history += f' {text}\n{to_}>'
            print(f'{to_}>', end='')
            history = sampler.sample(history)
