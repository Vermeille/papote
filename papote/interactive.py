import readline
import torch
import sys
from types import SimpleNamespace
from colored import fg, bg, attr
from papote.bpe import BPE
from papote.model import Transformer, transformer_from_checkpoint
import papote.sampler as S


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
    modelc = torch.compile(model)
    modelc.load_state_dict(checkpoint['model'])
    model.eval()
    del checkpoint

    opts = SimpleNamespace(temperature=0.7,
                           top_k=30,
                           top_p=0.9,
                           typical_p=None,
                           sep='',
                           length=CTX + 1)

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
                command, args = text[1:].split(' ', 1)
                if command == 'temperature':
                    opts.temperature = float(args)
                    print('Temperature set to', opts.temperature)
                elif command == 'top_k':
                    opts.top_k = int(args)
                    print('Top-k set to', opts.top_k)
                elif command == 'top_p':
                    opts.top_p = float(args)
                    print('Top-p set to', opts.top_p)
                elif command == 'typical_p':
                    opts.typical_p = float(args) if args != 'None' else None
                    print('Typical-p set to', opts.typical_p)
                elif command == 'exit':
                    sys.exit(0)
                elif command == 'load':
                    modelc.load_state_dict(
                        torch.load(args, map_location='cpu')['model'])
                elif command == 'sep':
                    opts.sep = args.strip()
                    print('Separator set to', opts.sep)
                elif command == 'length':
                    opts.length = int(args)
                    print('Length set to', opts.length)
                sampler = build_sampler(model, bpe, **opts.__dict__)
                continue

            out = sampler.sample(text)
            print()
