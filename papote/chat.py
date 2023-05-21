import readline
import torch
import sys
from types import SimpleNamespace
from papote.bpe import BPE
from papote.model import Transformer, transformer_from_checkpoint
from papote.sampler import default_sampler
from papote.interactive import build_sampler as interactive_sampler, Printer


class ReversePrompt:

    def __init__(self, bpe, reverse_prompt):
        self.bpe = bpe
        self.reverse_prompt = reverse_prompt

    def __call__(self, encoded):
        text = self.bpe.decode_text(encoded)
        return text.endswith(self.reverse_prompt)


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

    opts = SimpleNamespace(temperature=0.6,
                           top_k=500,
                           top_p=0.95,
                           typical_p=None,
                           sep='',
                           from_=sys.argv[2],
                           length=CTX)
    # Sample from the model
    sampler = build_sampler(model, bpe, dict(opts.__dict__))
    history = f'{opts.from_}>'
    print(f'{opts.from_}> ', end='')
    with torch.inference_mode():
        while True:
            text = input(' ').replace('\\n', '\n')
            if not text:
                break

            if text.startswith('!'):
                try:
                    command, args = text[1:].split(' ', 1)
                except ValueError:
                    command, args = text[1:], ''
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
                elif command == 'reset':
                    history = ''
                elif command == 'history':
                    print('\n````')
                    print(history)
                    print('````\n')
                print(f'{opts.from_}> ', end='')
                sampler = default_sampler(model, bpe, **opts.__dict__)
                continue
            history += f' {text}\nmoi>'
            print('moi>', end='')
            history = sampler.sample(history)
