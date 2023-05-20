import readline
import torch
import sys
from types import SimpleNamespace
from papote.bpe import BPE
from papote.model import Transformer, transformer_from_checkpoint
from papote.sampler import default_sampler


class ChatPrinter:

    def __init__(self, bpe, from_, to_):
        self.bpe = bpe
        self.history = ''
        self.from_ = from_
        self.to_ = to_

    def human_says(self, text):
        self.history += f'{self.from_}> {text}\n{self.to_}>'

    def print_stream(self, next_token):
        print(self.bpe.vocab[next_token].decode('utf-8', 'ignore'),
              end='',
              flush=True)
        self.history += self.bpe.vocab[next_token].decode('utf-8', 'ignore')
        if self.history.endswith(f'{self.from_}>'):
            self.history = self.history[:-len(f'{self.from_}>')]
            raise KeyboardInterrupt


def build_dataset(model, bpe, opts):
    printer = ChatPrinter(bpe, opts.from_, 'moi')
    sampler = default_sampler(model, bpe, **opts.__dict__)
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

    opts = SimpleNamespace(temperature=1,
                           top_k=500,
                           top_p=0.95,
                           typical_p=None,
                           sep='',
                           from_=sys.argv[2],
                           length=CTX)
    printer = ChatPrinter(bpe, opts.from_, 'moi')
    # Sample from the model
    sampler = default_sampler(model, bpe, **opts.__dict__)
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
                    printer.separator = args.strip()
                    opts.sep = printer.separator
                    print('Separator set to', printer.separator)
                elif command == 'length':
                    opts.length = int(args)
                    print('Length set to', opts.length)
                elif command == 'reset':
                    printer.history = ''
                elif command == 'history':
                    print('\n````')
                    print(printer.history)
                    print('````\n')
                print(f'{opts.from_}> ', end='')
                sampler = default_sampler(model, bpe, **opts.__dict__)
                continue
            printer.human_says(text)
            print(printer.to_ + '>', end='')
            out = sampler.sample(printer.history)
