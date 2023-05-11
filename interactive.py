import readline
import torch
import sys
from bpe import BPE
from model import Transformer, make_transformer
from train import sample
from types import SimpleNamespace


class StreamPrinter:

    def __init__(self, bpe, separator=''):
        self.bpe = bpe
        self.separator = separator

    def print_seed(self, encoded):
        print(self.bpe.decode_text(encoded, self.separator.encode()),
              end=self.separator,
              flush=True)

    def print_stream(self, next_token):
        print(self.bpe.vocab[next_token].decode('utf-8', 'ignore'),
              end=self.separator,
              flush=True)


if __name__ == '__main__':
    # Load the BPE
    CTX = 640
    checkpoint = torch.load(sys.argv[1], map_location='cpu')

    bpe = BPE()
    bpe.load_state_dict(checkpoint['bpe'])

    # Load the model
    model = make_transformer('xxs', len(bpe.vocab), CTX)
    modelc = torch.compile(model)
    modelc.load_state_dict(checkpoint['model'])
    model.eval()
    del checkpoint

    opts = SimpleNamespace(temperature=0.7,
                           top_k=30,
                           top_p=0.9,
                           typical_p=None,
                           sep='',
                           length=CTX)
    printer = StreamPrinter(bpe, opts.sep)
    # Sample from the model
    with torch.inference_mode():
        while True:
            print(opts.__dict__)
            text = input('>>> ').replace('\\n', '\n')
            if not text:
                break

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
                    printer.separator = args.strip()
                    opts.sep = printer.separator
                    print('Separator set to', printer.separator)
                elif command == 'length':
                    opts.length = int(args)
                    print('Length set to', opts.length)
                continue
            out = sample(
                model,
                bpe,
                #chr(bpe.SOH) + text + chr(bpe.STX),
                text,
                CTX,
                num_tokens=opts.length,
                top_k=opts.top_k,
                top_p=opts.top_p,
                temperature=opts.temperature,
                typical_p=opts.typical_p,
                callback_seed=printer.print_seed,
                callback_stream=printer.print_stream)
            print()
