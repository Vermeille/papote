import readline
import torch
import sys
from bpe import BPE
from model import Transformer, make_transformer
from train import sample
from types import SimpleNamespace

if __name__ == '__main__':
    # Load the BPE
    bpe = BPE.load('bpe.json')
    CTX = 640
    # Load the model
    model = make_transformer('xxs', len(bpe.vocab), CTX)
    modelc = torch.compile(model)
    modelc.load_state_dict(
        torch.load(sys.argv[1], map_location='cpu')['model'])

    opts = SimpleNamespace(
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        typical_p=None,
        sep='',
    )
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
                    opts.sep = args.strip()
                    print('Separator set to', opts.sep)
                continue
            out = sample(
                model,
                bpe,
                #chr(bpe.SOH) + text + chr(bpe.STX),
                text,
                CTX,
                num_tokens=CTX,
                top_k=opts.top_k,
                top_p=opts.top_p,
                temperature=opts.temperature,
                typical_p=opts.typical_p,
                print_as_you_go=True,
                sep=opts.sep)
            print()
            print(out)
