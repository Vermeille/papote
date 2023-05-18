import readline
import torch
import sys
from bpe import BPE
from model import Transformer, transformer_from_checkpoint
from train import sample
from types import SimpleNamespace
import sampler as S


class Printer:

    def __init__(self, bpe, separator=''):
        self.bpe = bpe
        self.separator = separator

    def __call__(self, prompt, next_token):
        if not next_token:
            print(self.bpe.decode_text(prompt, self.separator.encode()),
                  end=self.separator,
                  flush=True)
        else:
            print(self.bpe.vocab[next_token].decode('utf-8', 'ignore'),
                  end=self.separator,
                  flush=True)


def build_sampler(model, bpe, opts):
    return S.Sampler(model,
                     bpe,
                     logits_policy=S.LogitsComposite(
                         S.TopK(opts.top_k), S.TopP(opts.top_p),
                         S.FixedRepetitionPenalty(0.8, 32),
                         S.Typical(opts.typical_p),
                         S.Temperature(opts.temperature)),
                     stopping_criterion=S.StopTooLong(opts.length),
                     event_handler=Printer(bpe, opts.sep),
                     prompt_processor=S.CleanThink(5))


if __name__ == '__main__':
    # Load the BPE
    checkpoint = torch.load(sys.argv[1], map_location='cpu')
    CTX = checkpoint['model']['_orig_mod.positional_embedding'].shape[1]

    bpe = BPE()
    bpe.load_state_dict(checkpoint['bpe'])

    # Load the model
    model = transformer_from_checkpoint(checkpoint['model'])
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

    sampler = build_sampler(model, bpe, opts)
    # Sample from the model
    with torch.inference_mode():
        while True:
            print(opts.__dict__)
            text = input('>>> ')
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
                sampler = build_sampler(model, bpe, opts)
                continue

            out = sampler.sample(text)
            print()
