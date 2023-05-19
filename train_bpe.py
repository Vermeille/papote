from bpe import BPE, Text, ThinBPE

if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default='bpe.json')
    parser.add_argument('--vocab-size', type=int, default=4096)
    parser.add_argument('--thin', action='store_true')
    parser.add_argument('--data-path', type=str, default='data')
    args = parser.parse_args()

    test = Text(
        "La cible effectue un jet de sauvegarde de Sagesse puis tombe inconsciente. Elle est également immunisée aux dégâts de foudre pour 1 tour."
    )

    if os.path.exists(args.path):
        bpe = BPE.load(args.path)
    else:
        bpe = BPE()
    if not args.thin:
        bpe.learn(args.data_path,
                  target_vocab_size=args.vocab_size,
                  simultaneous_merges=10,
                  num_threads=16)
        test.tokenize(bpe.merges)

        print(test.as_str_tokens(bpe.vocab))
        bpe.save(args.path)
        print('ok')
    else:
        bpe = ThinBPE(bpe)
        bpe.learn(args.data_path,
                  target_vocab_size=args.vocab_size,
                  simultaneous_merges=10,
                  num_threads=16,
                  min_count=1000)
        bpe.save(args.path)
