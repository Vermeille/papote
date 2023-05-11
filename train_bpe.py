from bpe import BPE, Text

if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default='bpe.json')
    parser.add_argument('--vocab-size', type=int, default=4096)
    args = parser.parse_args()

    test = Text(
        "La cible effectue un jet de sauvegarde de Sagesse puis tombe inconsciente. Elle est également immunisée aux dégâts de foudre pour 1 tour."
    )

    if os.path.exists(args.path):
        bpe = BPE.load(args.path)
    else:
        bpe = BPE()
    bpe.learn('data',
              target_vocab_size=args.vocab_size,
              simultaneous_merges=10,
              num_threads=16)
    test.tokenize(bpe.merges)

    print(test.as_str_tokens(bpe.vocab))
    bpe.save(args.path)
    print('ok')
