from papote.bpe import BPE

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default="bpe.json")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--min-frequency", type=int, default=50)
    parser.add_argument("--data-path", type=str, default="data")
    args = parser.parse_args()

    test = "<|SOH|>La cible effectue un jet de sauvegarde de Sagesse puis tombe inconsciente. Elle est également immunisée aux dégâts de foudre pour 1 tour.<|EOT|><|NUL|>"

    if os.path.exists(args.path):
        bpe = BPE.load(args.path, writeable=True)
    else:
        bpe = BPE()
        bpe.learn(
            args.data_path,
            target_vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
        )

    print(bpe.tokenize(test))
    bpe.save(args.path)
    bpe.load(args.path)
    print("ok")
