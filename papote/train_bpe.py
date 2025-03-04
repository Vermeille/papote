from papote.bpe import BPE

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default="bpe.json")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--data-path", type=str, default="data")
    args = parser.parse_args()

    test = "La cible effectue un jet de sauvegarde de Sagesse puis tombe inconsciente. Elle est également immunisée aux dégâts de foudre pour 1 tour."

    if os.path.exists(args.path):
        bpe = BPE.load(args.path)
    else:
        bpe = BPE()
    bpe.learn(
        args.data_path,
        target_vocab_size=args.vocab_size,
    )

    print(bpe.tokenize(test))
    bpe.save(args.path)
    bpe.load(args.path)
    print("ok")
