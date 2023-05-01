from bpe import BPE, Text

if __name__ == '__main__':
    from discordloader import discord_load
    import os
    text = discord_load('../gpt2-dnd/dataset.txt')
    if os.path.exists('bpe.json'):
        bpe = BPE.load('bpe.json')
    else:
        bpe = BPE()
    bpe.learn(text, target_vocab_size=4096, simultaneous_merges=10)
    test = Text(
        "La cible effectue un jet de sauvegarde de Sagesse puis tombe inconsciente. Elle est également immunisée aux dégâts de foudre pour 1 tour."
    )
    test.tokenize(bpe.merges)

    print(test.as_str_tokens(bpe.vocab))
    bpe.save('bpe.json')
    print('ok')
