from data_utils import TextDirSampler, Tokenize
from bpe import BPE, Text


def test_sampler():
    bpe = BPE()
    sampler = TextDirSampler('data/5e-drs/docs/bestiaire/',
                             512 * 2,
                             '<|NUL|>',
                             transform=Tokenize(bpe, '<|DC1|>', dropout_p=0))
    print(bpe.decode_text(sampler[4]))


def test_bpe_dropout():
    for _ in range(10):
        bpe = BPE.load('bpe.json')
        t = bpe.encode_text(
            "La cible effectue un jet de sauvegarde de Sagesse puis tombe inconsciente. Elle est également immunisée aux dégâts de foudre pour 1 tour.",
            dropout=0.5)

        print(bpe.decode_text(t, b','))


def test_special_tokens():
    bpe = BPE()
    bpe.add_special('<|ZOB|>')
    print(len(bpe.vocab), bpe.vocab[-1])
    print(bpe.encode_text('a<|ZOB|>b'))


def test_random_augmentations():
    import data_utils as data
    from torchvision.transforms import Compose
    bpe = BPE.load('bpe.json')
    sampler = data.TextDirSampler(
        'data', 64 + 1, bpe.unicode_for_special('<|SOH|>'),
        Compose([
            data.RandomCase(bpe.unicode_for_special('<|DC2|>'),
                            bpe.unicode_for_special('<|DC3|>'),
                            uppercase_p=0.0,
                            lowercase_p=0.05),
            data.RandomPromptSplit(bpe.unicode_for_special('<|STX|>'), p=0.05),
            data.Tokenize(bpe,
                          bpe.unicode_for_special('<|DC1|>'),
                          dropout=0.1,
                          dropout_p=0.05)
        ]))

    for i in range(10):
        print(repr(bpe.decode_text(sampler[i], b',')))


if __name__ == '__main__':
    #test_sampler()
    #test_bpe_dropout()
    #test_special_tokens()
    #test_random_augmentations()
    pass
