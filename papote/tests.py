from papote.data_utils import TextDirSampler, Tokenize
from papote.bpe import BPE, Text


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
    import papote.data_utils as data
    from torchvision.transforms import Compose
    bpe = BPE.load('bpe.json')
    sampler = data.TextDirSampler(
        'data', 64 + 1, bpe.unicode_for_special('<|SOH|>'),
        Compose([
            data.Tokenize(bpe,
                          bpe.unicode_for_special('<|DC1|>'),
                          dropout=0.1,
                          dropout_p=0.05)
        ]))

    for i in range(10):
        print(repr(bpe.decode_text(sampler[i], b',')))


def test_simple_fim():
    from papote.data_utils import FillInTheMiddle
    bpe = BPE.load('bpe.json.bkp')
    bpe.add_special('<|SUFFIX|>', bpe.specials['<|NAK|>'])
    bpe.add_special('<|PREFIX|>', bpe.specials['<|SYN|>'])
    bpe.add_special('<|WRAP|>', bpe.specials['<|ETB|>'])
    text = bpe.encode_text(
        "Je suis un petit canard qui fait coin coin. Je patauge dans la mare "
        "et je suis vivant. Un jour j'ai rencontré un autre canard qui était "
        "très méchant. Il m'a dit que j'étais un vilain petit canard et que "
        "je devais partir. Alors je suis parti et j'ai rencontré un chasseur "
        "qui m'a collectionné. Fin.")
    print(
        bpe.decode_text(
            FillInTheMiddle(bpe.specials['<|SUFFIX|>'],
                            bpe.specials['<|PREFIX|>'],
                            bpe.specials['<|WRAP|>'],
                            p=1)(text)))


if __name__ == '__main__':
    #test_sampler()
    #test_bpe_dropout()
    #test_special_tokens()
    #test_random_augmentations()
    #test_simple_fim()
    import papote.data_utils as data
    from torchvision.transforms import Compose
    bpe = BPE.load('bpe.json.bkp')
    bpe.add_special('<|SUFFIX|>', bpe.specials['<|NAK|>'])
    bpe.add_special('<|PREFIX|>', bpe.specials['<|SYN|>'])
    bpe.add_special('<|WRAP|>', bpe.specials['<|ETB|>'])
    sampler = data.TextDirSampler(
        'data/raw/x', 64 + 1, bpe.unicode_for_special('<|SOH|>'),
        Compose([
            data.Tokenize(bpe,
                          bpe.unicode_for_special('<|DC1|>'),
                          dropout_p=0.0),
            data.Crop(64 + 1),
            data.FillInTheMiddle(bpe.specials['<|SUFFIX|>'],
                                 bpe.specials['<|PREFIX|>'],
                                 bpe.specials['<|WRAP|>'],
                                 p=1),
        ]))

    print(sampler[0][0])
    print(bpe.decode_text(sampler[0][0]))
