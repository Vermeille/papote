import pyximport

pyximport.install(setup_args={
    "script_args": ['--cython-cplus'],
    'extra_compile_args': ['-std=c++20'],
})

from tokenizer import Merges, Vocab, TextData

import json


def main():
    merges = Merges.from_list([(1, 2), (3, 4)])
    print(merges)
    print(merges.to_list())
    vocab = Vocab.from_list([b'a', b'b', b'c', b'd'])
    print(vocab)
    print(vocab.to_list())

    bpe = json.load(open('./general-no-break.bpe.2048'))
    vocab = Vocab.from_list([v.encode('latin1') for v in bpe['vocab']])
    merges = Merges.from_list(bpe['merges'])

    data = open(
        '/home/vermeille/papote/data/messenger/inbox/aureliesanchezteri_10206078206889460/message_2.json'
    ).read()
    text = TextData(data)
    text.fast_tokenize(merges, None)
    out = text.as_str(vocab).decode('utf-8')
    assert data == out


if __name__ == '__main__':
    main()
