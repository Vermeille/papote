import random
import json
import pyximport

pyximport.install(setup_args={"script_args": ['--cython-cplus']})
from text import Text


class BPE:
    NUL = 0
    SOH = 1
    STX = 2
    ETX = 3
    EOT = 4
    ENQ = 5
    ACK = 6
    BEL = 7
    BS = 8
    DC1 = 0x11
    DC2 = 0x12
    DC3 = 0x13
    DC4 = 0x14
    NAK = 0x15

    def __init__(self):
        self.vocab = [bytes([i]) for i in range(256)]
        self.merges = []
        self.special = []

    def learn(self, txt, target_vocab_size=1000, simultaneous_merges=1):
        merges = self.merges
        vocab = self.vocab

        text = Text(txt)
        text.tokenize(merges)

        while len(vocab) < target_vocab_size:
            all_pairs = text.most_frequent_pair()[1]
            all_pairs = sorted(all_pairs.keys(),
                               key=lambda x: all_pairs[x],
                               reverse=True)[:simultaneous_merges]
            print(all_pairs)
            for pair in all_pairs:
                if not text.merge(pair[0], pair[1], len(vocab) - 1):
                    continue
                merges.append(pair)
                print(len(vocab), vocab[merges[-1][0]], vocab[merges[-1][1]])
                vocab.append(vocab[merges[-1][0]] + vocab[merges[-1][1]])
        self.vocab = vocab
        self.merges = merges

    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump(
                {
                    'vocab': [v.decode('latin1') for v in self.vocab],
                    'merges': self.merges,
                    'special': self.special
                }, f)

    def add_special_token(self, token: str):
        self.special.append(token)
        self.vocab.append(bytes(token))
        self.merges((0, 0))

    @staticmethod
    def load(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        bpe = BPE()
        bpe.vocab = [v.encode('latin1') for v in data['vocab']]
        bpe.merges = data['merges']
        bpe.special = data.get('special')
        return bpe

    def tokenize(self, text, dropout=0.0):
        t = Text(text)
        t.tokenize(self.merges, dropout=dropout)
        return t.as_str_tokens(self.vocab)

    def encode(self, tokenized):
        vocab2idx = {v: i for i, v in enumerate(self.vocab)}
        return [vocab2idx[t] for t in tokenized]

    def decode(self, encoded):
        return [self.vocab[idx] for idx in encoded]

    def encode_text(self, text, dropout=0.0):
        t = Text(text)
        t.tokenize(self.merges, dropout=dropout)
        return t.as_tokens()

    def decode_text(self, encoded, separator=b''):
        return separator.join(self.decode(encoded)).decode('utf-8', 'replace')
