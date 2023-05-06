import time
import os
from collections import Counter
import random
import json
from multiprocessing import Pool
import pyximport

pyximport.install(setup_args={"script_args": ['--cython-cplus']})
from text import Text


def chunks(lst, n):
    """Yield n successive chunks from lst."""
    chunk_size = (len(lst) + n - 1) // n
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


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
    HT = 9
    LF = 10
    DLE = 0x10
    DC1 = 0x11
    DC2 = 0x12
    DC3 = 0x13
    DC4 = 0x14
    NAK = 0x15

    def __init__(self):
        self.vocab = [bytes([i]) for i in range(256)]
        self.merges = [(-1, -1)] * 256
        self.specials = {
            '<|NUL|>': 0,
            '<|SOH|>': 1,
            '<|STX|>': 2,
            '<|ETX|>': 3,
            '<|EOT|>': 4,
            '<|ENQ|>': 5,
            '<|ACK|>': 6,
            '<|BEL|>': 7,
            '<|BS|>': 8,
            #('<|HT|>': 9, \t
            #('<|LF|>': 10, \n
            '<|VT|>': 11,
            '<|FF|>': 12,
            '<|CR|>': 13,
            '<|SO|>': 14,
            '<|SI|>': 15,
            '<|DLE|>': 16,
            '<|DC1|>': 17,
            '<|DC2|>': 18,
            '<|DC3|>': 19,
            '<|DC4|>': 20,
            '<|NAK|>': 21,
            '<|SYN|>': 22,
            '<|ETB|>': 23,
            '<|CAN|>': 24,
            '<|EM|>': 25,
            '<|SUB|>': 26,
            '<|ESC|>': 27,
            '<|FS|>': 28,
            '<|GS|>': 29,
            '<|RS|>': 30,
            '<|US|>': 31,
        }
        for s, idx in self.specials.items():
            self.vocab[idx] = s.encode('utf-8')

    @staticmethod
    def _learn_from_files(args):
        cnt = Counter()
        filenames, merges = args
        start_time = time.time()
        for filename in filenames:
            #print('starting', filename)
            with open(filename, 'r', errors='ignore') as f:
                text = Text(f.read())
            text.fast_tokenize(merges)
            pairs = text.most_frequent_pair()[1]
            cnt.update(pairs)
        return dict(cnt.most_common(len(cnt) // 10))

    def learn(self,
              directory,
              target_vocab_size=1000,
              simultaneous_merges=1,
              num_threads=8):
        merges = self.merges
        vocab = self.vocab

        files = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(directory) for file in files
        ]
        random.shuffle(files)
        with Pool(num_threads) as pool:
            while len(vocab) < target_vocab_size:
                start_time = time.time()
                all_pairs = Counter()
                for pairs in pool.imap_unordered(
                        self._learn_from_files,
                    ((chunk, merges)
                     for chunk in chunks([file
                                          for file in files], num_threads))):
                    all_pairs.update(pairs)

                all_pairs = sorted(all_pairs.keys(),
                                   key=lambda x: all_pairs[x],
                                   reverse=True)[:simultaneous_merges]
                print(all_pairs)
                for pair in all_pairs:
                    merges.append(pair)
                    print(len(vocab), vocab[merges[-1][0]],
                          vocab[merges[-1][1]])
                    vocab.append(vocab[merges[-1][0]] + vocab[merges[-1][1]])
                print('time:', time.time() - start_time)
        self.vocab = vocab
        self.merges = merges

    def unicode_for_special(self, token):
        return chr(self.specials[token] +
                   (0 if self.specials[token] < 256 else 0xE000))

    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump(
                {
                    'vocab': [v.decode('latin1') for v in self.vocab],
                    'merges': self.merges,
                    'specials': self.specials
                }, f)

    @staticmethod
    def load(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        bpe = BPE()
        bpe.vocab = [v.encode('latin1') for v in data['vocab']]
        bpe.merges = data['merges']
        bpe.specials = data['specials']
        return bpe

    def add_special(self, special):
        self.specials[special] = len(self.vocab)
        self.vocab.append(special)
        self.merges.append((-1, -1))

    def process_special(self, txt):
        for special, idx in self.specials.items():
            txt = txt.replace(special, chr(idx + (0 if idx < 256 else 0xE000)))
        return txt

    def tokenize(self, text, dropout=0.0):
        text = self.process_special(text)
        t = Text(text)
        t.unicode_private_to_token()
        t.tokenize(self.merges, dropout=dropout)
        return t.as_str_tokens(self.vocab)

    def encode(self, tokenized):
        vocab2idx = {v: i for i, v in enumerate(self.vocab)}
        return [vocab2idx[t] for t in tokenized]

    def decode(self, encoded):
        return [self.vocab[idx] for idx in encoded]

    def encode_text(self, text, dropout=0.0):
        text = self.process_special(text)
        t = Text(text)
        t.unicode_private_to_token()
        t.tokenize(self.merges, dropout=dropout)
        return t.as_tokens()

    def decode_text(self, encoded, separator=b''):
        return separator.join(self.decode(encoded)).decode('utf-8', 'replace')
