import time
import os
from collections import Counter
import random
import json
from multiprocessing import Pool
import pyximport

pyximport.install(setup_args={"script_args": ['--cython-cplus']})
from papote.text import Text


def clean_private_unicode(text):
    return ''.join([c for c in text if not 0xE000 <= ord(c) <= 0xF8FF])


def is_valid_merge(a, b):
    token = a + b
    special = b'.,;:?!()[]{}"<>|/@\\^$*+-=_~`#%&'
    return ((b[0] != ord(' ') or all(c == ord(' ') for c in a))
            and (all(c not in token
                     for c in special + b'\n') or all(c in (special + b' ')
                                                      for c in token)))


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

    def __len__(self):
        return len(self.vocab)

    @staticmethod
    def is_merge_conflicting(this_merge, previous_merges):
        for merge in previous_merges:
            if merge[1] == this_merge[0] or merge[0] == this_merge[1]:
                return True
        return False

    @staticmethod
    def _learn_from_files(args):
        cnt = Counter()
        filenames, merges, vocab = args
        for filename in filenames:
            #print('starting', filename)
            with open(filename, 'r', errors='ignore') as f:
                text = Text(f.read())
            text.fast_tokenize(merges)
            pairs = text.most_frequent_pair()[1]

            for merge, count in pairs.items():
                if is_valid_merge(vocab[merge[0]], vocab[merge[1]]):
                    cnt[merge] += count

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
                    ((chunk, merges, vocab)
                     for chunk in chunks([file
                                          for file in files], num_threads))):
                    all_pairs.update(pairs)

                all_pairs = sorted(all_pairs.keys(),
                                   key=lambda x: all_pairs[x],
                                   reverse=True)
                num_new_tokens = 0
                for pair in all_pairs:
                    if (num_new_tokens >= simultaneous_merges
                            or len(vocab) >= target_vocab_size):
                        break
                    if self.is_merge_conflicting(
                            pair, merges[len(merges) - num_new_tokens:]):
                        print('conflict:',
                              ' [',
                              vocab[pair[0]].decode(errors='replace'),
                              '::',
                              vocab[pair[1]].decode(errors='replace'),
                              ']',
                              sep='')
                        continue
                    print(len(merges),
                          ' [',
                          vocab[pair[0]].decode(errors='replace'),
                          '::',
                          vocab[pair[1]].decode(errors='replace'),
                          ']',
                          sep='')
                    merges.append(pair)
                    new_token = vocab[pair[0]] + vocab[pair[1]]
                    vocab.append(new_token)
                    num_new_tokens += 1
                print('time:', time.time() - start_time)
        self.vocab = vocab
        self.merges = merges

    def unicode_for_special(self, token):
        return chr(self.specials[token] +
                   (0 if self.specials[token] < 256 else 0xE000))

    def state_dict(self):
        return {
            'vocab': [v.decode('latin1') for v in self.vocab],
            'merges': self.merges,
            'specials': self.specials
        }

    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.state_dict(), f)

    def load_state_dict(self, state_dict):
        self.vocab = [v.encode('latin1') for v in state_dict['vocab']]
        self.merges = state_dict['merges']
        self.specials = state_dict['specials']

    @staticmethod
    def load(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        bpe = BPE()
        bpe.load_state_dict(data)
        return bpe

    def add_special(self, special: str, idx=None):
        if idx is None:
            self.specials[special] = len(self.vocab)
            self.vocab.append(special.encode())
            self.merges.append((-1, -1))
        else:
            self.specials[special] = idx

    def process_special(self, txt):
        for special, idx in self.specials.items():
            txt = txt.replace(special, chr(idx + (0 if idx < 256 else 0xE000)))
        return txt

    def tokenize(self, text, dropout=0.0):
        text = self.process_special(text)
        t = Text(text)
        t.unicode_private_to_token()
        t.fast_tokenize(self.merges, dropout=dropout)
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
        t.fast_tokenize(self.merges, dropout=dropout)
        return t.as_tokens()

    def decode_text(self, encoded, separator=b''):
        return separator.join(self.decode(encoded)).decode('utf-8', 'replace')


class ThinBPE:

    def __init__(self, bpe: BPE):
        self.bpe = bpe
        self.from_bpe = []
        self.to_bpe = []

    @staticmethod
    def _count_in_file(args):
        cnt = Counter()
        filenames, merges = args
        for filename in filenames:
            #print('starting', filename)
            with open(filename, 'r', errors='ignore') as f:
                text = Text(f.read())
            text.fast_tokenize(merges)
            cnt.update(Counter(text.as_tokens()))

        return dict(cnt.most_common(len(cnt)))

    def clean(self, directory, min_count=200, num_threads=8):
        files = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(directory) for file in files
        ]
        random.shuffle(files)
        with Pool(num_threads) as pool:
            start_time = time.time()
            all_counts = Counter()
            for counts in pool.imap_unordered(
                    self._count_in_file,
                ((chunk, self.bpe.merges)
                 for chunk in chunks([file for file in files], num_threads))):
                all_counts.update(counts)

        dead_tokens = {
            i
            for i in range(len(self.bpe.vocab))
            if all_counts[i] <= min_count and i >= 256
        }
        return dead_tokens

    def learn(self,
              directory,
              target_vocab_size=1000,
              simultaneous_merges=1,
              min_count=200,
              num_threads=8):
        dead_tokens = self.clean(directory, min_count, num_threads)
        while len(self.bpe.vocab) - len(dead_tokens) < target_vocab_size:
            print('#tokens:', len(self.bpe.vocab), '#dead:', len(dead_tokens),
                  ', learning',
                  target_vocab_size - len(self.bpe.vocab) + len(dead_tokens),
                  'new tokens')
            self.bpe.learn(directory, target_vocab_size + len(dead_tokens),
                           simultaneous_merges, num_threads)
            dead_tokens = self.clean(directory, min_count, num_threads)
        self.dead_tokens = dead_tokens

    def gen_mapping(self):
        self.from_bpe = []
        self.to_bpe = []
        i = 0
        for idx, token in enumerate(self.bpe.vocab):
            if idx not in self.dead_tokens:
                self.to_bpe.append(idx)
                self.from_bpe.append(i)
                i += 1
            else:
                self.from_bpe.append(-1)

    def __len__(self):
        return len(self.to_bpe)

    def save(self, filename):
        state = self.bpe.state_dict()
        state['dead_tokens'] = list(self.dead_tokens)
        with open(filename, 'w') as f:
            json.dump(state, f)

    def load_state_dict(self, state_dict):
        self.bpe.load_state_dict(state_dict)
        self.dead_tokens = set(state_dict['dead_tokens'])
        self.gen_mapping()

    @staticmethod
    def load(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        bpe = ThinBPE(BPE())
        bpe.load_state_dict(data)
        return bpe

    def encode_text(self, text, dropout=0.0):
        return [
            self.from_bpe[i]
            for i in self.bpe.encode_text(text, dropout=dropout)
        ]

    def decode_text(self, encoded, separator=b''):
        return self.bpe.decode_text([self.to_bpe[i] for i in encoded],
                                    separator=separator)
