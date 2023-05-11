import torch
import random
from bpe import BPE, Text
import os


class TextSampler:

    def __init__(self, text, num_tokens, bpe, dropout=0.0):
        self.text = text
        self.num_tokens = num_tokens
        self.bpe = bpe
        self.dropout = dropout

    def __len__(self):
        return len(self.text) // (10 * self.num_tokens)

    def __getitem__(self, i):
        start = random.randint(0, len(self.text) - self.num_tokens * 10)
        enc = self.bpe.encode_text(
            self.text[start:start + self.num_tokens * 10],
            dropout=self.dropout if random.uniform(0, 1) < 0.5 else 0)
        return torch.tensor(enc[:self.num_tokens], dtype=torch.long)


class EvalDirSampler:

    def __init__(self, directory, num_tokens, bpe):
        # recursively iterate over all files in directory
        text = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                fpath = os.path.join(root, file)
                with open(fpath, 'r', errors='ignore') as f:
                    text.append(f.read())

        text = '\n'.join(text)
        text = bpe.encode_text(text)
        self.samples = [
            text[i:i + num_tokens]
            for i in range(0,
                           len(text) - num_tokens, num_tokens // 2)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return torch.tensor(self.samples[i], dtype=torch.long)


class RandomCase:

    def __init__(self,
                 lower_token,
                 upper_token,
                 lowercase_p=0.25,
                 uppercase_p=0.25):
        self.lowercase_p = lowercase_p
        self.uppercase_p = uppercase_p
        self.lower_token = lower_token
        self.upper_token = upper_token

    def __call__(self, text):
        p = random.uniform(0, 1)
        if p < self.lowercase_p:
            text = self.lower_token + text.lower()
        elif p < self.lowercase_p + self.uppercase_p:
            text = self.upper_token + text.upper()
        return text


class RandomPromptSplit:

    def __init__(self, split_token, p=0.5):
        self.p = p
        self.split_token = split_token

    def __call__(self, text):
        if len(text) < 2:
            return text

        if random.uniform(0, 1) < self.p:
            split_point = random.randrange(0, len(text) - 1)
            text = text[:split_point] + self.split_token + text[split_point:]
        return text


class Tokenize:

    def __init__(self, bpe, dropout_token, dropout=0.1, dropout_p=0.5):
        self.bpe = bpe
        self.dropout = dropout
        self.dropout_p = dropout_p
        self.dropout_token = dropout_token

    def __call__(self, text):
        if random.uniform(0, 1) < self.dropout_p:
            text = self.dropout_token + text
            return self.bpe.encode_text(text, dropout=self.dropout)
        else:
            return self.bpe.encode_text(text)


class TextDirSampler:

    def __init__(self, directory, num_tokens, start_of_file_token, transform):
        # recursively iterate over all files in directory
        self.samples = []
        nchars = 0
        for root, dirs, files in os.walk(directory):
            for file in files:
                fpath = os.path.join(root, file)
                with open(fpath, 'r') as f:
                    txt = f.read()
                    nchars += len(txt)
                    self.samples.append((fpath, len(txt), nchars))

        self.num_tokens = num_tokens
        self.start_of_file_token = start_of_file_token
        self.transform = transform

    def __len__(self):
        return self.samples[-1][2] // self.num_tokens

    def _read_file(self, i):
        # find the file that contains the ith character
        idx = 0
        for sample in self.samples:
            if i < sample[1]:
                break
            i -= sample[1]
            idx += 1

        # read the file
        with open(self.samples[idx][0], 'r', errors='ignore') as f:
            f.seek(i)
            # we consider that a token is ~10 characters
            text = f.read(self.num_tokens * 10)
            if i == 0:
                text = self.start_of_file_token + text

        return self.transform(text)

    def __getitem__(self, i):
        enc = self._read_file(i * self.num_tokens +
                              random.randint(0, self.num_tokens))
        while len(enc) < self.num_tokens:
            enc += self._read_file(
                random.choice([s[2] for s in [(0, 0, 0)] + self.samples[:-1]]))
        return torch.tensor(enc[:self.num_tokens], dtype=torch.long)
