import tqdm
import torch
import random
import os
import torch.nn.functional as F
from torch.utils.data import IterableDataset
import ftfy
from papote.utils import txt_extensions


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

    def __call__(self, text):
        if random.uniform(0, 1) < self.dropout_p:
            text = self.dropout_token + text
            return self.bpe.encode_text(text, dropout=self.dropout)
        else:
            return self.bpe.encode_text(text + '<|EOT|>')


class Crop:

    def __init__(self, max_tokens):
        self.max_tokens = max_tokens

    def __call__(self, text):
        if len(text) > self.max_tokens:
            text = text[:self.max_tokens]
        return text


class FillInTheMiddle:

    def __init__(self, suffix, prefix, wrap, p=0.5):
        self.p = p
        self.suffix = suffix
        self.prefix = prefix
        self.wrap = wrap

    def __call__(self, text):
        if not random.uniform(0, 1) < self.p or len(text) <= 6:
            return text
        text = text[:-3]
        split_point = random.randrange(0, len(text) - 1)
        text = ([self.suffix] + text[split_point:] + [self.prefix] +
                text[:split_point] + [self.wrap])
        return text


def tokens_dropout(tokens, mask_id, dropout_p):
    return torch.where(
        torch.rand(*tokens.shape, device=tokens.device) < dropout_p, mask_id,
        tokens)


class NextTokenObjective:

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            return x[:-1], x[1:].clone()
        else:
            return x[:-1], list(x[1:])


class ChunkSampler(IterableDataset):

    def __init__(self,
                 directory,
                 num_tokens,
                 start_of_file_token,
                 transform,
                 to_input_and_target=NextTokenObjective()):
        self.num_tokens = num_tokens
        self.start_of_file_token = start_of_file_token
        self.transform = transform
        self.to_input_and_target = to_input_and_target

        # load list of files to ignore from directory/ignore.txt
        ignore_path = os.path.join(directory, 'ignore.txt')
        ignore = set()
        if os.path.exists(ignore_path):
            with open(ignore_path, 'r') as f:
                ignore = set(f.read().split('\n'))
        # recursively iterate over all files in directory
        self.files = []
        nchars = 0
        for root, dirs, files in os.walk(directory):
            for file in files:
                fpath = os.path.join(root, file)
                if not any(file.lower().endswith(ext) for ext in txt_extensions):
                    continue
                if fpath in ignore:
                    print('ignoring', fpath)
                    continue
                self.files.append(fpath)

        random.shuffle(self.files)
        print('found', len(self.files), 'files')

    def __iter__(self):
        return self.generate()

    def generate(self):
        worker = torch.utils.data.get_worker_info()
        progress = (tqdm.tqdm if worker is None or worker.id == 0 else
                    (lambda x: x))
        for file in progress(self.files):
            with open(file, 'r', errors='ignore') as f:
                content = self.start_of_file_token + f.read()
            tokens = self.transform(content)
            for i in range(0, len(tokens), self.num_tokens):
                out = self.to_input_and_target(
                    torch.tensor(tokens[i:i + self.num_tokens],
                                 dtype=torch.long))
                yield out


class Tagger:

    def __init__(self, tag_file, max_tags=4, prefix=''):
        self.max_tags = max_tags
        self.path2tags = {}
        with open(tag_file, 'r') as f:
            for line in f:
                path, *tags = line.split()
                self.path2tags[prefix + path] = tags

    def tags(self):
        return set(t for path_tags in self.path2tags.values()
                   for t in path_tags)

    def tag_path(self, path):
        tags = []
        components = path.split('/')
        while len(components) > 0:
            tag = self.path2tags.get('/'.join(components))
            if tag is not None:
                tags += tag
            components = components[:-1]

        if self.max_tags is not None:
            tags = tags[:self.max_tags]
            tags = tags + ['<|NUL|>'] * (self.max_tags - len(tags))
        random.shuffle(tags)
        return ''.join(tags)


class Pad:

    def __init__(self, ctx, pad_id):
        self.ctx = ctx
        self.pad_id = pad_id

    def __call__(self, data):
        return data + [self.pad_id] * max(0, self.ctx - len(data))


class Align:

    def __init__(self, ctx, pad_id):
        self.ctx = ctx
        self.pad_id = pad_id

    def __call__(self, data):
        # align data to a multiple of ctx tokens by adding padding
        if len(data) % self.ctx != 0:
            data += [self.pad_id] * (self.ctx - len(data) % self.ctx)
        return data


class SeqWeightedLoss(torch.nn.Module):

    def __init__(self, beta=0.999, loss_fn=F.cross_entropy):
        super().__init__()
        self.beta = beta
        self.loss_fn = loss_fn

    def forward(self, x, y, mask, reduction='none'):
        loss = self.loss_fn(x, y, reduction=reduction)
        with torch.no_grad():
            w = (loss * mask).sum(0) / mask.sum(0)
            w /= w.median()
            if not hasattr(self, 'weight'):
                self.register_buffer('weight', w.detach())
            self.weight = self.beta * self.weight + (1 - self.beta) * w
        return loss / self.weight


@torch.jit.script
def binary_entropy(logits, labels, reduction: str):
    # make one hot from labels
    labels = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1.0)
    return F.binary_cross_entropy_with_logits(logits,
                                              labels,
                                              reduction=reduction).sum(1)


class NFKC:

    def __call__(self, text):
        return ftfy.fix_text(text, normalization='NFKC')
