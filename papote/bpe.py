import os
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from papote.utils import txt_extensions
import torch


class BPE:
    def __init__(self):
        self.tokenizer = ByteLevelBPETokenizer(unicode_normalizer="nfkc")
        self.specials = {}

    @property
    def vocab(self):
        return self.tokenizer.get_vocab()

    @staticmethod
    def load(directory, writeable=False):
        self = BPE()
        self.tokenizer = Tokenizer.from_file(directory)
        return self

    def tokenize(self, text):
        ids = self.tokenizer.encode(text).ids
        return self.tokenizer.decode_batch([[i] for i in ids])

    def encode_text(self, text):
        return self.tokenizer.encode(text).ids

    def decode_text(self, ids, separator=None):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        text = self.tokenizer.decode(ids)
        return text if separator is None else text

    def state_dict(self):
        return {
            "vocab": self.tokenizer.get_vocab(),
            "merges": self.tokenizer.get_merges(),
        }

    def save(self, directory):
        self.tokenizer.save(directory)

    def learn(self, directory, target_vocab_size=4096, min_frequency=2):
        """
        Mimics the custom BPE.learn() API.
        Gathers all files in the given directory (recursively) and trains the tokenizer.
        The extra parameters (simultaneous_merges, num_threads, merge_all) are ignored.
        """
        files = []
        for root, _, filenames in os.walk(directory):
            for file in filenames:
                if not any(file.endswith(ext) for ext in txt_extensions):
                    print(f"Skipping {file} as it is not a text file.")
                    continue
                files.append(os.path.join(root, file))
        # Use a default list of special tokens similar to your original code.
        default_specials = [
            "<|NUL|>",
            "<|SOH|>",
            "<|STX|>",
            "<|ETX|>",
            "<|EOT|>",
            "<|ENQ|>",
            "<|ACK|>",
            "<|BEL|>",
            "<|BS|>",
            "<|HT|>",
            "<|EOT|>",
        ]
        print(f"Training on {len(files)} files.")
        self.tokenizer.train(
            files=files,
            vocab_size=target_vocab_size,
            min_frequency=min_frequency,
            special_tokens=default_specials,
        )
        # Record the special token ids into our own dictionary.
        for token in default_specials:
            self.specials[token] = self.tokenizer.get_vocab().get(token, None)

    def token_to_id(self, token):
        return self.tokenizer.get_vocab().get(token, None)

    def add_special(self, special: str):
        """
        Adds a special token.
        """
        if special not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens([special])
        self.specials[special] = self.tokenizer.get_vocab()[special]

