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
    def load(directory):
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

        if separator is None:
            return self.tokenizer.decode(ids)
        else:
            tokens = [self.tokenizer.decode([i]) for i in ids]
            return separator.join(tokens)

    def state_dict(self):
        import tempfile
        import json

        with tempfile.NamedTemporaryFile() as f:
            f.close()
            self.tokenizer.save(f.name)

            with open(f.name) as fopen:
                return json.load(fopen)

    def load_state_dict(self, state_dict):
        import tempfile
        import json

        with tempfile.NamedTemporaryFile() as f:
            f.close()
            with open(f.name, "w") as fopen:
                json.dump(state_dict, fopen)
            self.tokenizer = Tokenizer.from_file(f.name)

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
        ]
        if len(default_specials) != len(set(default_specials)):
            raise ValueError("Default special tokens must be unique")
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
