import os
from tokenizers import ByteLevelBPETokenizer
from tokenizers import normalizers
from tokenizers.normalizers import NFKC


class BPE:
    def __init__(self):
        self.tokenizer = ByteLevelBPETokenizer()
        self.tokenizer.normalizer = NFKC()
        self.specials = {}

    @staticmethod
    def load(directory):
        self = BPE()
        self.tokenizer = ByteLevelBPETokenizer.from_file(
            directory + "/vocab.json", directory + "/merges.txt"
        )
        self.tokenizer.normalizer = NFKC()
        return self

    def tokenize(self, text):
        ids = self.tokenizer.encode(text).ids
        return self.tokenizer.decode_batch([[i] for i in ids])

    def encode_text(self, text):
        return self.tokenizer.encode(text).ids

    def decode_text(self, ids):
        return self.tokenizer.decode(ids)

    def state_dict(self):
        return {
            "vocab": self.tokenizer.get_vocab(),
            "merges": self.tokenizer.get_merges(),
        }

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        self.tokenizer.save_model(directory)

    def learn(self, directory, target_vocab_size=4096, min_frequency=2):
        """
        Mimics the custom BPE.learn() API.
        Gathers all files in the given directory (recursively) and trains the tokenizer.
        The extra parameters (simultaneous_merges, num_threads, merge_all) are ignored.
        """
        txt_extensions = [
            ".txt",
            ".json",
            ".csv",
            ".tsv",
            ".epub",
            ".pdf",
            ".docx",
            ".py",
        ]
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
            "<|LF|>",
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

    def add_special(self, special: str, idx=None):
        """
        Adds a special token.
        If idx is provided, we attempt to force the special token to that id (by modifying internal dictionaries).
        (This is not officially supported by HF tokenizers, so use with caution.)
        """
        if special not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens([special])
        if idx is not None:
            # Force assignment in the internal vocab dictionaries.
            self.tokenizer._token_to_id[special] = idx
            self.tokenizer._id_to_token[idx] = special
        self.specials[special] = self.tokenizer.get_vocab()[special]
