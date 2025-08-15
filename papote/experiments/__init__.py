import random
import torch
from torchvision.transforms import Compose
import papote.data_utils as data
from papote.experiments.think.helpers import think_gain, ThinkObjective
from papote.experiments.randomcase.utils import RandomCase


class ExperimentsByName(dict):
    """A dictionary-like class to store and retrieve experiments by name.
    Can register experiments class using it as a decorator."""

    def register(self, name):
        """Decorator to register an experiment class."""

        def decorator(cls):
            self[name] = cls
            return cls

        return decorator


EXPERIMENTS = ExperimentsByName()


@EXPERIMENTS.register("base")
class Experiment:
    """Base experiment for training.

    By default this corresponds to vanilla next-token prediction. Subclasses can
    customize the tokenizer, data transforms, objectives and additional metrics.
    """

    def __init__(self, bpe, ctx):
        self.bpe = bpe
        self.ctx = ctx
        self.configure_tokenizer()

    # --- tokenizer -----------------------------------------------------
    def configure_tokenizer(self):
        """Hook to modify the tokenizer before model instantiation."""
        return self.bpe

    # --- data ----------------------------------------------------------
    def transforms(self):
        """Return transforms applied to raw text before chunking."""
        return Compose(
            [
                data.NFKC(),
                data.Tokenize(self.bpe),
                data.Align(self.ctx + 1, self.bpe.token_to_id("<|NUL|>")),
            ]
        )

    def objective(self):
        """Return objective converting tokenized chunks to model inputs."""
        return data.NextTokenObjective()

    def prepare_batch(self, batch):
        """Process a batch from the dataloader before feeding the model."""
        return batch

    def model_inputs(self, x):
        """Transform inputs before passing them to the model."""
        return x

    def loss_mask(self, x, y):
        """Return the mask to use when computing the loss."""
        return y.ne(self.bpe.token_to_id("<|NUL|>"))

    def metrics(self, x, y, loss):
        """Return additional metrics to log during training."""
        return {}


@EXPERIMENTS.register("shuffle")
class ShuffleExperiment(Experiment):
    """Experiment shuffling token order with explicit position indices."""

    LEFT_TO_RIGHT = "<|LEFT-TO-RIGHT|>"

    def __init__(self, bpe, ctx, num_permutations):
        self.num_permutations = num_permutations
        super().__init__(bpe, ctx)

    def configure_tokenizer(self):
        tokens = [self.LEFT_TO_RIGHT] + [
            f"<|SHUFFLE{i}|>" for i in range(1, self.num_permutations)
        ]
        for t in tokens:
            self.bpe.add_special(t)
        self.permutation_tokens = [self.bpe.specials[t] for t in tokens]

    def objective(self):
        permuter = data.Permute(self.num_permutations, self.ctx)
        next_token = data.NextTokenObjective()
        perm_tokens = self.permutation_tokens

        def obj(seq):
            seq = seq[:-1]
            idx, perm, permuted = permuter(seq)
            prefix = torch.tensor([perm_tokens[idx]], dtype=seq.dtype)
            tokens = torch.cat([prefix, permuted])
            x, y = next_token(tokens)
            positions = torch.cat(
                [torch.tensor([0], dtype=perm.dtype), perm[:-1] + 1]
            )
            return (x, positions), y

        return obj


@EXPERIMENTS.register("randomcase")
class RandomCaseExperiment(Experiment):
    UPPER_CASE = "<|UPPER_CASE|>"
    LOWER_CASE = "<|LOWER_CASE|>"

    def configure_tokenizer(self):
        self.bpe.add_special(self.UPPER_CASE)
        self.bpe.add_special(self.LOWER_CASE)
        self.upper_token = self.bpe.specials[self.UPPER_CASE]
        self.lower_token = self.bpe.specials[self.LOWER_CASE]

    # --- data ----------------------------------------------------------
    def transforms(self):
        """Return transforms applied to raw text before chunking."""
        return Compose(
            [
                data.NFKC(),
                RandomCase(self.lower_token, self.upper_token),
                data.Tokenize(self.bpe),
                data.Align(self.ctx + 1, self.bpe.token_to_id("<|NUL|>")),
            ]
        )


@EXPERIMENTS.register("think")
class ThinkExperiment(Experiment):
    """Experiment inserting a <|THINK|> token during training."""

    TOKEN = "<|THINK|>"

    def configure_tokenizer(self):
        self.bpe.add_special(self.TOKEN)
        self.think_token = self.bpe.specials[self.TOKEN]

    def objective(self):
        return ThinkObjective(self.think_token)

    def metrics(self, x, y, loss):
        tg = think_gain(x, y, loss, self.bpe)
        return {"think_gain": tg.item() if isinstance(tg, torch.Tensor) else tg}


@EXPERIMENTS.register("mlm")
class MLMExperiment(Experiment):
    MASK = "<|MASK|>"

    def configure_tokenizer(self):
        self.bpe.add_special(self.MASK)
        self.mask_token = self.bpe.specials[self.MASK]

    def objective(self):
        return data.MLMObjective(self.mask_token, 0.15)


@EXPERIMENTS.register("fim")
class FillInTheMiddleExperiment(Experiment):
    """Experiment using fill-in-the-middle data augmentation."""

    SUFFIX = "<|SUFFIX|>"
    PREFIX = "<|PREFIX|>"
    WRAP = "<|WRAP|>"

    def configure_tokenizer(self):
        self.bpe.add_special(self.SUFFIX)
        self.bpe.add_special(self.PREFIX)
        self.bpe.add_special(self.WRAP)
        self.suffix_token = self.bpe.specials[self.SUFFIX]
        self.prefix_token = self.bpe.specials[self.PREFIX]
        self.wrap_token = self.bpe.specials[self.WRAP]

    def transforms(self):
        return Compose(
            [
                data.NFKC(),
                data.Tokenize(self.bpe),
                data.Align(self.ctx + 1, self.bpe.token_to_id("<|NUL|>")),
                data.FillInTheMiddle(
                    self.suffix_token, self.prefix_token, self.wrap_token, p=0.5
                ),
            ]
        )
