import random
import torch
from torchvision.transforms import Compose
import papote.data_utils as data
from papote.experiments.think.helpers import think_gain, ThinkObjective


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
        return Compose([
            data.NFKC(),
            data.Tokenize(self.bpe),
            data.Align(self.ctx + 1, self.bpe.token_to_id('<|NUL|>')),
        ])

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
        return y.ne(self.bpe.token_to_id('<|NUL|>'))

    def metrics(self, x, y, loss):
        """Return additional metrics to log during training."""
        return {}


class ThinkExperiment(Experiment):
    """Experiment inserting a <|THINK|> token during training."""

    TOKEN = '<|THINK|>'

    def configure_tokenizer(self):
        self.bpe.add_special(self.TOKEN)
        self.think_token = self.bpe.specials[self.TOKEN]

    def objective(self):
        return ThinkObjective(self.think_token)

    def metrics(self, x, y, loss):
        tg = think_gain(x, y, loss, self.bpe)
        return {'think_gain': tg.item() if isinstance(tg, torch.Tensor) else tg}


class FillInTheMiddleExperiment(Experiment):
    """Experiment using fill-in-the-middle data augmentation."""

    SUFFIX = '<|SUFFIX|>'
    PREFIX = '<|PREFIX|>'
    WRAP = '<|WRAP|>'

    def configure_tokenizer(self):
        self.bpe.add_special(self.SUFFIX)
        self.bpe.add_special(self.PREFIX)
        self.bpe.add_special(self.WRAP)
        self.suffix_token = self.bpe.specials[self.SUFFIX]
        self.prefix_token = self.bpe.specials[self.PREFIX]
        self.wrap_token = self.bpe.specials[self.WRAP]

    def transforms(self):
        return Compose([
            data.NFKC(),
            data.Tokenize(self.bpe),
            data.Align(self.ctx + 1, self.bpe.token_to_id('<|NUL|>')),
            data.FillInTheMiddle(
                self.suffix_token, self.prefix_token, self.wrap_token, p=0.5
            ),
        ])
