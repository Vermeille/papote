import random
import torch
from torchvision.transforms import Compose
import papote.data_utils as data


def unthink(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove think tokens from inputs and return the mask of removed tokens."""
    think_mask = (inputs == 5)
    think_offset = think_mask.cumsum(1) * think_mask
    inputs = inputs.view(-1)[torch.arange(inputs.numel(), device=inputs.device)
                             - think_offset.to(inputs.device).view(-1)].view(
                                 *inputs.shape)
    think_offset = think_mask[..., None]
    return inputs, think_mask


def think_gain(x: torch.Tensor, y: torch.Tensor, loss: torch.Tensor,
               bpe) -> torch.Tensor:
    """Proxy for perplexity gain obtained from thinking."""
    think_mask = (x == bpe.specials['<|THINK|>'])
    first_think = think_mask.int().argmax(dim=1) - 1
    num_think = (first_think >= 0).sum()
    last_think = first_think + think_mask.sum(1)
    gain = torch.sum(loss[torch.arange(len(x)), last_think] -
                     loss[torch.arange(len(x)), first_think]) / num_think

    assert (y[torch.arange(len(x)), last_think] ==
            y[torch.arange(len(x)), first_think]).all()
    return gain


class ThinkObjective:

    def __init__(self, think_token: int):
        self.think_token = think_token

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = data.NextTokenObjective()(x)
        N = x.shape[0]
        num_think_tokens = random.randint(0, 5)
        pos_think_tokens = random.randint(1, len(x) - num_think_tokens - 1)
        x = torch.cat([
            x[:pos_think_tokens],
            torch.tensor([self.think_token] * num_think_tokens,
                         dtype=torch.long), x[pos_think_tokens:]
        ])
        y = torch.cat([
            y[:pos_think_tokens - 1],
            torch.tensor([y[pos_think_tokens - 1]] * num_think_tokens,
                         dtype=torch.long), y[pos_think_tokens - 1:]
        ])
        return x[:N], y[:N]


class Think:

    def __init__(self,
                 think_token: int,
                 confusion_threshold: int = 50,
                 max_think: int = 5,
                 verbose: bool = False):
        self.think_token = think_token
        self.confusion_threshold = confusion_threshold
        self.verbose = verbose
        self.max_think = max_think

    def __call__(self, logits: torch.Tensor, idx: torch.Tensor,
                 prompt: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        thinking = [tok == self.think_token for tok in prompt[-self.max_think:]]
        num_thinking = sum(thinking)
        if all(thinking):
            return logits, idx
        elif len(idx) > self.confusion_threshold:
            if self.verbose:
                print(f'#{num_thinking},{len(idx)},{logits.max():.3f}',
                      end='')
            return torch.tensor([1.0]), torch.tensor([self.think_token],
                                                     dtype=torch.long)
        return logits, idx


class AlwaysThink:

    def __init__(self,
                 think_token: int,
                 confusion_threshold: int = 50,
                 max_think: int = 5,
                 verbose: bool = False):
        self.think_token = think_token
        self.confusion_threshold = confusion_threshold
        self.verbose = verbose
        self.max_think = max_think

    def __call__(self, logits: torch.Tensor, idx: torch.Tensor,
                 prompt: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        thinking = prompt[-1] == self.think_token

        print(f'#{len(idx)},{logits.max():.3f}', end='')
        if thinking:
            return logits, idx
        else:
            return torch.tensor([1.0]), torch.tensor([self.think_token],
                                                     dtype=torch.long)


class CleanThink:
    """Remove think tokens from the prompt before sampling."""

    def __init__(self, think_token: int):
        self.think_token = think_token

    def __call__(self, prompt: list[int]) -> list[int]:
        n_think = 0
        while prompt[len(prompt) - n_think - 1] == self.think_token:
            n_think += 1
        prompt = [tok for tok in prompt if tok != self.think_token]
        return prompt + [self.think_token] * n_think


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
