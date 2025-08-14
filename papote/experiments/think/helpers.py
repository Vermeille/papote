import random
import torch
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
