import torch
from typing import List, Optional
from contextlib import suppress
from bpe import Text
import torch.nn.functional as F


class RawLogits:

    def __call__(self, logits, idx, prompt):
        return logits, idx


class ForbiddenTokens:

    def __init__(self, forbidden: List[int]):
        self.forbidden = torch.tensor(forbidden)

    def __call__(self, logits, idx, prompt):
        logits[self.forbidden] = -float('inf')
        return logits, idx


class CumulativeRepetitionPenalty:

    def __init__(self, penalty: float = 0.95, window: int = 32):
        self.penalty = penalty
        self.window = window

    def __call__(self, logits, idx, prompt):
        counts = torch.bincount(prompt[-self.window:],
                                minlength=logits.shape[-1])
        counts = self.penalty**counts.float()
        return logits * counts, idx


class FixedRepetitionPenalty:

    def __init__(self, penalty: float = 0.8, window: int = 32):
        self.penalty = penalty
        self.window = window

    def __call__(self, logits, idx, prompt):
        counts = torch.bincount(prompt[-self.window:],
                                minlength=logits.shape[-1])
        return logits * torch.where(counts > 1, self.penalty, 1.0), idx


class Temperature:

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def __call__(self, logits, idx, prompt):
        return logits / self.temperature, idx


class TopK:

    def __init__(self, k: int = 40):
        self.k = k

    def __call__(self, logits, idx, prompt):
        top_indices = torch.topk(logits, min(self.k, logits.shape[-1]))[1]
        return logits[top_indices], idx[top_indices]


class TopP:

    def __init__(self, p: float = 0.95):
        self.p = p

    def __call__(self, logits, idx, prompt):
        probs = F.softmax(logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        # Keep at least one token
        cumulative_probs[0] = 0
        mask = cumulative_probs < self.p
        logits = logits[mask]
        return logits, idx[mask]


class Typical:

    def __init__(self, p: Optional[float]):
        self.p = p

    def __call__(self, logits, idx, prompt):
        if self.p is None:
            return logits, idx
        normalized = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(normalized.exp() * normalized, dim=-1)
        shifted = torch.abs(-normalized - entropy)
        typical_sorted = torch.argsort(shifted, dim=-1)
        idx = idx[typical_sorted]
        typical_probs = F.softmax(logits[idx], dim=-1).cumsum(dim=-1)

        typical_probs[0] = 0
        typical_sorted = typical_sorted[typical_probs < typical_p]
        logits = logits[typical_sorted]
        return logits, idx[typical_sorted]


class StopTooLong:

    def __init__(self, max_length: int = 1024):
        self.max_length = max_length

    def __call__(self, prompt):
        return len(prompt) >= self.max_length


class StopOnEos:

    def __init__(self, eos: int):
        self.eos = eos

    def __call__(self, prompt):
        return prompt[-1] == self.eos


class LogitsComposite:

    def __init__(self, *samplers):
        self.samplers = samplers

    def __call__(self, logits, idx, prompt):
        for sampler in self.samplers:
            logits, idx = sampler(logits, idx, prompt)
        return logits, idx


class StopComposite:

    def __init__(self, *stoppers):
        self.stoppers = stoppers

    def __call__(self, prompt):
        for stopper in self.stoppers:
            if stopper(prompt):
                return True
        return False


class NullEventHandler:

    def __call__(self, encoded, next_token):
        pass


class Sampler:

    def __init__(self,
                 model,
                 bpe,
                 logits_policy=RawLogits(),
                 stopping_criterion=StopTooLong(1024),
                 event_handler=NullEventHandler()):
        self.model = model
        self.bpe = bpe
        self.ctx_len = model.context_size
        self.logits_policy = logits_policy
        self.stopping_criterion = stopping_criterion
        self.event_handler = event_handler

    def sample(self, prompt):
        model = self.model
        bpe = self.bpe

        rank = next(model.parameters()).device
        model.eval()
        encoded = bpe.encode_text(prompt)
        self.event_handler(encoded, None)
        with suppress(KeyboardInterrupt):
            while not self.stopping_criterion(encoded):
                t = Text('')
                t.set_tokens(encoded)
                t.tokenize(bpe.merges)
                encoded = t.as_tokens()
                prompt = torch.tensor(encoded[-self.ctx_len:],
                                      dtype=torch.long)

                logits = model(prompt[None].to(rank))[0][-1].float().cpu()
                idx = torch.arange(logits.shape[-1])

                logits, idx = self.logits_policy(logits, idx, prompt)

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                next_token = idx[next_token]
                self.event_handler(encoded, next_token)
                encoded.append(next_token)
            return bpe.decode_text(encoded)
