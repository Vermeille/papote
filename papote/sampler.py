import torch
from typing import List, Optional
from contextlib import suppress
import torch.nn.functional as F
from papote.data_utils import NFKC


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
                                minlength=max(idx.max(), prompt.max()) + 1)
        counts = torch.gather(counts, 0, idx)
        counts = self.penalty**counts.float()
        return logits * counts, idx


class FixedRepetitionPenalty:

    def __init__(self, penalty: float = 0.8, window: int = 32):
        self.penalty = penalty
        self.window = window

    def __call__(self, logits, idx, prompt):
        counts = torch.bincount(prompt[-self.window:],
                                minlength=max(idx.max(), prompt.max()) + 1)
        counts = torch.gather(counts, 0, idx)
        m = logits.min()
        return (logits - m) * torch.where(counts >= 1, self.penalty, 1.0), idx


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

    def __call__(self, encoded, next_token, prob, logit):
        pass


class PromptPassthrough:

    def __call__(self, prompt):
        return prompt


class Sampler:

    def __init__(self,
                 model,
                 bpe,
                 logits_policy=RawLogits(),
                 stopping_criterion=StopTooLong(1024),
                 event_handler=NullEventHandler(),
                 prompt_processor=PromptPassthrough()):
        self.model = model
        self.bpe = bpe
        self.ctx_len = model.context_size
        self.logits_policy = logits_policy
        self.stopping_criterion = stopping_criterion
        self.event_handler = event_handler
        self.prompt_processor = prompt_processor

    def sample(self, prompt):
        model = self.model
        bpe = self.bpe

        kv_cache = None  #[]
        rank = next(model.parameters()).device
        model.eval()
        prompt = NFKC()(prompt)
        encoded = bpe.encode_text(prompt)
        with suppress(KeyboardInterrupt):
            while not self.stopping_criterion(encoded):
                encoded = self.prompt_processor(encoded)

                prompt = torch.tensor(encoded[-self.ctx_len:],
                                      dtype=torch.long)

                logits = F.log_softmax(model((prompt[None]).to(rank),
                                             kv_cache)[0][-1].float().cpu(),
                                       dim=-1)
                idx = torch.arange(logits.shape[-1])

                logits, idx = self.logits_policy(logits, idx, prompt)

                probs = F.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, 1).item()
                next_token = idx[next_idx]
                self.event_handler(encoded, next_token, probs[next_idx],
                                   logits[next_idx])
                encoded.append(next_token)
            return bpe.decode_text(encoded)


class PromptCFG:

    def __init__(self, model, cfg, bpe):
        self.model = model
        self.prompt = []
        self.prev_len = 10000000
        self.cfg = cfg
        self.device = next(model.parameters()).device
        self.bpe = bpe

    @staticmethod
    def list_startswith(l, prefix):
        return len(l) >= len(prefix) and all(l[i] == prefix[i]
                                             for i in range(len(prefix)))

    def __call__(self, logits, idx, prompt):
        # Detect if the prompt has changed (new sampling run)
        if len(prompt) < self.prev_len:
            self.prompt = list(prompt)
        self.prev_len = len(prompt)
        # Remove the saved prompt from the beginning of the current prompt
        # Warning: we have to account for the fact that the prompt may start to
        # be out of the context window
        for i in range(len(self.prompt)):
            if self.list_startswith(prompt, self.prompt[i:]):
                prompt = prompt[len(self.prompt) - i:]
                break
        else:
            # If the prompt hasn't changed, we don't need to do anything
            return logits, idx
        if len(prompt) <= 3:
            return logits, idx

        unconditional_logits = F.log_softmax(self.model(prompt[None].to(
            self.device))[0][-1].float().cpu(),
                                             dim=-1)

        cond_logits = F.log_softmax(logits, dim=-1)
        cfg_logits = self.cfg * cond_logits + (1 -
                                               self.cfg) * unconditional_logits
        return logits, idx


class ForbiddenLogits:

    def __init__(self, forbidden):
        self.forbidden = forbidden

    def __call__(self, logits, idx, prompt):
        mask = torch.full_like(idx, True, dtype=torch.bool)
        mask[torch.tensor(self.forbidden)] = False
        return logits[mask], idx[mask]


def candidates_for_continuation(vocab: list[bytes],
                                continuation: bytes) -> list[int]:
    # TODO: use a trie (Copilot's suggestion. I thought about it too but are we
    # gaining much and is it worth the complexity?)
    candidates = [
        i for i, token in enumerate(vocab)
        if continuation.startswith(token) or token.startswith(continuation)
    ]
    candidates.sort(key=lambda i: len(vocab[i]), reverse=True)
    return candidates


def tokenize_continuation(
        vocab: list[bytes],
        continuation: bytes) -> tuple[list[int], list[int], bytes]:
    tokens = []
    while True:
        candidates = candidates_for_continuation(vocab, continuation)
        assert len(candidates) > 0, f'No candidates for {continuation}'
        token = vocab[candidates[0]]
        if continuation.startswith(token):
            tokens.append(candidates[0])
            continuation = continuation[len(token):]
        else:
            return tokens, candidates, continuation


class TokenHealer:

    def __init__(self, vocab: list[bytes]):
        self.vocab = vocab
        self.num_rollback_chars = max(len(token) for token in vocab)
        self.continuation = b''
        self.first = True
        self.candidates = None

    def prompt_processor(self, tokens: list[int]) -> list[int]:
        if self.first:
            self.first = False
            # backtrack as many tokens / characters as the longest token
            continuation = b''
            while len(continuation) < self.num_rollback_chars and len(
                    tokens) > 1:
                continuation = self.vocab[tokens[-1]] + continuation
                tokens = tokens[:-1]
        else:
            if self.continuation == b'':
                self.candidates = None
                return tokens
            assert (self.continuation.startswith(self.vocab[tokens[-1]])
                    or self.vocab[tokens[-1]].startswith(self.continuation))
            continuation = self.continuation[len(self.vocab[tokens[-1]]):]

        # restore unambiguous tokens from the continuation
        next_tokens, candidates, continuation = tokenize_continuation(
            self.vocab, continuation)
        tokens += next_tokens
        # set candidates
        self.candidates = candidates
        self.continuation = continuation
        return tokens

    def only_valid_logits(self, logits, idx, prompt):
        if self.candidates is None:
            return logits, idx
        mask = torch.full_like(idx, False, dtype=torch.bool)
        mask[torch.tensor(self.candidates)] = True
        return logits[mask], idx[mask]

    def __call__(self, *args):
        if len(args) == 1 and isinstance(args[0], list):
            return self.prompt_processor(args[0])
        elif len(args) == 3 and isinstance(args[0], torch.Tensor):
            return self.only_valid_logits(*args)
        else:
            raise ValueError(f'Invalid arguments: {args}')


def default_sampler(model,
                    bpe,
                    top_k=30,
                    top_p=0.95,
                    temperature=0.7,
                    length=1024,
                    repeat_penalty=0.8,
                    repeat_window=32,
                    typical_p=None,
                    cfg=None,
                    sep=''):
    healer = TokenHealer(bpe.vocab)
    policy = []

    if cfg is not None:
        policy.append(PromptCFG(model, cfg, bpe))

    policy += [
        #healer,
        #ForbiddenLogits([ bpe.specials['<|SYN|>'], bpe.specials['<|ETB|>'], bpe.specials['<|NAK|>'] ])
    ]
    policy += [
        #FixedRepetitionPenalty(repeat_penalty, repeat_window),
        TopK(top_k),
        TopP(top_p),
    ]

    if typical_p is not None:
        policy.append(Typical(typical_p))

    policy += [Temperature(temperature)]

    return Sampler(model,
                   bpe,
                   logits_policy=LogitsComposite(*policy),
                   stopping_criterion=StopTooLong(length),
                   #prompt_processor=healer
                   )
