def unthink(inputs):
    think_mask = (inputs == 5)
    think_offset = think_mask.cumsum(1) * think_mask
    inputs = inputs.view(-1)[torch.arange(inputs.numel(), device=inputs.device)
                             - think_offset.to(inputs.device).view(-1)].view(
                                 *inputs.shape)
    think_offset = think_mask[..., None]
    return inputs, think_mask


def think_gain(x, y, loss, bpe):
    # Compare the loss of the token preceding <|THINK|> to the loss of the token following <|THINK|>
    # This is a proxy for the gain in perplexity that we get from thinking
    think_mask = (x == bpe.specials['<|THINK|>'])
    first_think = think_mask.int().argmax(dim=1) - 1
    num_think = (first_think >= 0).sum()
    last_think = first_think + think_mask.sum(1)
    think_gain = torch.sum(loss[torch.arange(len(x)), last_think] -
                           loss[torch.arange(len(x)), first_think]) / num_think

    assert (y[torch.arange(len(x)), last_think] == y[torch.arange(len(x)),
                                                     first_think]).all()
    return think_gain


class ThinkObjective:

    def __init__(self, think_token):
        self.think_token = think_token

    def __call__(self, x):
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
                 think_token,
                 confusion_threshold=50,
                 max_think=5,
                 verbose=False):
        self.think_token = think_token
        self.confusion_threshold = confusion_threshold
        self.verbose = verbose
        self.max_think = max_think

    def __call__(self, logits, idx, prompt):
        thinking = [
            tok == self.think_token for tok in prompt[-self.max_think:]
        ]
        num_thinking = sum(thinking)
        if all(thinking):
            return logits, idx
        elif len(idx) > self.confusion_threshold:
            if self.verbose:
                print(f'#{num_thinking},{len(idx)},{logits.max():.3f}', end='')
            return torch.tensor([1.0]), torch.tensor([self.think_token],
                                                     dtype=torch.long)
        return logits, idx


class AlwaysThink:

    def __init__(self,
                 think_token,
                 confusion_threshold=50,
                 max_think=5,
                 verbose=False):
        self.think_token = think_token
        self.confusion_threshold = confusion_threshold
        self.verbose = verbose
        self.max_think = max_think

    def __call__(self, logits, idx, prompt):
        thinking = prompt[-1] == self.think_token

        print(f'#{len(idx)},{logits.max():.3f}', end='')
        if thinking:
            return logits, idx
        else:
            return torch.tensor([1.0]), torch.tensor([self.think_token],
                                                     dtype=torch.long)
        return logits, idx


class CleanThink:
    """
    Removes the think token from the prompt before sampling. We only keep the
    think tokens at the end of the prompt.
    """

    def __init__(self, think_token):
        self.think_token = think_token

    def __call__(self, prompt):
        n_think = 0
        while prompt[len(prompt) - n_think - 1] == self.think_token:
            n_think += 1
        prompt = [tok for tok in prompt if tok != self.think_token]
        return prompt + [self.think_token] * n_think
