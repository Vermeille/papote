import torch
import torch.nn as nn
import torchelie.utils as tu
import torch.nn.functional as F


class SquaredReLU(nn.Module):

    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu(x, self.inplace).square_()


class SelfAttention(nn.Module):

    def __init__(self, hidden_size, num_heads, head_size, dropout_rate):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.dropout_rate = dropout_rate
        self.qkv = tu.kaiming(
            nn.Linear(hidden_size, head_size * num_heads * 3, bias=False))
        self.fc = tu.constant_init(
            nn.Linear(head_size * num_heads, hidden_size, bias=False), 0)
        self.dropout = nn.Dropout(dropout_rate, True)

    def forward(self, x):
        b, l, h, d = x.shape[0], x.shape[1], self.num_heads, self.head_size
        # bld -> (q/k/v)bhld
        qkv = self.qkv(x).reshape(b, l, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if q.device.type == 'cuda':
            q, k, v = q.half(), k.half(), v.half()
        att = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=0 * self.dropout_rate if self.training else 0.0).float()
        # bhld -> blhd
        att = att.permute(0, 2, 1, 3).contiguous().reshape(b, l, h * d)
        return self.dropout(self.fc(att))


class TransformerBlock(nn.Module):

    def __init__(self, hidden_size, num_heads, head_size, dropout_rate):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.sa = SelfAttention(hidden_size, num_heads, head_size,
                                dropout_rate)
        self.feed_forward = nn.Sequential(
            tu.kaiming(nn.Linear(hidden_size, 4 * hidden_size, bias=False)),
            nn.GELU(),
            tu.constant_init(
                nn.Linear(4 * hidden_size, hidden_size, bias=False), 0),
            nn.Dropout(dropout_rate, True))
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.sa(self.layer_norm1(x)) + x
        x = self.feed_forward(self.layer_norm2(x)).add_(x)
        return x


class Transformer(nn.Module):

    def __init__(self, num_tokens, hidden_size, num_layers, num_heads,
                 head_size, context_size, dropout_rate):
        super().__init__()
        self.context_size = context_size
        self.token_embedding = tu.normal_init(
            nn.Embedding(num_tokens, hidden_size))
        self.positional_embedding = nn.Parameter(
            torch.randn(1, context_size, hidden_size) * 0.02)
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, head_size, dropout_rate)
            for _ in range(num_layers)
        ])

        self.ln_out = nn.LayerNorm(hidden_size)
        #self.projector = nn.Linear(hidden_size, hidden_size, bias=False)

        self.fc = nn.Linear(hidden_size, num_tokens, bias=False)
        self.fc.weight = self.token_embedding.weight

        self.check_params()
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
                m.eps = 1e-12

    def check_params(self):
        from collections import Counter
        allp = set(self.parameters())
        namedp = {p: n for n, p in self.named_parameters()}
        decays = set(self.decayable_parameters())
        undecays = set(self.undecayable_parameters())
        assert len(decays & undecays) == 0
        assert len(allp - decays - undecays) == 0
        dup_decay = [
            namedp[p]
            for p, cnt in Counter(self.decayable_parameters()).items()
            if cnt > 1
        ]
        assert len(
            dup_decay) == 0, f'Duplicated decayable parameters: {dup_decay}'
        dup_undecay = [
            namedp[p]
            for p, cnt in Counter(self.undecayable_parameters()).items()
            if cnt > 1
        ]
        assert len(dup_undecay
                   ) == 0, f'Duplicated undecayable parameters: {dup_undecay}'

    def forward(self, inputs):
        outputs = self.token_embedding(
            inputs) + self.positional_embedding[:, :inputs.size(1)]
        outputs = self.ln(outputs)
        outputs = self.dropout(outputs)

        for transformer_block in self.transformer_blocks:
            outputs = transformer_block(outputs)

        outputs = self.ln_out(outputs)
        #outputs = self.projector(outputs)
        logits = self.fc(outputs)
        return logits

    def decayable_parameters(self):
        for mod in self.children():
            if (mod is self.token_embedding or mod is self.fc):
                continue
            for p in mod.parameters():
                yield p

    def undecayable_parameters(self):
        yield self.positional_embedding
        yield self.token_embedding.weight

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def num_parameters_without_embeddings(self):
        return (sum(p.numel() for p in self.parameters()) -
                self.token_embedding.weight.numel() -
                self.positional_embedding.numel())


def make_transformer(size, vocab_size, context_len, dropout=0.1):
    if size == 'xxxs':
        return Transformer(vocab_size,
                           hidden_size=256,
                           context_size=context_len,
                           head_size=64,
                           num_layers=6,
                           num_heads=6,
                           dropout_rate=dropout)
    if size == 'xxs':
        return Transformer(vocab_size,
                           hidden_size=384,
                           context_size=context_len,
                           head_size=64,
                           num_layers=6,
                           num_heads=6,
                           dropout_rate=dropout)
    elif size == 'xs':
        return Transformer(vocab_size,
                           hidden_size=512,
                           context_size=context_len,
                           head_size=64,
                           num_layers=8,
                           num_heads=8,
                           dropout_rate=dropout)
    elif size == 's':
        return Transformer(vocab_size,
                           hidden_size=768,
                           context_size=context_len,
                           head_size=64,
                           num_layers=12,
                           num_heads=12,
                           dropout_rate=dropout)
    elif size == 'm':
        return Transformer(vocab_size,
                           hidden_size=1024,
                           context_size=context_len,
                           head_size=64,
                           num_layers=24,
                           num_heads=16,
                           dropout_rate=dropout)
    elif size == 'l':
        return Transformer(vocab_size,
                           hidden_size=1536,
                           context_size=context_len,
                           head_size=96,
                           num_layers=24,
                           num_heads=16,
                           dropout_rate=dropout)
    elif size == 'xl':
        return Transformer(vocab_size,
                           hidden_size=2048,
                           context_size=context_len,
                           head_size=128,
                           num_layers=24,
                           num_heads=16,
                           dropout_rate=dropout)
    elif size == 'xxl':
        return Transformer(vocab_size,
                           hidden_size=2560,
                           context_size=context_len,
                           head_size=80,
                           num_layers=32,
                           num_heads=32,
                           dropout_rate=dropout)
    elif size == 'xxxl':
        return Transformer(vocab_size,
                           hidden_size=4096,
                           context_size=context_len,
                           head_size=128,
                           num_layers=32,
                           num_heads=32,
                           dropout_rate=dropout)
