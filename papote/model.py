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


class ZeroScale(nn.Module):

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1) + 1e-5)

    def forward(self, x):
        return x * self.scale


class SinusoidalPositional(torch.nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed. Here, we use sine and cosine
    functions of different frequencies.
    """

    def __init__(self, embedding_dim, max_seq_length=5000):
        super().__init__()

        import math
        pe = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length,
                                dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() *
            (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, input_ids):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        return self.pe[:, :input_ids.shape[1], :]


class ScaledSinosoidal(SinusoidalPositional):
    """Sinusoidal with scaling (see FLASH paper)."""

    def __init__(self, embedding_dim, max_seq_length):
        super().__init__(embedding_dim, max_seq_length)
        self.scale_factor = torch.nn.Parameter(
            torch.tensor([1.0 / embedding_dim**0.5]))

    def forward(self, input_ids):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        return self.scale_factor * self.pe[:, :input_ids.shape[1], :]


class SelfAttention(nn.Module):

    def __init__(self, hidden_size, num_heads, head_size, dropout_rate):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.dropout_rate = dropout_rate
        self.qkv = tu.xavier(
            nn.Linear(hidden_size, head_size * num_heads * 3, bias=False))
        self.fc = tu.constant_init(
            nn.Linear(head_size * num_heads, hidden_size, bias=False), 0)
        self.dropout = (nn.Dropout(dropout_rate, True)
                        if dropout_rate > 0 else nn.Identity())

    def forward(self, x):
        b, l, h, d = x.shape[0], x.shape[1], self.num_heads, self.head_size
        # bld -> (q/k/v)bhld
        qkv = self.qkv(x).reshape(b, l, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = nn.functional.scaled_dot_product_attention(q,
                                                         k,
                                                         v,
                                                         is_causal=True,
                                                         dropout_p=0.0)
        # bhld -> blhd
        att = att.permute(0, 2, 1, 3).contiguous().reshape(b, l, h * d)
        return self.dropout(self.fc(att))


class GEGLU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class TransformerBlock(nn.Module):

    def __init__(self, hidden_size, num_heads, head_size, dropout_rate):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.sa = SelfAttention(hidden_size, num_heads, head_size,
                                dropout_rate)
        self.feed_forward = nn.Sequential(
            tu.kaiming(nn.Linear(hidden_size, 4 * hidden_size, bias=False), ),
            GEGLU(),
            tu.constant_init(
                nn.Linear(2 * hidden_size, hidden_size, bias=False), 0.),
            (nn.Dropout(dropout_rate, True)
             if dropout_rate > 0 else nn.Identity()))
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, pos=None):
        x = self.sa(self.layer_norm1(x)) + x
        x = self.feed_forward(self.layer_norm2(x)).add_(x)
        return x


class Transformer(nn.Module):

    def __init__(self, num_tokens, hidden_size, num_layers, num_heads,
                 head_size, context_size, dropout_rate):
        super().__init__()
        self.context_size = context_size
        self.token_embedding = nn.Embedding(num_tokens, hidden_size)
        self.positional_embedding = nn.Parameter(
            torch.randn(1, context_size, hidden_size))
        sin = ScaledSinosoidal(hidden_size, context_size)
        self.positional_embedding.data = 0.02 * sin.pe
        self.layer_norm_in = nn.LayerNorm(hidden_size)
        self.dropout = (nn.Dropout(dropout_rate)
                        if dropout_rate > 0 else nn.Identity())

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, head_size, dropout_rate)
            for _ in range(num_layers)
        ])

        self.layer_norm_out = nn.LayerNorm(hidden_size)

        self.unembed = nn.Linear(hidden_size, num_tokens, bias=False)
        with torch.no_grad():
            nn.init.normal_(self.token_embedding.weight, 0, 0.02)
            nn.init.normal_(self.unembed.weight, 0, 1 / hidden_size)

        #self.check_params()
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
                m.eps = 1e-6

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
        # - all kinds of l2 norm were worse than none
        # - all different ways to plug in positional embedding were worse
        pos = self.positional_embedding[:, :inputs.size(1)]
        outputs = self.token_embedding(inputs)
        outputs = self.layer_norm_in(outputs) + self.pos_norm(pos)
        outputs = self.dropout(outputs)

        for transformer_block in self.transformer_blocks:
            outputs = transformer_block(outputs, pos)

        outputs = self.layer_norm_out(outputs)
        logits = self.unembed(outputs)
        return logits

    def decayable_parameters(self):
        for mod in self.children():
            if (mod is self.token_embedding or mod is self.unembed):
                continue
            for p in mod.parameters():
                yield p

    def undecayable_parameters(self):
        yield self.positional_embedding
        yield self.token_embedding.weight

    def mu_parametrization(self):
        from collections import defaultdict
        params = defaultdict(list)
        for name, p in self.named_parameters():
            #if 'unembed' in name: continue
            decayable = not (len(p.size()) == 1 or 'embedding' in name)
            if len(p.size()) >= 2:
                size_in = (1 if 'embedding' in name else 2
                           )  #p.view(p.size(0), -1).size(1))
            else:
                size_in = 1
            params[(decayable, size_in)].append(p)
        return params

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def num_parameters_without_embeddings(self):
        return (sum(p.numel() for p in self.parameters()) -
                self.token_embedding.weight.numel() -
                self.positional_embedding.numel() -
                self.unembed.weight.numel())


def transformer_from_checkpoint(checkpoint):
    specs = list_models()[checkpoint['model_type']]
    return make_transformer(
        checkpoint['model_type'],
        checkpoint['model']['_orig_mod.token_embedding.weight'].shape[0],
        checkpoint['model']['_orig_mod.positional_embedding'].shape[1],
        checkpoint['model'].get('_orig_mod.dropout.p', 0))


def list_models():
    from model_specs.tiny import list_models as tiny
    from model_specs.fim import list_models as fim
    from model_specs.chinchilla import list_models as chinchilla
    return {
        **tiny(),
        **fim(),
        **chinchilla(),
    }


def make_transformer(size, vocab_size, context_len, dropout=0.1):
    """
    For a vocab size of 4096 and a context of 512:

    tiny-1M
        0.88704 M params
        0.329984 M params without embeddings

    tiny-3M
        2.42944 M params
        1.315328 M params without embeddings

    tiny-8M
        7.48032 M params
        5.252096 M params without embeddings

    tiny-28M
        25.4464 M params
        20.989952 M params without embeddings

    tiny-33M
        30.292992 M params
        23.60832 M params without embeddings

    fim-xxs
        12.200448 M params
        8.858112 M params without embeddings

    fim-xs
        25.4464 M params
        20.989952 M params without embeddings

    fim-s
        77.503488 M params
        70.818816 M params without embeddings

    fim-m
        260.673536 M params
        251.76064 M params without embeddings

    fim-l
        579.753984 M params
        566.38464 M params without embeddings

    fim-xl
        1024.663552 M params
        1006.83776 M params without embeddings

    fim-xxl
        2119.77216 M params
        2097.48992 M params without embeddings

    fim-xxxl
        5404.901376 M params
        5369.249792 M params without embeddings

    chinchilla-44M
        25.4464 M params
        20.989952 M params without embeddings

    chinchilla-57M
        34.896384 M params
        29.88288 M params without embeddings

    chinchilla-74M
        46.55872 M params
        40.98816 M params without embeddings

    chinchilla-90M
        58.8544 M params
        53.28384 M params without embeddings

    chinchilla-106M
        71.15008 M params
        65.57952 M params without embeddings

    chinchilla-117M
        77.503488 M params
        70.818816 M params without embeddings

    chinchilla-140M
        95.207424 M params
        88.522752 M params without embeddings

    chinchilla-163M
        112.91136 M params
        106.226688 M params without embeddings

    chinchilla-175M
        120.246784 M params
        112.448 M params without embeddings

    chinchilla-196M
        136.310272 M params
        128.511488 M params without embeddings

    chinchilla-217M
        152.37376 M params
        144.574976 M params without embeddings

    chinchilla-251M
        176.754688 M params
        167.841792 M params without embeddings

    chinchilla-278M
        197.7344 M params
        188.821504 M params without embeddings

    chinchilla-306M
        218.714112 M params
        209.801216 M params without embeddings

    chinchilla-425M
        306.1504 M params
        295.00928 M params without embeddings

    chinchilla-489M
        355.31776 M params
        344.17664 M params without embeddings

    chinchilla-509M
        369.20576 M params
        356.950528 M params without embeddings

    chinchilla-552M
        404.48512 M params
        393.344 M params without embeddings

    chinchilla-587M
        428.696576 M params
        416.441344 M params without embeddings

    chinchilla-632M
        461.758464 M params
        448.38912 M params without embeddings

    chinchilla-664M
        488.187392 M params
        475.93216 M params without embeddings

    chinchilla-724M
        532.555776 M params
        519.186432 M params without embeddings

    chinchilla-816M
        603.353088 M params
        589.983744 M params without embeddings

    chinchilla-893M
        658.000896 M params
        642.403328 M params without embeddings

    chinchilla-1018M
        754.36032 M params
        738.762752 M params without embeddings

    chinchilla-1143M
        850.719744 M params
        835.122176 M params without embeddings

    chinchilla-1266M
        940.761088 M params
        922.935296 M params without embeddings

    chinchilla-1424M
        1060.834816 M params
        1041.894912 M params without embeddings

    chinchilla-1429M
        1066.614784 M params
        1048.788992 M params without embeddings

    chinchilla-1593M
        1192.46848 M params
        1174.642688 M params without embeddings

    chinchilla-1609M
        1202.910208 M params
        1183.970304 M params without embeddings

    chinchilla-1731M
        1294.304256 M params
        1274.25024 M params without embeddings

    chinchilla-1794M
        1344.9856 M params
        1326.045696 M params without embeddings

    chinchilla-2007M
        1506.67776 M params
        1486.623744 M params without embeddings

    chinchilla-2283M
        1719.051264 M params
        1698.997248 M params without embeddings

    chinchilla-2298M
        1726.49472 M params
        1704.21248 M params without embeddings

    chinchilla-2639M
        1988.67968 M params
        1966.39744 M params without embeddings

    chinchilla-2980M
        2250.86464 M params
        2228.5824 M params without embeddings

    chinchilla-3530M
        2674.463232 M params
        2651.06688 M params without embeddings

    chinchilla-3802M
        2879.675392 M params
        2855.164928 M params without embeddings

    chinchilla-4084M
        3091.96544 M params
        3066.340864 M params without embeddings

    tiny-X models from https://arxiv.org/abs/2305.07759
    fim-X models from https://arxiv.org/abs/2207.14255
    """
    models = list_models()
    return Transformer(vocab_size,
                       context_size=context_len,
                       dropout_rate=dropout,
                       **models[size])
