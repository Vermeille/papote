import torch
import torch.nn as nn
from papote.model import (
    Transformer,
    Rotary,
    RotarySingle,
    NoRotary,
    make_transformer,
    transformer_from_checkpoint,
)


def test_transformer_rotary_flags():
    model = Transformer(
        num_tokens=10,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        head_size=4,
        context_size=8,
        rotary=True,
        rotary_single=True,
    )
    assert isinstance(model.transformer_blocks[0].sa.rotary, Rotary)
    assert isinstance(model.rotary, RotarySingle)

    model_no = Transformer(
        num_tokens=10,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        head_size=4,
        context_size=8,
        rotary=False,
        rotary_single=False,
    )
    assert isinstance(model_no.transformer_blocks[0].sa.rotary, NoRotary)
    assert isinstance(model_no.rotary, nn.Identity)


def test_rotary_flags_checkpoint_roundtrip():
    model = make_transformer(
        "tiny-1M",
        vocab_size=10,
        context_len=8,
        rotary=False,
        rotary_single=False,
    )
    ckpt = {"model_type": "tiny-1M", "model": model.state_dict()}
    loaded = transformer_from_checkpoint(ckpt)
    assert isinstance(loaded.transformer_blocks[0].sa.rotary, NoRotary)
    assert isinstance(loaded.rotary, nn.Identity)
