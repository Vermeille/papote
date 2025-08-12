"""Basic smoke tests for the papote package.

This module only exercises the HuggingFace based BPE tokenizer to ensure it
can be trained and used for a simple round-trip encode/decode cycle.
"""

from papote.bpe import BPE
from papote.sampler import Typical
import torch


def test_roundtrip() -> None:
    bpe = BPE()
    # Train a tiny tokenizer directly from an iterator so we do not depend on
    # external files.
    bpe.tokenizer.train_from_iterator([
        "hello world",
    ], vocab_size=100, min_frequency=1)
    ids = bpe.encode_text("hello world")
    decoded = bpe.decode_text(ids)
    assert decoded == "hello world"


def test_typical_filters_tokens() -> None:
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0])
    idx = torch.arange(logits.size(0))
    filtered_logits, _ = Typical(0.5)(logits, idx, torch.tensor([]))
    assert len(filtered_logits) < len(logits)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    test_roundtrip()
    print("ok")

