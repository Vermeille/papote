"""Basic smoke tests for the papote package.

This module only exercises the HuggingFace based BPE tokenizer to ensure it
can be trained and used for a simple round-trip encode/decode cycle.
"""

from papote.bpe import BPE


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


if __name__ == "__main__":  # pragma: no cover - manual invocation
    test_roundtrip()
    print("ok")

