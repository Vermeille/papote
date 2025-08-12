from papote.bpe import BPE


def test_bpe_roundtrip():
    bpe = BPE()
    # Train a tiny tokenizer from an iterator so no external files are needed.
    bpe.tokenizer.train_from_iterator([
        "hello world",
    ], vocab_size=100, min_frequency=1)
    ids = bpe.encode_text("hello world")
    decoded = bpe.decode_text(ids)
    assert decoded == "hello world"

