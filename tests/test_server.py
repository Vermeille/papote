import torch

from papote.server import build_sampler, Printer
from papote.model import make_transformer, transformer_from_checkpoint
from papote.bpe import BPE


def test_checkpoint_roundtrip(tmp_path):
    """Ensure checkpoints saved by training can be loaded by the server."""
    bpe = BPE()
    model = make_transformer("tiny-1M", vocab_size=8, context_len=16)
    checkpoint = {
        "model_type": "tiny-1M",
        "model": model.state_dict(),
        "bpe": bpe.state_dict(),
    }

    path = tmp_path / "ckpt.pt"
    torch.save(checkpoint, path)

    ckpt = torch.load(path)
    bpe2 = BPE()
    bpe2.load_state_dict(ckpt["bpe"])
    model2 = transformer_from_checkpoint(ckpt)
    load_res = model2.load_state_dict(ckpt["model"])
    assert not load_res.missing_keys
    assert not load_res.unexpected_keys


def test_build_sampler_uses_printer():
    """``build_sampler`` should attach a ``Printer`` event handler."""

    class DummyBPE:
        vocab = [b"a", b"b", b"c"]
        specials = {}

        def decode_text(self, ids, separator=""):
            if isinstance(ids, int):
                ids = [ids]
            tokens = []
            for i in ids:
                if i < len(self.vocab):
                    tokens.append(self.vocab[i].decode("utf-8"))
                else:
                    tokens.append(str(i))
            return separator.join(tokens)

    sent = []

    def send(msg):
        sent.append(msg)

    model = make_transformer("tiny-1M", vocab_size=5, context_len=8)
    bpe = DummyBPE()
    sampler = build_sampler(model, bpe, send)

    assert isinstance(sampler.event_handler, Printer)

    sampler.event_handler([1, 2], 1, 0.5, 0.0)
    assert sent, "send should be called by the printer"

