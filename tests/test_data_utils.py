import os
import random
import tempfile
import pytest

torch = pytest.importorskip("torch")

from papote.data_utils import (
    RandomCase,
    RandomPromptSplit,
    Tokenize,
    Crop,
    NFKC,
    Pad,
    Align,
    Tagger,
    NextTokenObjective,
    SeqWeightedLoss,
    binary_entropy,
)


class DummyBPE:
    def encode_text(self, text):
        return [ord(c) for c in text]


def test_random_case():
    rc_lower = RandomCase("L_", "U_", lowercase_p=1.0, uppercase_p=0.0)
    assert rc_lower("Hi") == "L_hi"
    rc_upper = RandomCase("L_", "U_", lowercase_p=0.0, uppercase_p=1.0)
    assert rc_upper("Hi") == "U_HI"
    rc_none = RandomCase("L_", "U_", lowercase_p=0.0, uppercase_p=0.0)
    assert rc_none("Hi") == "Hi"


def test_random_prompt_split():
    rps = RandomPromptSplit("<S>", p=1.0)
    random.seed(0)
    result = rps("hello")
    assert "<S>" in result and len(result) > len("hello")


def test_tokenize_and_crop():
    tok = Tokenize(DummyBPE())
    result = tok("hi")
    expected = [ord(c) for c in "hi<|EOT|>"]
    assert result == expected
    crop = Crop(3)
    assert crop([1, 2, 3, 4]) == [1, 2, 3]


def test_nfkc_pad_align():
    nfkc = NFKC()
    assert nfkc("ｔｅｓｔ") == "test"
    pad = Pad(5, 0)
    assert pad([1, 2]) == [1, 2, 0, 0, 0]
    align = Align(4, 0)
    assert align([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5, 0, 0, 0]


def test_tagger():
    with tempfile.TemporaryDirectory() as tmpdir:
        tag_file = os.path.join(tmpdir, "tags.txt")
        with open(tag_file, "w") as f:
            f.write("dir/subdir tag1 tag2\n")
            f.write("dir tag3\n")
        tagger = Tagger(tag_file)
        random.seed(0)
        tags = tagger.tag_path("dir/subdir/file")
        assert tags == "tag3tag1tag2<|NUL|>"
        assert tagger.tags() == {"tag1", "tag2", "tag3"}


def test_next_token_objective():
    obj = NextTokenObjective()
    data_list = [1, 2, 3]
    x, y = obj(data_list)
    assert x == [1, 2]
    assert y == [2, 3]
    tensor = torch.tensor([1, 2, 3])
    x_t, y_t = obj(tensor)
    assert torch.equal(x_t, torch.tensor([1, 2]))
    assert torch.equal(y_t, torch.tensor([2, 3]))


def test_seq_weighted_loss_and_binary_entropy():
    loss_fn = SeqWeightedLoss()
    x = torch.randn(3, 5)
    y = torch.tensor([1, 2, 3])
    mask = torch.ones(3)
    loss = loss_fn(x, y, mask)
    assert loss.shape == (3,)
    logits = torch.zeros(2, 2)
    labels = torch.tensor([0, 1])
    be = binary_entropy(logits, labels, 'none')
    expected = torch.full((2,), 2 * torch.log(torch.tensor(2.0)))
    assert torch.allclose(be, expected)
