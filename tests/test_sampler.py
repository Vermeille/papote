import pytest

torch = pytest.importorskip("torch")

from papote.sampler import (
    ForbiddenTokens,
    CumulativeRepetitionPenalty,
    FixedRepetitionPenalty,
    Temperature,
    TopK,
    TopP,
    Typical,
    StopTooLong,
    StopOnEos,
    candidates_for_continuation,
    tokenize_continuation,
)


def test_forbidden_tokens():
    logits = torch.zeros(5)
    idx = torch.arange(5)
    ft = ForbiddenTokens([1, 3])
    new_logits, new_idx = ft(logits.clone(), idx.clone(), torch.tensor([0]))
    assert torch.isneginf(new_logits[1])
    assert torch.isneginf(new_logits[3])
    assert torch.equal(new_idx, idx)


def test_cumulative_repetition_penalty():
    logits = torch.ones(5)
    idx = torch.arange(5)
    prompt = torch.tensor([1, 2, 1, 3])
    crp = CumulativeRepetitionPenalty(penalty=0.5, window=4)
    new_logits, _ = crp(logits.clone(), idx, prompt)
    assert torch.isclose(new_logits[1], torch.tensor(0.25))
    assert torch.isclose(new_logits[2], torch.tensor(0.5))
    assert torch.isclose(new_logits[4], torch.tensor(1.0))


def test_fixed_repetition_penalty():
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    idx = torch.arange(4)
    prompt = torch.tensor([0, 1, 0, 2])
    frp = FixedRepetitionPenalty(penalty=0.5, window=4)
    new_logits, _ = frp(logits.clone(), idx, prompt)
    expected = torch.tensor([0.0, 0.5, 1.0, 3.0])
    assert torch.allclose(new_logits, expected)


def test_temperature():
    logits = torch.tensor([2.0, 4.0])
    idx = torch.arange(2)
    temp = Temperature(2.0)
    new_logits, _ = temp(logits, idx, torch.tensor([]))
    assert torch.allclose(new_logits, logits / 2)


def test_topk():
    logits = torch.tensor([1.0, 3.0, 2.0])
    idx = torch.tensor([0, 1, 2])
    topk = TopK(k=2)
    new_logits, new_idx = topk(logits, idx, torch.tensor([]))
    assert torch.equal(new_idx, torch.tensor([1, 2]))
    assert torch.equal(new_logits, torch.tensor([3.0, 2.0]))


def test_topp():
    logits = torch.tensor([1.0, 2.0, 3.0])
    idx = torch.tensor([0, 1, 2])
    topp = TopP(p=0.5)
    new_logits, new_idx = topp(logits, idx, torch.tensor([]))
    assert len(new_logits) == 2
    assert set(new_idx.tolist()) == {0, 1}


def test_typical():
    logits = torch.tensor([1.0, 2.0, 3.0])
    idx = torch.tensor([0, 1, 2])
    typ_none = Typical(None)
    l_none, i_none = typ_none(logits, idx, torch.tensor([]))
    assert torch.equal(l_none, logits)
    assert torch.equal(i_none, idx)
    typ = Typical(0.5)
    l, i = typ(logits, idx, torch.tensor([]))
    assert len(l) <= len(logits)
    assert len(i) == len(l)


def test_stop_criteria():
    too_long = StopTooLong(max_length=3)
    assert not too_long([1, 2])
    assert too_long([1, 2, 3])
    on_eos = StopOnEos(eos=2)
    assert on_eos([1, 2])
    assert not on_eos([1, 3])


def test_candidates_for_continuation():
    vocab = [b"a", b"ab", b"b"]
    cands = candidates_for_continuation(vocab, b"a")
    assert cands == [1, 0]


def test_tokenize_continuation():
    vocab = [b"a", b"b", b"c", b"ab"]
    tokens, cands, cont = tokenize_continuation(vocab, b"abc")
    assert tokens == [3, 2]
    assert cont == b""
    assert cands[0] == 3
