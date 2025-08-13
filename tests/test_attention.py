import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchelie")

from papote.model import SelfAttention, MultiQuerySelfAttention


def test_self_attention_kv_cache_matches_full():
    torch.manual_seed(0)
    sa = SelfAttention(hidden_size=16, num_heads=4, head_size=4)
    x = torch.randn(1, 5, 16)
    full = sa(x)
    kv_cache = [torch.empty(1, 4, 0, 4), torch.empty(1, 4, 0, 4)]
    out_seq = []
    for t in range(5):
        out_seq.append(sa(x[:, t:t+1, :], kv_cache=kv_cache))
    cached = torch.cat(out_seq, dim=1)
    assert torch.allclose(cached, full, atol=1e-6)
    assert kv_cache[0].shape[2] == 5
    assert kv_cache[1].shape[2] == 5


def test_multiquery_self_attention_kv_cache_matches_full():
    torch.manual_seed(0)
    sa = MultiQuerySelfAttention(hidden_size=16, num_heads=4, head_size=4)
    x = torch.randn(1, 5, 16)
    full = sa(x)
    kv_cache = [torch.empty(1, 1, 0, 4), torch.empty(1, 1, 0, 4)]
    out_seq = []
    for t in range(5):
        out_seq.append(sa(x[:, t:t+1, :], kv_cache=kv_cache))
    cached = torch.cat(out_seq, dim=1)
    assert torch.allclose(cached, full, atol=1e-6)
    assert kv_cache[0].shape[2] == 5
    assert kv_cache[1].shape[2] == 5
