import torch
import pytest
from papote.model import Rotary


def _manual_rotary(rot, q, k, seq_dim=-2):
    seq_len = q.shape[seq_dim]
    t = torch.arange(seq_len, device=q.device).type_as(rot.inv_freq)
    freqs = torch.einsum("i,j->ij", t, rot.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    # reshape cos/sin for broadcasting
    shape = [1] * q.ndim
    shape[seq_dim] = seq_len
    shape[-1] = cos.shape[-1]
    cos = cos.view([seq_len, cos.shape[-1]])
    sin = sin.view([seq_len, sin.shape[-1]])
    cos = cos.reshape(shape)
    sin = sin.reshape(shape)
    return (
        q * cos + rot.rotate_half(q) * sin,
        k * cos + rot.rotate_half(k) * sin,
    )


def test_rotate_half_swaps_and_negates():
    rot = Rotary(4)
    x = torch.tensor([[1., 2., 3., 4.]])
    expected = torch.tensor([[-3., -4., 1., 2.]])
    assert torch.allclose(rot.rotate_half(x), expected)


def test_apply_rotary_pos_emb_matches_manual():
    torch.manual_seed(0)
    rot = Rotary(4)
    q = torch.randn(1, 1, 3, 4)
    k = torch.randn(1, 1, 3, 4)
    v = torch.randn(1, 1, 3, 4)
    seq_len = q.shape[-2]
    t = torch.arange(seq_len, device=q.device).type_as(rot.inv_freq)
    freqs = torch.einsum("i,j->ij", t, rot.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    q_manual = q * cos[None, None, :, :] + rot.rotate_half(q) * sin[None, None, :, :]
    k_manual = k * cos[None, None, :, :] + rot.rotate_half(k) * sin[None, None, :, :]
    q_rot, k_rot, v_out = rot.apply_rotary_pos_emb(q, k, v, cos, sin)
    assert torch.allclose(q_rot, q_manual)
    assert torch.allclose(k_rot, k_manual)
    assert torch.allclose(v_out, v)


def test_forward_updates_cache_and_matches_manual():
    torch.manual_seed(1)
    rot = Rotary(8)
    q = torch.randn(2, 3, 10, 8)
    k = torch.randn(2, 3, 10, 8)
    v = torch.randn(2, 3, 10, 8)
    q_rot, k_rot, v_out = rot(q, k, v)
    key = (rot.dim, rot.base, q.device)
    assert Rotary._cache[key][2] == 10
    q_manual, k_manual = _manual_rotary(rot, q, k)
    assert torch.allclose(q_rot, q_manual)
    assert torch.allclose(k_rot, k_manual)
    assert torch.allclose(v_out, v)


def test_forward_reuses_cache_for_shorter_sequence():
    torch.manual_seed(2)
    rot1 = Rotary(6)
    rot2 = Rotary(6)
    q1 = torch.randn(1, 1, 8, 6)
    k1 = torch.randn(1, 1, 8, 6)
    v1 = torch.randn(1, 1, 8, 6)
    rot1(q1, k1, v1)
    key = (rot1.dim, rot1.base, q1.device)
    cached_ptr = Rotary._cache[key][0].data_ptr()
    q2 = torch.randn(1, 1, 4, 6)
    k2 = torch.randn(1, 1, 4, 6)
    v2 = torch.randn(1, 1, 4, 6)
    q_rot, k_rot, v_out = rot2(q2, k2, v2)
    assert Rotary._cache[key][0].data_ptr() == cached_ptr
    assert Rotary._cache[key][2] == 8
    q_manual, k_manual = _manual_rotary(rot1, q2, k2)
    assert torch.allclose(q_rot, q_manual)
    assert torch.allclose(k_rot, k_manual)
    assert torch.allclose(v_out, v2)


def test_forward_seq_dim_argument():
    torch.manual_seed(3)
    rot = Rotary(4)
    q = torch.randn(1, 5, 1, 4)
    k = torch.randn(1, 5, 1, 4)
    v = torch.randn(1, 5, 1, 4)
    q_rot, k_rot, v_out = rot(q, k, v, seq_dim=1)
    q_manual, k_manual = _manual_rotary(rot, q, k, seq_dim=1)
    assert torch.allclose(q_rot, q_manual)
    assert torch.allclose(k_rot, k_manual)
    assert torch.allclose(v_out, v)


def test_forward_with_positions_indexes_cache():
    torch.manual_seed(4)
    rot = Rotary(4)
    q = torch.randn(2, 3, 5, 4)
    k = torch.randn(2, 3, 5, 4)
    v = torch.randn(2, 3, 5, 4)
    positions = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
    q_rot, k_rot, v_out = rot(q, k, v, positions=positions)
    t = torch.arange(positions.max().item() + 1,
                    device=q.device).type_as(rot.inv_freq)
    freqs = torch.einsum("i,j->ij", t, rot.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().index_select(0, positions.reshape(-1)).view(
        positions.shape + (rot.dim,))
    sin = emb.sin().index_select(0, positions.reshape(-1)).view(
        positions.shape + (rot.dim,))
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_manual = q * cos + rot.rotate_half(q) * sin
    k_manual = k * cos + rot.rotate_half(k) * sin
    assert torch.allclose(q_rot, q_manual)
    assert torch.allclose(k_rot, k_manual)
    assert torch.allclose(v_out, v)
