import torch
import pytest
from papote.model import RotarySingle


def _manual_rotary(rot, q, seq_dim=-2):
    seq_len = q.shape[seq_dim]
    t = torch.arange(seq_len, device=q.device).type_as(rot.inv_freq)
    freqs = torch.einsum("i,j->ij", t, rot.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    shape = [1] * q.ndim
    shape[seq_dim] = seq_len
    shape[-1] = cos.shape[-1]
    cos = cos.view([seq_len, cos.shape[-1]]).reshape(shape)
    sin = sin.view([seq_len, sin.shape[-1]]).reshape(shape)
    return q * cos + rot.rotate_half(q) * sin


def test_rotate_half_swaps_and_negates():
    rot = RotarySingle(4)
    x = torch.tensor([[1., 2., 3., 4.]])
    expected = torch.tensor([[-3., -4., 1., 2.]])
    assert torch.allclose(rot.rotate_half(x), expected)


def test_apply_rotary_pos_emb_matches_manual():
    torch.manual_seed(0)
    rot = RotarySingle(4)
    q = torch.randn(1, 3, 4)
    seq_len = q.shape[-2]
    t = torch.arange(seq_len, device=q.device).type_as(rot.inv_freq)
    freqs = torch.einsum("i,j->ij", t, rot.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    q_manual = q * cos[None, :, :] + rot.rotate_half(q) * sin[None, :, :]
    q_rot = rot.apply_rotary_pos_emb(q, cos, sin)
    assert torch.allclose(q_rot, q_manual)


def test_forward_updates_cache_and_matches_manual():
    torch.manual_seed(1)
    rot = RotarySingle(8)
    q = torch.randn(2, 10, 8)
    q_rot = rot(q)
    assert rot.seq_len_cached == 10
    q_manual = _manual_rotary(rot, q)
    assert torch.allclose(q_rot, q_manual)


def test_forward_reuses_cache_for_shorter_sequence():
    torch.manual_seed(2)
    rot = RotarySingle(6)
    q1 = torch.randn(1, 8, 6)
    rot(q1)
    cached_ptr = rot.cos_cached.data_ptr()
    q2 = torch.randn(1, 4, 6)
    q_rot = rot(q2)
    assert rot.cos_cached.data_ptr() == cached_ptr
    assert rot.seq_len_cached == 8
    q_manual = _manual_rotary(rot, q2)
    assert torch.allclose(q_rot, q_manual)


def test_forward_seq_dim_argument():
    torch.manual_seed(3)
    rot = RotarySingle(4)
    q = torch.randn(1, 5, 1, 4)
    q_rot = rot(q, seq_dim=1)
    q_manual = _manual_rotary(rot, q, seq_dim=1)
    assert torch.allclose(q_rot, q_manual)

