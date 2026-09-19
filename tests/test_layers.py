from __future__ import annotations

import torch
import torch.nn.functional as F

from infer.model.layers import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    build_attn_mask,
    repeat_kv,
    rms_norm,
    rotate_half,
)


def test_rms_norm_matches_reference() -> None:
    torch.manual_seed(1)
    hidden = torch.randn(2, 5, 8)
    weight = torch.randn(8)
    actual = rms_norm(hidden, weight, 1e-6)
    hidden_f = hidden.float()
    expected = weight * (hidden_f * torch.rsqrt(hidden_f.pow(2).mean(-1, keepdim=True) + 1e-6)).to(hidden.dtype)
    torch.testing.assert_close(actual, expected)


def test_rotate_half_swaps_halves() -> None:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    torch.testing.assert_close(rotate_half(x), torch.tensor([-3.0, -4.0, 1.0, 2.0]))


def test_rope_is_length_preserving() -> None:
    rope = RotaryEmbedding(head_dim=8, rope_theta=10000.0)
    positions = torch.arange(6).unsqueeze(0)
    query = torch.randn(1, 2, 6, 8)
    key = torch.randn(1, 2, 6, 8)
    cos, sin = rope(positions, query.dtype)
    q_rot, k_rot = apply_rotary_pos_emb(query, key, cos, sin)
    assert q_rot.shape == query.shape
    assert k_rot.shape == key.shape
    torch.testing.assert_close(q_rot.norm(), query.norm(), rtol=1e-5, atol=1e-5)


def test_repeat_kv_expands_heads() -> None:
    hidden = torch.randn(2, 2, 5, 4)
    repeated = repeat_kv(hidden, 3)
    assert repeated.shape == (2, 6, 5, 4)
    torch.testing.assert_close(repeated[:, 0], hidden[:, 0])
    torch.testing.assert_close(repeated[:, 1], hidden[:, 0])
    torch.testing.assert_close(repeated[:, 2], hidden[:, 0])


def test_attn_mask_is_causal_and_length_aware() -> None:
    positions = torch.tensor([[0, 1, 2]])
    lengths = torch.tensor([2])
    mask = build_attn_mask(positions, lengths, kv_len=4, dtype=torch.float32)
    allowed = mask[0, 0] == 0
    # q=0 attends to 0; q=1 attends to 0,1; q=2 is past length 2 so only keys 0,1
    assert allowed[0, 0]
    assert not allowed[0, 1]
    assert allowed[1, 0] and allowed[1, 1]
    assert not allowed[1, 2]
    assert allowed[2, 0] and allowed[2, 1]
    assert not allowed[2, 2]


def test_sdpa_matches_manual_softmax() -> None:
    torch.manual_seed(2)
    batch, heads, seq_len, dim = 1, 2, 4, 8
    query = torch.randn(batch, heads, seq_len, dim)
    key = torch.randn(batch, heads, seq_len, dim)
    value = torch.randn(batch, heads, seq_len, dim)
    scale = dim**-0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    causal = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
    ref = torch.matmul(torch.softmax(scores, dim=-1), value)
    actual = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    torch.testing.assert_close(actual, ref, atol=1e-5, rtol=1e-5)
