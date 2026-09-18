from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from infer.kv_cache import KVCache


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return rms_norm(hidden, self.weight, self.eps)


def rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig_dtype = hidden.dtype
    hidden_f = hidden.float()
    variance = hidden_f.pow(2).mean(dim=-1, keepdim=True)
    hidden_f = hidden_f * torch.rsqrt(variance + eps)
    return weight * hidden_f.to(orig_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # query/key: [B, n_heads, T, head_dim]; cos/sin: [B, T, head_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (query * cos) + (rotate_half(query) * sin)
    k_embed = (key * cos) + (rotate_half(key) * sin)
    return q_embed, k_embed


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, head_dim: int, rope_theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device=positions.device, dtype=torch.float32)
        # positions: [B, T]
        freqs = torch.einsum("bt,d->btd", positions.float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def repeat_kv(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden
    batch, n_kv, seq_len, head_dim = hidden.shape
    hidden = hidden[:, :, None, :, :].expand(batch, n_kv, n_rep, seq_len, head_dim)
    return hidden.reshape(batch, n_kv * n_rep, seq_len, head_dim)


def build_attn_mask(
    query_positions: torch.Tensor,
    lengths: torch.Tensor,
    kv_len: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return an additive mask of shape [B, 1, T, kv_len]."""
    kv_pos = torch.arange(kv_len, device=query_positions.device)
    allowed = (kv_pos[None, None, :] <= query_positions[:, :, None]) & (
        kv_pos[None, None, :] < lengths[:, None, None]
    )
    mask = torch.zeros(
        query_positions.shape[0],
        1,
        query_positions.shape[1],
        kv_len,
        device=query_positions.device,
        dtype=dtype,
    )
    mask = mask.masked_fill(~allowed[:, None, :, :], torch.finfo(dtype).min)
    return mask


class Attention(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        bias: bool,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.n_rep = num_heads // num_kv_heads
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = torch.nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = torch.nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = torch.nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
        slots: torch.Tensor | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden.shape
        query = self.q_proj(hidden).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if cache is not None:
            assert slots is not None and layer_idx is not None
            cache.update(layer_idx, key, value, positions, slots)
            key, value = cache.gather(layer_idx, slots)
            attn_mask = build_attn_mask(positions, cache.lengths[slots], cache.max_seq_len, query.dtype)
            is_causal = False
        else:
            attn_mask = None
            is_causal = True

        key = repeat_kv(key, self.n_rep)
        value = repeat_kv(value, self.n_rep)
        attn = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn)


class MLP(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        *,
        bias: bool,
    ) -> None:
        super().__init__()
        if hidden_act not in {"silu", "swish"}:
            raise ValueError(f"Unsupported hidden_act={hidden_act!r}; expected silu.")
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class DecoderLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        hidden_act: str,
        *,
        attention_bias: bool,
        mlp_bias: bool,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            bias=attention_bias,
        )
        self.mlp = MLP(hidden_size, intermediate_size, hidden_act, bias=mlp_bias)
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
        slots: torch.Tensor | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden = residual + self.self_attn(
            hidden, positions, cos, sin, cache=cache, slots=slots, layer_idx=layer_idx
        )
        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        return residual + self.mlp(hidden)
