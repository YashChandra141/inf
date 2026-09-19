from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from infer.config import ModelConfig
from infer.model.layers import DecoderLayer, RMSNorm, RotaryEmbedding

if TYPE_CHECKING:
    from infer.kv_cache import KVCache


class LlamaForCausalLM(torch.nn.Module):
    """Llama / Qwen2 / Mistral-style decoder-only transformer."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = torch.nn.ModuleList(
            [
                DecoderLayer(
                    config.hidden_size,
                    config.intermediate_size,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                    config.rms_norm_eps,
                    config.hidden_act,
                    attention_bias=config.attention_bias,
                    mlp_bias=config.mlp_bias,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def load_hf_state_dict(self, remapped: dict[str, torch.Tensor]) -> None:
        if self.config.tie_word_embeddings:
            remapped.pop("lm_head.weight", None)
        missing, unexpected = self.load_state_dict(remapped, strict=False)
        missing = [key for key in missing if key != "rotary_emb.inv_freq"]
        if missing:
            raise RuntimeError(f"Missing weights: {missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected weights: {unexpected}")

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        cache: KVCache | None = None,
        slots: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.embed_tokens(tokens)
        cos, sin = self.rotary_emb(positions, hidden.dtype)
        for layer_idx, layer in enumerate(self.layers):
            hidden = layer(
                hidden,
                positions,
                cos,
                sin,
                cache=cache,
                slots=slots,
                layer_idx=layer_idx,
            )
        hidden = self.norm(hidden)
        weight = self.embed_tokens.weight if self.lm_head is None else self.lm_head.weight
        return F.linear(hidden, weight)
