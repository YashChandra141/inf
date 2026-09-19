from __future__ import annotations

from typing import Any

from infer.config import ModelConfig


class DummyTokenizer:
    eos_token_id = 1
    eos_token_ids = [1]
    bos_token_id = 0
    pad_token_id = 2

    def encode(self, text: str) -> list[int]:
        ids = [min(127, max(3, ord(ch))) for ch in text]
        return ids or [3]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        chars: list[str] = []
        for token_id in ids:
            if skip_special_tokens and token_id < 3:
                continue
            chars.append(chr(token_id) if 32 <= token_id < 127 else "?")
        return "".join(chars)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
    ) -> list[int] | str:
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("assistant:")
        text = "\n".join(parts)
        return self.encode(text) if tokenize else text


def tiny_config(**overrides: Any) -> ModelConfig:
    data = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=256,
        head_dim=16,
        tie_word_embeddings=True,
        attention_bias=True,
        mlp_bias=False,
        hidden_act="silu",
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        architecture="Qwen2ForCausalLM",
        model_type="qwen2",
        model_name="tiny-test",
        eos_token_ids=[1],
    )
    data.update(overrides)
    return ModelConfig(**data)
