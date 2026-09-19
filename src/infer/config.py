from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_MODEL_TYPES = frozenset({"llama", "qwen2", "mistral"})


@dataclass
class ModelConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    head_dim: int
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False
    hidden_act: str = "silu"
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    pad_token_id: int | None = None
    architecture: str = "llama"
    model_type: str = "llama"
    model_name: str = ""
    rope_scaling: dict[str, Any] | None = None
    eos_token_ids: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, model_name: str = "") -> ModelConfig:
        model_type = str(data.get("model_type") or "llama").lower()
        archs = data.get("architectures") or []
        architecture = str(archs[0] if archs else model_type)
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type={model_type!r}. "
                f"This engine implements Llama-family models: {sorted(SUPPORTED_MODEL_TYPES)}."
            )
        if data.get("rope_scaling"):
            scaling = data["rope_scaling"]
            scaling_type = str(scaling.get("rope_type") or scaling.get("type") or "default")
            if scaling_type not in {"default", "none", ""}:
                raise ValueError(
                    f"rope_scaling type {scaling_type!r} is not implemented; use a model without RoPE scaling."
                )

        n_heads = int(data["num_attention_heads"])
        hidden = int(data["hidden_size"])
        head_dim = int(data["head_dim"]) if data.get("head_dim") else hidden // n_heads
        n_kv = int(data.get("num_key_value_heads") or n_heads)

        attention_bias = bool(data.get("attention_bias", False))
        if model_type == "qwen2":
            attention_bias = True

        eos = data.get("eos_token_id")
        eos_ids: list[int] = []
        if isinstance(eos, list):
            eos_ids = [int(x) for x in eos]
        elif eos is not None:
            eos_ids = [int(eos)]

        return cls(
            hidden_size=hidden,
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv,
            vocab_size=int(data["vocab_size"]),
            rms_norm_eps=float(data.get("rms_norm_eps", 1e-6)),
            rope_theta=float(data.get("rope_theta", 10000.0)),
            max_position_embeddings=int(data.get("max_position_embeddings", 2048)),
            head_dim=head_dim,
            tie_word_embeddings=bool(data.get("tie_word_embeddings", True)),
            attention_bias=attention_bias,
            mlp_bias=bool(data.get("mlp_bias", False)),
            hidden_act=str(data.get("hidden_act", "silu")),
            bos_token_id=int(data["bos_token_id"]) if data.get("bos_token_id") is not None else None,
            eos_token_id=eos,
            pad_token_id=int(data["pad_token_id"]) if data.get("pad_token_id") is not None else None,
            architecture=architecture,
            model_type=model_type,
            model_name=model_name,
            rope_scaling=data.get("rope_scaling"),
            eos_token_ids=eos_ids,
        )

    @classmethod
    def from_json(cls, path: str | Path, *, model_name: str = "") -> ModelConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = cls.from_dict(data, model_name=model_name)
        gen_path = Path(path).with_name("generation_config.json")
        if gen_path.exists():
            gen = json.loads(gen_path.read_text(encoding="utf-8"))
            extra = gen.get("eos_token_id")
            ids = set(cfg.eos_token_ids)
            if isinstance(extra, list):
                ids.update(int(x) for x in extra)
            elif extra is not None:
                ids.add(int(extra))
            cfg.eos_token_ids = sorted(ids)
        return cfg


def merge_eos_ids(*groups: list[int] | int | None) -> list[int]:
    ids: set[int] = set()
    for group in groups:
        if group is None:
            continue
        if isinstance(group, int):
            ids.add(group)
        else:
            ids.update(int(x) for x in group)
    return sorted(ids)
