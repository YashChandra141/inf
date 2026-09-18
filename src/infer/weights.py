from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from infer.config import ModelConfig


HF_ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "*.safetensors",
    "*.safetensors.index.json",
]


def resolve_model_dir(model_id: str, cache_dir: str | Path | None = None) -> Path:
    path = Path(model_id)
    if path.is_dir() and (path / "config.json").exists():
        return path.resolve()
    snapshot = snapshot_download(
        repo_id=model_id,
        cache_dir=str(cache_dir) if cache_dir else None,
        allow_patterns=HF_ALLOW_PATTERNS,
    )
    return Path(snapshot)


def load_config(model_dir: Path, *, model_name: str = "") -> ModelConfig:
    return ModelConfig.from_json(model_dir / "config.json", model_name=model_name or model_dir.name)


def load_safetensors(model_dir: Path) -> dict[str, torch.Tensor]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        files = sorted(set(weight_map.values()))
    else:
        files = sorted(
            p.name
            for p in model_dir.glob("*.safetensors")
            if p.is_file() and not p.name.startswith("._")
        )
    if not files:
        raise FileNotFoundError(f"No .safetensors weight files found in {model_dir}")

    state: dict[str, torch.Tensor] = {}
    for name in files:
        state.update(load_file(str(model_dir / name)))
    return state


def remap_hf_state_dict(hf_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip the Hugging Face `model.` prefix so keys match LlamaForCausalLM."""
    out: dict[str, torch.Tensor] = {}
    for key, value in hf_state.items():
        if "rotary_emb" in key or "inv_freq" in key:
            continue
        mapped = key[6:] if key.startswith("model.") else key
        out[mapped] = value
    return out
