from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from infer.config import SUPPORTED_MODEL_TYPES, ModelConfig
from infer.tokenizer import HuggingFaceTokenizer, render_chat_template
from infer.weights import remap_hf_state_dict


def test_config_parses_qwen2_dict() -> None:
    cfg = ModelConfig.from_dict(
        {
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "vocab_size": 151936,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
            "max_position_embeddings": 32768,
            "tie_word_embeddings": True,
            "hidden_act": "silu",
            "eos_token_id": 151645,
            "bos_token_id": 151643,
        },
        model_name="demo",
    )
    assert cfg.attention_bias is True
    assert cfg.head_dim == 64
    assert cfg.eos_token_ids == [151645]
    assert cfg.model_name == "demo"


def test_config_rejects_unknown_architecture() -> None:
    try:
        ModelConfig.from_dict({"model_type": "gemma", "hidden_size": 4, "num_attention_heads": 2})
    except ValueError as exc:
        assert "Unsupported" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_supported_model_types() -> None:
    assert SUPPORTED_MODEL_TYPES == frozenset({"llama", "qwen2", "mistral"})


def test_remap_strips_model_prefix() -> None:
    mapped = remap_hf_state_dict(
        {
            "model.embed_tokens.weight": torch.ones(2, 2),
            "lm_head.weight": torch.ones(2, 2),
            "model.layers.0.self_attn.rotary_emb.inv_freq": torch.ones(4),
        }
    )
    assert "embed_tokens.weight" in mapped
    assert "lm_head.weight" in mapped
    assert all("rotary" not in key for key in mapped)


def test_chat_template_render() -> None:
    template = (
        "{% for message in messages %}"
        "{{ bos_token }}{{ message['role'] }}\n{{ message['content'] }}{{ eos_token }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ bos_token }}assistant\n{% endif %}"
    )
    text = render_chat_template(
        template,
        [{"role": "user", "content": "hi"}],
        add_generation_prompt=True,
        special_tokens={"bos_token": "<s>", "eos_token": "</s>"},
    )
    assert text == "<s>user\nhi</s>\n<s>assistant\n"


def test_tokenizer_roundtrip(tmp_path: Path) -> None:
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "hello", "world"])
    tokenizer.train_from_iterator(["hello world hello"], trainer)
    tok_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tok_path))
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    wrapped = HuggingFaceTokenizer.from_dir(tmp_path)
    ids = wrapped.encode("hello world")
    assert wrapped.decode(ids) == "hello world"
