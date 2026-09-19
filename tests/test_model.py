from __future__ import annotations

import pytest
import torch

from infer.config import ModelConfig
from infer.engine import Engine
from infer.model.llama import LlamaForCausalLM
from infer.sampling import SamplingParams
from infer.weights import remap_hf_state_dict
from tests.helpers import DummyTokenizer


def _tiny_hf_qwen2():
    transformers = pytest.importorskip("transformers")
    config_cls = getattr(transformers, "Qwen2Config", None)
    model_cls = getattr(transformers, "Qwen2ForCausalLM", None)
    if config_cls is None or model_cls is None:
        pytest.skip("transformers build does not include Qwen2")
    hf_cfg = config_cls(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        hidden_act="silu",
        attention_dropout=0.0,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
    )
    torch.manual_seed(0)
    hf = model_cls(hf_cfg)
    hf.eval()
    return hf, hf_cfg


@pytest.mark.oracle
def test_logits_match_transformers_qwen2() -> None:
    hf, hf_cfg = _tiny_hf_qwen2()
    cfg = ModelConfig.from_dict(hf_cfg.to_dict(), model_name="oracle-qwen2")
    ours = LlamaForCausalLM(cfg)
    ours.load_hf_state_dict(remap_hf_state_dict(hf.state_dict()))
    ours.eval()

    tokens = torch.tensor([[3, 5, 8, 13, 21]], dtype=torch.long)
    positions = torch.arange(tokens.shape[1]).unsqueeze(0)
    with torch.inference_mode():
        ours_logits = ours(tokens, positions, cache=None)
        hf_logits = hf(tokens).logits
    torch.testing.assert_close(ours_logits.float(), hf_logits.float(), atol=1e-3, rtol=1e-3)


@pytest.mark.oracle
def test_greedy_tokens_match_transformers_generate() -> None:
    hf, hf_cfg = _tiny_hf_qwen2()
    cfg = ModelConfig.from_dict(hf_cfg.to_dict(), model_name="oracle-qwen2")
    ours = LlamaForCausalLM(cfg)
    ours.load_hf_state_dict(remap_hf_state_dict(hf.state_dict()))
    engine = Engine(
        ours,
        DummyTokenizer(),
        cfg,
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_batch_size=1,
        max_seq_len=64,
    )
    engine.stop_token_ids = []

    prompt = [3, 5, 8, 13]
    max_new = 8
    with torch.inference_mode():
        hf_out = hf.generate(
            torch.tensor([prompt], dtype=torch.long),
            max_new_tokens=max_new,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=2,
        )
    hf_new = hf_out[0, len(prompt) :].tolist()
    params = SamplingParams(max_tokens=max_new, temperature=0.0, stop_token_ids=[])
    ours_new = list(engine.generate(prompt, params))
    assert ours_new == hf_new
