from __future__ import annotations

import pytest
import torch

from infer.config import ModelConfig
from infer.engine import Engine
from infer.model.llama import LlamaForCausalLM
from tests.helpers import DummyTokenizer, tiny_config


@pytest.fixture
def config() -> ModelConfig:
    return tiny_config()


@pytest.fixture
def model(config: ModelConfig) -> LlamaForCausalLM:
    torch.manual_seed(0)
    net = LlamaForCausalLM(config)
    net.eval()
    return net


@pytest.fixture
def engine(model: LlamaForCausalLM, config: ModelConfig) -> Engine:
    return Engine(
        model,
        DummyTokenizer(),
        config,
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_batch_size=4,
        max_seq_len=64,
    )
