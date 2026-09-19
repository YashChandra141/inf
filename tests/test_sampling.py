from __future__ import annotations

import torch

from infer.sampling import SamplingParams, apply_repetition_penalty, make_generator, sample_token


def test_greedy_is_argmax() -> None:
    logits = torch.tensor([0.1, 4.0, 0.2, 1.5])
    token = sample_token(logits, SamplingParams(temperature=0.0), [])
    assert token == 1


def test_temperature_zero_ignores_topk() -> None:
    logits = torch.tensor([0.0, 3.0, 2.0])
    token = sample_token(logits, SamplingParams(temperature=0.0, top_k=1), [])
    assert token == 1


def test_topk_one_equals_greedy() -> None:
    logits = torch.tensor([0.2, 5.0, 0.1, 4.0])
    params = SamplingParams(temperature=1.0, top_k=1)
    generator = make_generator(0)
    token = sample_token(logits, params, [], generator)
    assert token == 1


def test_seeded_sampling_is_deterministic() -> None:
    logits = torch.linspace(-1, 1, 50)
    params = SamplingParams(temperature=0.9, top_p=0.95, top_k=20)
    a = sample_token(logits, params, [], make_generator(123))
    b = sample_token(logits, params, [], make_generator(123))
    c = sample_token(logits, params, [], make_generator(99))
    assert a == b
    assert a != c


def test_repetition_penalty_downweights_seen_tokens() -> None:
    logits = torch.tensor([2.0, 2.0, -2.0, -2.0])
    penalized = apply_repetition_penalty(logits, [0, 2], penalty=2.0)
    assert penalized[0].item() == 1.0
    assert penalized[1].item() == 2.0
    assert penalized[2].item() == -4.0
    assert penalized[3].item() == -2.0


def test_engine_respects_max_tokens(engine) -> None:
    params = SamplingParams(max_tokens=7, temperature=0.0, stop_token_ids=[])
    engine.stop_token_ids = []
    tokens = list(engine.generate([3, 4, 5], params))
    assert len(tokens) == 7


def test_engine_stops_on_stop_token(engine) -> None:
    engine.stop_token_ids = []
    params = SamplingParams(max_tokens=32, temperature=0.0, stop_token_ids=[1])
    # Force the next token by sampling greedy from a prompt; just ensure stop ids are honoured
    # by injecting a vocab where argmax is 1 after we can't easily control weights.
    # Instead, generate with stop id equal to the greedy first token.
    slot = engine.allocate_slot()
    try:
        logits = engine.prefill_slot(slot, [3, 4, 5])
        greedy = int(torch.argmax(logits).item())
    finally:
        engine.free_slot(slot)
    params.stop_token_ids = [greedy]
    tokens = list(engine.generate([3, 4, 5], params))
    assert greedy not in tokens
    assert len(tokens) == 0
