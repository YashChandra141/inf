from __future__ import annotations

import torch

from infer.engine import Engine
from infer.sampling import SamplingParams
from infer.scheduler import Scheduler, iter_queue


def _uncached_logits(engine: Engine, token_ids: list[int]) -> torch.Tensor:
    tokens = torch.tensor([token_ids], dtype=torch.long)
    positions = torch.arange(len(token_ids)).unsqueeze(0)
    with torch.inference_mode():
        return engine.model(tokens, positions, cache=None)


def test_prefill_matches_full_forward(engine: Engine) -> None:
    prompt = [3, 5, 7, 9, 11]
    slot = engine.allocate_slot()
    try:
        cached = engine.prefill_slot(slot, prompt)
        full = _uncached_logits(engine, prompt)[0, -1].float()
        torch.testing.assert_close(cached, full, atol=1e-4, rtol=1e-4)
    finally:
        engine.free_slot(slot)


def test_decode_step_matches_full_forward(engine: Engine) -> None:
    prompt = [3, 5, 7, 9]
    nxt = 11
    slot = engine.allocate_slot()
    try:
        engine.prefill_slot(slot, prompt)
        decoded = engine.decode_slots([slot], [nxt])[0]
        full = _uncached_logits(engine, prompt + [nxt])[0, -1].float()
        torch.testing.assert_close(decoded, full, atol=1e-4, rtol=1e-4)
    finally:
        engine.free_slot(slot)


def test_greedy_generate_matches_uncached_argmax(engine: Engine) -> None:
    prompt = [3, 5, 7]
    engine.stop_token_ids = []
    params = SamplingParams(max_tokens=6, temperature=0.0, stop_token_ids=[])
    generated = list(engine.generate(prompt, params))

    tokens = list(prompt)
    for token_id in generated:
        logits = _uncached_logits(engine, tokens)[0, -1]
        assert int(torch.argmax(logits).item()) == token_id
        tokens.append(token_id)


def test_scheduler_matches_serial_generate(engine: Engine) -> None:
    engine.stop_token_ids = []
    params = SamplingParams(max_tokens=8, temperature=0.0, stop_token_ids=[])
    prompt_a = [3, 4, 5, 6]
    prompt_b = [7, 8, 9]

    serial_a = list(engine.generate(prompt_a, params))
    serial_b = list(engine.generate(prompt_b, params))

    scheduler = Scheduler(engine)
    scheduler.start()
    try:
        q_a = scheduler.submit(prompt_a, params)
        q_b = scheduler.submit(prompt_b, params)
        batched_a = iter_queue(q_a)
        batched_b = iter_queue(q_b)
    finally:
        scheduler.stop()

    assert batched_a == serial_a
    assert batched_b == serial_b
