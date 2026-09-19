#!/usr/bin/env python3
"""Measure prefill and decode tokens/sec across batch size, dtype, and compile.

Examples:
    uv run python benchmarks/bench_decode.py --tiny
    uv run python benchmarks/bench_decode.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from infer.engine import Engine, pick_device, pick_dtype  # noqa: E402
from infer.model.llama import LlamaForCausalLM  # noqa: E402


def _tiny_engine(device: torch.device, dtype: torch.dtype, compile_model: bool, max_batch: int) -> Engine:
    sys.path.insert(0, str(ROOT))
    from tests.helpers import DummyTokenizer, tiny_config

    config = tiny_config()
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    return Engine(
        model,
        DummyTokenizer(),
        config,
        device=device,
        dtype=dtype,
        max_batch_size=max_batch,
        max_seq_len=256,
        compile_model=compile_model,
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _run(
    engine: Engine,
    prompt_len: int,
    gen_len: int,
    batch_size: int,
) -> tuple[float, float]:
    prompts = [[3 + (i % 50)] * prompt_len for i in range(batch_size)]
    slots = [engine.allocate_slot() for _ in range(batch_size)]
    try:
        _sync(engine.device)
        t0 = time.perf_counter()
        last = []
        for slot, prompt in zip(slots, prompts, strict=True):
            logits = engine.prefill_slot(slot, prompt)
            last.append(int(torch.argmax(logits).item()))
        _sync(engine.device)
        t1 = time.perf_counter()
        tokens = list(last)
        for _ in range(gen_len):
            logits = engine.decode_slots(slots, tokens)
            tokens = [int(row.argmax().item()) for row in logits]
        _sync(engine.device)
        t2 = time.perf_counter()
        return t1 - t0, t2 - t1
    finally:
        for slot in slots:
            engine.free_slot(slot)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefill/decode throughput table")
    parser.add_argument("--model", default=None, help="HF repo or local dir (ignored with --tiny)")
    parser.add_argument("--tiny", action="store_true", help="Use a random 2-layer toy model")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--gen-len", type=int, default=64)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    device = pick_device(args.device)
    dtype = pick_dtype(device, args.dtype)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x]
    max_batch = max(batch_sizes)

    if args.tiny:
        sys.path.insert(0, str(ROOT))
        engine = _tiny_engine(device, dtype, args.compile, max_batch)
        label = "tiny-random"
    else:
        model_id = args.model or "Qwen/Qwen2.5-0.5B-Instruct"
        engine = Engine.from_pretrained(
            model_id,
            device=args.device,
            dtype=args.dtype,
            compile_model=args.compile,
            max_batch_size=max_batch,
            max_seq_len=max(args.prompt_len + args.gen_len + 8, 256),
        )
        label = model_id

    print(
        f"{'model':<28} {'device':<8} {'dtype':<10} {'compile':<8} "
        f"{'batch':>5} {'prompt':>6} {'gen':>5} {'prefill tok/s':>14} {'decode tok/s':>12}"
    )
    for batch in batch_sizes:
        _run(engine, args.prompt_len, args.gen_len, batch)  # warmup
        prefill_times: list[float] = []
        decode_times: list[float] = []
        for _ in range(args.iters):
            prefill_s, decode_s = _run(engine, args.prompt_len, args.gen_len, batch)
            prefill_times.append(prefill_s)
            decode_times.append(decode_s)
        prefill_s = sum(prefill_times) / len(prefill_times)
        decode_s = sum(decode_times) / len(decode_times)
        prefill_tps = (args.prompt_len * batch) / prefill_s if prefill_s else 0.0
        decode_tps = (args.gen_len * batch) / decode_s if decode_s else 0.0
        print(
            f"{label:<28} {str(engine.device):<8} {str(engine.dtype).replace('torch.', ''):<10} "
            f"{str(bool(args.compile)):<8} {batch:>5} {args.prompt_len:>6} {args.gen_len:>5} "
            f"{prefill_tps:>14.2f} {decode_tps:>12.2f}"
        )


if __name__ == "__main__":
    main()
