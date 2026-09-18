from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from infer.engine import Engine
from infer.sampling import SamplingParams
from infer.scheduler import Scheduler
from infer.server.api import create_app
from infer.tokenizer import IncrementalDecoder

DEFAULT_MODEL = os.environ.get("INFER_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def _add_engine_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local checkpoint directory")
    parser.add_argument("--device", default="auto", help="cpu | cuda | mps | auto")
    parser.add_argument("--dtype", default=None, help="float32 | bfloat16 | float16")
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--compile", action="store_true", help="torch.compile the model (decode speedup)")


def _build_engine(args: argparse.Namespace) -> Engine:
    print(f"Loading {args.model} on {args.device}...", file=sys.stderr)
    engine = Engine.from_pretrained(
        args.model,
        device=args.device,
        dtype=args.dtype,
        compile_model=args.compile,
        max_batch_size=args.max_batch,
        max_seq_len=args.max_seq_len,
    )
    print(
        f"Ready: {engine.config.architecture}  "
        f"layers={engine.config.num_hidden_layers}  "
        f"device={engine.device}  dtype={engine.dtype}",
        file=sys.stderr,
    )
    return engine


def _params_from_args(args: argparse.Namespace, engine: Engine) -> SamplingParams:
    return SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        stop_token_ids=list(engine.stop_token_ids),
        seed=args.seed,
    )


def cmd_chat(args: argparse.Namespace) -> None:
    engine = _build_engine(args)
    params = _params_from_args(args, engine)
    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    def run_once(user_text: str) -> None:
        messages.append({"role": "user", "content": user_text})
        prompt = engine.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        assert isinstance(prompt, list)
        decoder = IncrementalDecoder(engine.tokenizer)
        pieces: list[str] = []
        sys.stdout.write("assistant> ")
        sys.stdout.flush()
        for token_id in engine.generate(prompt, params):
            piece = decoder.push(token_id)
            pieces.append(piece)
            sys.stdout.write(piece)
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
        messages.append({"role": "assistant", "content": "".join(pieces)})

    if args.prompt:
        run_once(args.prompt)
        return

    print("Interactive chat. Ctrl-C or an empty line to exit.", file=sys.stderr)
    try:
        while True:
            user_text = input("you> ").strip()
            if not user_text:
                break
            run_once(user_text)
    except (KeyboardInterrupt, EOFError):
        print(file=sys.stderr)


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    engine = _build_engine(args)
    scheduler = Scheduler(engine)
    app = create_app(engine, scheduler)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def cmd_bench(args: argparse.Namespace) -> None:
    engine = _build_engine(args)
    prompt = [1] * args.prompt_len
    if prompt[0] == engine.tokenizer.eos_token_id:
        prompt = [0] * args.prompt_len

    def run_once() -> tuple[float, float]:
        slot = engine.allocate_slot()
        try:
            torch.cuda.synchronize() if engine.device.type == "cuda" else None
            t0 = time.perf_counter()
            logits = engine.prefill_slot(slot, prompt)
            if engine.device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            token = int(torch.argmax(logits).item())
            for _ in range(args.gen_len):
                logits = engine.decode_slots([slot], [token])[0]
                token = int(torch.argmax(logits).item())
            if engine.device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            return t1 - t0, t2 - t1
        finally:
            engine.free_slot(slot)

    print("warmup...", file=sys.stderr)
    run_once()
    prefill_times: list[float] = []
    decode_times: list[float] = []
    for _ in range(args.iters):
        prefill_s, decode_s = run_once()
        prefill_times.append(prefill_s)
        decode_times.append(decode_s)

    prefill_s = sum(prefill_times) / len(prefill_times)
    decode_s = sum(decode_times) / len(decode_times)
    prefill_tps = args.prompt_len / prefill_s if prefill_s else 0.0
    decode_tps = args.gen_len / decode_s if decode_s else 0.0
    print(
        f"{'device':<8} {'dtype':<10} {'compile':<8} {'prompt':>8} {'gen':>6} "
        f"{'prefill tok/s':>14} {'decode tok/s':>12}"
    )
    print(
        f"{str(engine.device):<8} {str(engine.dtype).replace('torch.', ''):<10} "
        f"{str(args.compile):<8} {args.prompt_len:>8} {args.gen_len:>6} "
        f"{prefill_tps:>14.2f} {decode_tps:>12.2f}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="infer", description="Llama-family LLM inference engine")
    sub = parser.add_subparsers(dest="cmd", required=True)

    chat = sub.add_parser("chat", help="Generate from a prompt or an interactive loop")
    _add_engine_args(chat)
    chat.add_argument("--prompt", default=None)
    chat.add_argument("--system", default=None)
    chat.add_argument("--max-tokens", type=int, default=128)
    chat.add_argument("--temperature", type=float, default=0.7)
    chat.add_argument("--top-k", type=int, default=0)
    chat.add_argument("--top-p", type=float, default=0.9)
    chat.add_argument("--repetition-penalty", type=float, default=1.0)
    chat.add_argument("--seed", type=int, default=None)
    chat.set_defaults(func=cmd_chat)

    serve = sub.add_parser("serve", help="OpenAI-compatible HTTP server")
    _add_engine_args(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    bench = sub.add_parser("bench", help="Measure prefill and decode tokens/sec")
    _add_engine_args(bench)
    bench.add_argument("--prompt-len", type=int, default=64)
    bench.add_argument("--gen-len", type=int, default=64)
    bench.add_argument("--iters", type=int, default=3)
    bench.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    args.func(args)
