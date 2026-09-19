from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field

import torch

from infer.engine import Engine
from infer.sampling import SamplingParams, make_generator, sample_token


@dataclass
class GenerationRequest:
    prompt_tokens: list[int]
    params: SamplingParams
    output: queue.Queue[int | None]
    tokens: list[int] = field(default_factory=list)
    slot: int | None = None
    last_logits: torch.Tensor | None = None
    generated: int = 0
    generator: torch.Generator | None = None


class Scheduler:
    """Admits waiting requests into free KV slots and decodes them together."""

    def __init__(self, engine: Engine, poll_timeout: float = 0.02) -> None:
        self.engine = engine
        self.poll_timeout = poll_timeout
        self._waiting: queue.Queue[GenerationRequest | None] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="infer-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._waiting.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def submit(self, prompt_tokens: list[int], params: SamplingParams) -> queue.Queue[int | None]:
        out: queue.Queue[int | None] = queue.Queue()
        self._waiting.put(
            GenerationRequest(
                prompt_tokens=list(prompt_tokens),
                params=params,
                output=out,
                tokens=list(prompt_tokens),
                generator=make_generator(params.seed),
            )
        )
        return out

    def _admit(self, active: list[GenerationRequest]) -> None:
        while self.engine.free_slots() > 0:
            try:
                item = self._waiting.get_nowait()
            except queue.Empty:
                return
            if item is None:
                continue
            self._prefill(item)
            active.append(item)

    def _prefill(self, req: GenerationRequest) -> None:
        slot = self.engine.allocate_slot()
        req.slot = slot
        req.last_logits = self.engine.prefill_slot(slot, req.prompt_tokens)

    def _complete(self, req: GenerationRequest) -> None:
        req.output.put(None)
        if req.slot is not None:
            self.engine.free_slot(req.slot)
            req.slot = None

    def _abort_all(self, active: list[GenerationRequest]) -> None:
        for req in active:
            self._complete(req)
        active.clear()

    def _run(self) -> None:
        active: list[GenerationRequest] = []
        try:
            while not self._stop.is_set():
                self._admit(active)
                if not active:
                    try:
                        item = self._waiting.get(timeout=self.poll_timeout)
                    except queue.Empty:
                        continue
                    if item is None:
                        continue
                    self._prefill(item)
                    active.append(item)
                    continue

                next_ids: list[int] = []
                still_active: list[GenerationRequest] = []
                for req in active:
                    assert req.last_logits is not None and req.slot is not None
                    max_tokens = min(
                        req.params.max_tokens,
                        self.engine.max_seq_len - len(req.prompt_tokens),
                    )
                    if req.generated >= max_tokens:
                        self._complete(req)
                        continue
                    stop_ids = set(req.params.stop_token_ids) | set(self.engine.stop_token_ids)
                    token_id = sample_token(
                        req.last_logits,
                        req.params,
                        req.tokens,
                        req.generator,
                    )
                    if token_id in stop_ids:
                        self._complete(req)
                        continue
                    req.output.put(token_id)
                    req.tokens.append(token_id)
                    req.generated += 1
                    next_ids.append(token_id)
                    still_active.append(req)

                active = still_active
                if not active:
                    continue
                slots = [req.slot for req in active if req.slot is not None]
                logits = self.engine.decode_slots(slots, next_ids)
                for req, row in zip(active, logits, strict=True):
                    req.last_logits = row
        finally:
            self._abort_all(active)


def iter_queue(out: queue.Queue[int | None]) -> list[int]:
    tokens: list[int] = []
    while True:
        item = out.get()
        if item is None:
            return tokens
        tokens.append(item)
