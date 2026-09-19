from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SamplingParams:
    max_tokens: int = 256
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    stop_token_ids: list[int] = field(default_factory=list)
    seed: int | None = None


def apply_repetition_penalty(
    logits: torch.Tensor,
    token_ids: list[int],
    penalty: float,
) -> torch.Tensor:
    if penalty == 1.0 or not token_ids:
        return logits
    ids = torch.tensor(list(set(token_ids)), device=logits.device, dtype=torch.long)
    selected = logits[ids]
    logits = logits.clone()
    logits[ids] = torch.where(selected > 0, selected / penalty, selected * penalty)
    return logits


def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.numel():
        return logits
    kth = torch.topk(logits, top_k).values[-1]
    return logits.masked_fill(logits < kth, torch.finfo(logits.dtype).min)


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    if top_p <= 0.0:
        raise ValueError("top_p must be in (0, 1].")
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    mask = cumulative > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits.new_empty(logits.shape).fill_(torch.finfo(logits.dtype).min).scatter_(0, sorted_idx, sorted_logits)


def sample_token(
    logits: torch.Tensor,
    params: SamplingParams,
    past_ids: list[int],
    generator: torch.Generator | None = None,
) -> int:
    """Sample a single next-token id from a 1-D logits vector."""
    if logits.ndim != 1:
        raise ValueError(f"expected 1-D logits, got shape {tuple(logits.shape)}")
    logits = logits.float()
    logits = apply_repetition_penalty(logits, past_ids, params.repetition_penalty)

    if params.temperature <= 1e-5:
        return int(torch.argmax(logits).item())

    logits = logits / params.temperature
    logits = _apply_top_k(logits, params.top_k)
    logits = _apply_top_p(logits, params.top_p)
    probs = torch.softmax(logits, dim=-1)
    if generator is None:
        return int(torch.multinomial(probs, num_samples=1).item())
    cpu_probs = probs.detach().cpu()
    return int(torch.multinomial(cpu_probs, num_samples=1, generator=generator).item())


def make_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator
