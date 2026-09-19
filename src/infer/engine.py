from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import torch

from infer.config import ModelConfig, merge_eos_ids
from infer.kv_cache import KVCache
from infer.model.llama import LlamaForCausalLM
from infer.sampling import SamplingParams, make_generator, sample_token
from infer.tokenizer import HuggingFaceTokenizer, TokenizerLike
from infer.weights import load_config, load_safetensors, remap_hf_state_dict, resolve_model_dir


def pick_device(name: str | None = None) -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device, dtype_name: str | None = None) -> torch.dtype:
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if dtype_name:
        if dtype_name not in aliases:
            raise ValueError(f"Unknown dtype {dtype_name!r}")
        return aliases[dtype_name]
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


class Engine:
    def __init__(
        self,
        model: LlamaForCausalLM,
        tokenizer: TokenizerLike,
        config: ModelConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
        max_batch_size: int = 8,
        max_seq_len: int | None = None,
        compile_model: bool = False,
    ) -> None:
        self.model = model.to(device=device, dtype=dtype)
        self.model.eval()
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len or min(config.max_position_embeddings, 4096)
        if compile_model:
            self.model = torch.compile(self.model)
        self.cache = KVCache(
            num_layers=config.num_hidden_layers,
            max_batch_size=max_batch_size,
            num_kv_heads=config.num_key_value_heads,
            max_seq_len=self.max_seq_len,
            head_dim=config.head_dim,
            device=device,
            dtype=dtype,
        )
        self._used = [False] * max_batch_size
        self._lock = threading.Lock()
        self.stop_token_ids = merge_eos_ids(config.eos_token_ids, tokenizer.eos_token_ids)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device: str | None = None,
        dtype: str | None = None,
        compile_model: bool = False,
        max_batch_size: int = 8,
        max_seq_len: int | None = None,
        cache_dir: str | Path | None = None,
    ) -> Engine:
        model_dir = resolve_model_dir(model_id, cache_dir=cache_dir)
        config = load_config(model_dir, model_name=model_id)
        tokenizer = HuggingFaceTokenizer.from_dir(model_dir)
        model = LlamaForCausalLM(config)
        state = remap_hf_state_dict(load_safetensors(model_dir))
        model.load_hf_state_dict(state)
        torch_device = pick_device(device)
        torch_dtype = pick_dtype(torch_device, dtype)
        return cls(
            model,
            tokenizer,
            config,
            device=torch_device,
            dtype=torch_dtype,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            compile_model=compile_model,
        )

    def allocate_slot(self) -> int:
        for slot, used in enumerate(self._used):
            if not used:
                self._used[slot] = True
                self.cache.reset_slot(slot)
                return slot
        raise RuntimeError("no free KV-cache slots")

    def free_slot(self, slot: int) -> None:
        self._used[slot] = False
        self.cache.reset_slot(slot)

    def free_slots(self) -> int:
        return sum(1 for used in self._used if not used)

    def _forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        with self._lock, torch.inference_mode():
            return self.model(tokens, positions, cache=self.cache, slots=slots)

    def prefill_slot(self, slot: int, token_ids: list[int]) -> torch.Tensor:
        if not token_ids:
            raise ValueError("prompt is empty")
        if len(token_ids) > self.max_seq_len:
            raise ValueError(f"prompt length {len(token_ids)} exceeds max_seq_len {self.max_seq_len}")
        tokens = torch.tensor([token_ids], device=self.device, dtype=torch.long)
        positions = torch.arange(len(token_ids), device=self.device, dtype=torch.long).unsqueeze(0)
        slots = torch.tensor([slot], device=self.device, dtype=torch.long)
        logits = self._forward(tokens, positions, slots)
        return logits[0, -1].float()

    def decode_slots(self, slots: list[int], token_ids: list[int]) -> torch.Tensor:
        if not slots:
            raise ValueError("decode_slots requires at least one slot")
        batch = len(slots)
        tokens = torch.tensor(token_ids, device=self.device, dtype=torch.long).view(batch, 1)
        slot_t = torch.tensor(slots, device=self.device, dtype=torch.long)
        positions = self.cache.lengths[slot_t].view(batch, 1)
        if int(positions.max().item()) >= self.max_seq_len:
            raise ValueError("sequence would exceed max_seq_len")
        logits = self._forward(tokens, positions, slot_t)
        return logits[:, -1].float()

    def generate(self, prompt_tokens: list[int], params: SamplingParams) -> Iterator[int]:
        stop_ids = set(params.stop_token_ids) | set(self.stop_token_ids)
        max_tokens = min(params.max_tokens, self.max_seq_len - len(prompt_tokens))
        if max_tokens <= 0:
            return
        slot = self.allocate_slot()
        generator = make_generator(params.seed)
        try:
            logits = self.prefill_slot(slot, prompt_tokens)
            tokens = list(prompt_tokens)
            for _ in range(max_tokens):
                token_id = sample_token(logits, params, tokens, generator)
                if token_id in stop_ids:
                    return
                yield token_id
                tokens.append(token_id)
                logits = self.decode_slots([slot], [token_id])[0]
        finally:
            self.free_slot(slot)

    def generate_text(self, messages: list[dict[str, str]], params: SamplingParams) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        assert isinstance(prompt, list)
        ids = list(self.generate(prompt, params))
        return self.tokenizer.decode(ids)
