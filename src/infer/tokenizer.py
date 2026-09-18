from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import BaseLoader, Environment, StrictUndefined
from tokenizers import Tokenizer
from typing import Protocol


class ChatMessage(dict):
    """Minimal mapping used by chat templates (`role`, `content`)."""


def _raise_exception(message: str) -> None:
    raise ValueError(message)


def render_chat_template(
    template: str,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool = True,
    special_tokens: dict[str, Any] | None = None,
) -> str:
    env = Environment(
        loader=BaseLoader(),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.globals["raise_exception"] = _raise_exception
    env.filters["tojson"] = lambda value, **_kwargs: json.dumps(value, ensure_ascii=False)
    compiled = env.from_string(template)
    ctx = dict(special_tokens or {})
    return compiled.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        tools=None,
        **ctx,
    )


class HuggingFaceTokenizer:
    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        chat_template: str | None = None,
        special_tokens: dict[str, Any] | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        eos_token_ids: list[int] | None = None,
    ) -> None:
        self._tok = tokenizer
        self.chat_template = chat_template
        self.special_tokens = special_tokens or {}
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_ids = eos_token_ids or ([eos_token_id] if eos_token_id is not None else [])

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> HuggingFaceTokenizer:
        model_dir = Path(model_dir)
        tok_path = model_dir / "tokenizer.json"
        if not tok_path.exists():
            raise FileNotFoundError(
                f"{tok_path} is missing. This engine loads Hugging Face tokenizer.json files."
            )
        tokenizer = Tokenizer.from_file(str(tok_path))

        cfg_path = model_dir / "tokenizer_config.json"
        tok_cfg: dict[str, Any] = {}
        if cfg_path.exists():
            tok_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        special_path = model_dir / "special_tokens_map.json"
        special_map: dict[str, Any] = {}
        if special_path.exists():
            special_map = json.loads(special_path.read_text(encoding="utf-8"))

        special_tokens: dict[str, Any] = {}
        for key in (
            "bos_token",
            "eos_token",
            "unk_token",
            "pad_token",
            "sep_token",
            "cls_token",
            "additional_special_tokens",
        ):
            value = tok_cfg.get(key, special_map.get(key))
            if isinstance(value, dict):
                value = value.get("content")
            if value is not None:
                special_tokens[key] = value

        def _id_for(token_value: Any) -> int | None:
            if token_value is None:
                return None
            text = token_value["content"] if isinstance(token_value, dict) else str(token_value)
            tid = tokenizer.token_to_id(text)
            return int(tid) if tid is not None else None

        eos_token_id = _id_for(tok_cfg.get("eos_token") or special_tokens.get("eos_token"))
        bos_token_id = _id_for(tok_cfg.get("bos_token") or special_tokens.get("bos_token"))
        pad_token_id = _id_for(tok_cfg.get("pad_token") or special_tokens.get("pad_token"))

        added = tok_cfg.get("added_tokens_decoder") or {}
        eos_ids = []
        if eos_token_id is not None:
            eos_ids.append(eos_token_id)
        for idx, info in added.items():
            if isinstance(info, dict) and info.get("content") in {
                special_tokens.get("eos_token"),
                "<|im_end|>",
                "<|endoftext|>",
            }:
                eos_ids.append(int(idx))

        return cls(
            tokenizer,
            chat_template=tok_cfg.get("chat_template"),
            special_tokens=special_tokens,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            eos_token_ids=sorted(set(eos_ids)),
        )

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
    ) -> list[int] | str:
        if not self.chat_template:
            text = _fallback_chat_prompt(messages, add_generation_prompt=add_generation_prompt)
        else:
            text = render_chat_template(
                self.chat_template,
                messages,
                add_generation_prompt=add_generation_prompt,
                special_tokens=self.special_tokens,
            )
        if not tokenize:
            return text
        return self.encode(text)


def _fallback_chat_prompt(messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> str:
    parts: list[str] = []
    for message in messages:
        parts.append(f"{message['role']}\n{message['content']}\n")
    if add_generation_prompt:
        parts.append("assistant\n")
    return "".join(parts)


class TokenDecoder(Protocol):
    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str: ...


class IncrementalDecoder:
    """Decode token-by-token by diffing the full decoded prefix."""

    def __init__(self, tokenizer: TokenDecoder) -> None:
        self._tokenizer = tokenizer
        self._ids: list[int] = []
        self._prev = ""

    def push(self, token_id: int) -> str:
        self._ids.append(token_id)
        text = self._tokenizer.decode(self._ids)
        delta = text[len(self._prev) :]
        self._prev = text
        return delta
