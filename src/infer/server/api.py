from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from sse_starlette.sse import EventSourceResponse

from infer.engine import Engine
from infer.sampling import SamplingParams
from infer.scheduler import Scheduler
from infer.server.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessageOut,
    ChoiceOut,
    DeltaOut,
    ModelCard,
    ModelList,
    StreamChoice,
    UsageOut,
    request_to_messages,
    stop_strings,
)
from infer.tokenizer import IncrementalDecoder


def _params_from_request(req: ChatCompletionRequest, engine: Engine) -> SamplingParams:
    penalty = req.repetition_penalty
    if req.frequency_penalty:
        penalty = 1.0 + req.frequency_penalty
    return SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=penalty,
        stop_token_ids=list(engine.stop_token_ids),
        seed=req.seed,
    )


def create_app(engine: Engine, scheduler: Scheduler) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        scheduler.start()
        yield
        scheduler.stop()

    app = FastAPI(title="infer", version="0.1.0", lifespan=lifespan)
    app.state.engine = engine
    app.state.scheduler = scheduler

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models() -> ModelList:
        model_id = engine.config.model_name or "local"
        return ModelList(data=[ModelCard(id=model_id)])

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        messages = request_to_messages(req)
        prompt = engine.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        if not isinstance(prompt, list):
            raise HTTPException(status_code=500, detail="tokenizer did not return token ids")
        if not prompt:
            raise HTTPException(status_code=400, detail="empty prompt")
        params = _params_from_request(req, engine)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        model_id = req.model or engine.config.model_name
        stops = stop_strings(req.stop)

        if req.stream:
            return EventSourceResponse(
                _stream_completion(
                    engine,
                    scheduler,
                    prompt,
                    params,
                    completion_id,
                    created,
                    model_id,
                    stops,
                )
            )

        out = scheduler.submit(prompt, params)
        token_ids: list[int] = []
        decoder = IncrementalDecoder(engine.tokenizer)
        text_parts: list[str] = []
        while True:
            item = await asyncio.to_thread(out.get)
            if item is None:
                break
            token_ids.append(item)
            text_parts.append(decoder.push(item))
            joined = "".join(text_parts)
            if stops and any(stop in joined for stop in stops):
                break
        text = "".join(text_parts)
        for stop in stops:
            idx = text.find(stop)
            if idx >= 0:
                text = text[:idx]
                break
        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=model_id,
            choices=[
                ChoiceOut(
                    index=0,
                    message=ChatMessageOut(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
            usage=UsageOut(
                prompt_tokens=len(prompt),
                completion_tokens=len(token_ids),
                total_tokens=len(prompt) + len(token_ids),
            ),
        )

    return app


async def _stream_completion(
    engine: Engine,
    scheduler: Scheduler,
    prompt: list[int],
    params: SamplingParams,
    completion_id: str,
    created: int,
    model_id: str,
    stops: list[str],
) -> AsyncIterator[dict[str, str]]:
    first = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=model_id,
        choices=[StreamChoice(delta=DeltaOut(role="assistant", content=""), finish_reason=None)],
    )
    yield {"data": first.model_dump_json()}

    out = scheduler.submit(prompt, params)
    decoder = IncrementalDecoder(engine.tokenizer)
    produced = ""
    while True:
        item = await asyncio.to_thread(out.get)
        if item is None:
            break
        piece = decoder.push(item)
        produced += piece
        emit = piece
        done = False
        for stop in stops:
            idx = produced.find(stop)
            if idx >= 0:
                extra = len(produced) - idx
                keep = max(0, len(piece) - extra)
                emit = piece[:keep]
                done = True
                break
        if emit:
            chunk = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=model_id,
                choices=[StreamChoice(delta=DeltaOut(content=emit), finish_reason=None)],
            )
            yield {"data": chunk.model_dump_json()}
        if done:
            break

    final = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=model_id,
        choices=[StreamChoice(delta=DeltaOut(), finish_reason="stop")],
    )
    yield {"data": final.model_dump_json()}
    yield {"data": "[DONE]"}
