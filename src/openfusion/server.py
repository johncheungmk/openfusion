from __future__ import annotations

import hmac
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from .config import AppConfig, load_config
from .fusion import FusionEngine
from .providers import ModelProvider
from .schema import OpenAIChatCompletionRequest, OpenAIModel

logger = logging.getLogger(__name__)


def create_app(config: AppConfig, providers: dict[str, ModelProvider] | None = None) -> FastAPI:
    engine = FusionEngine(config, providers=providers)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            await engine.aclose()

    app = FastAPI(title="OpenFusion", version="0.1.0", lifespan=lifespan)
    if not config.server.resolved_api_key():
        logger.warning("OpenFusion API authentication is disabled. Set OPENFUSION_API_KEY before exposing it.")

    async def require_auth(authorization: Annotated[str | None, Header()] = None) -> None:
        expected = config.server.resolved_api_key()
        if not expected:
            return
        supplied = authorization or ""
        if not hmac.compare_digest(supplied, f"Bearer {expected}"):
            raise HTTPException(status_code=401, detail="Missing or invalid bearer token")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"ok": True, "providers": list(engine.providers.keys())}

    @app.get("/v1/models")
    async def models(_: None = Depends(require_auth)) -> dict[str, object]:
        data = [OpenAIModel(id="openfusion/panel-judge").model_dump()]
        data.append(OpenAIModel(id="openfusion/fallback").model_dump())
        for provider_config in config.providers:
            if provider_config.enabled:
                data.append(
                    OpenAIModel(
                        id=f"provider/{provider_config.name}/{provider_config.model}",
                        owned_by=provider_config.name,
                    ).model_dump()
                )
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: OpenAIChatCompletionRequest,
        _: None = Depends(require_auth),
    ):
        result = await _run_completion(request)
        if request.stream:
            return StreamingResponse(
                _stream_completion_response(request.model, result),
                media_type="text/event-stream",
            )
        return _completion_response(request.model, result)

    async def _run_completion(request: OpenAIChatCompletionRequest):
        if request.model.startswith("provider/"):
            return await _run_direct_provider(request)

        strategy = request.fusion_strategy
        if strategy is None:
            strategy = "fallback" if request.model == "openfusion/fallback" else "panel_judge"

        return await engine.run(
            messages=request.messages,
            strategy=strategy,
            panel=request.fusion_panel,
            judge_provider=request.fusion_judge,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            extra_body=request.provider_extra_body(),
        )

    async def _run_direct_provider(request: OpenAIChatCompletionRequest):
        parts = request.model.split("/", 2)
        if len(parts) != 3 or not parts[1] or not parts[2]:
            raise HTTPException(
                status_code=400,
                detail="Provider model IDs must use provider/{provider_name}/{model}",
            )
        provider_name = parts[1]
        requested_model = parts[2]
        configured_providers = config.provider_map(enabled_only=False)
        provider_config = configured_providers.get(provider_name)
        if provider_config is None:
            raise HTTPException(status_code=404, detail=f"Provider not found: {provider_name}")
        if not provider_config.enabled:
            raise HTTPException(status_code=400, detail=f"Provider is disabled: {provider_name}")
        if requested_model != provider_config.model:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Provider {provider_name} is configured for model {provider_config.model}, "
                    f"not {requested_model}"
                ),
            )

        result = await engine.run_provider(
            provider_name=provider_name,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            extra_body=request.provider_extra_body(),
        )
        if not result.final:
            detail = result.candidates[0].error if result.candidates else "Provider returned no content"
            raise HTTPException(status_code=502, detail=detail or "Provider returned no content")
        return result

    def _completion_response(model: str, result) -> JSONResponse:
        created = int(time.time())
        payload = {
            "id": f"chatcmpl-openfusion-{uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.final},
                    "finish_reason": "stop",
                }
            ],
            "usage": result.usage.model_dump(),
            "openfusion": result.model_dump(),
        }
        return JSONResponse(payload)

    async def _stream_completion_response(model: str, result):
        created = int(time.time())
        completion_id = f"chatcmpl-openfusion-{uuid4().hex[:12]}"
        first_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": result.final},
                    "finish_reason": None,
                }
            ],
            "openfusion": result.model_dump(),
        }
        stop_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(first_chunk)}\n\n"
        yield f"data: {json.dumps(stop_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return app


def app_from_config_path(config_path: str) -> FastAPI:
    return create_app(load_config(config_path))
