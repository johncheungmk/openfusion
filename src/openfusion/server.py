from __future__ import annotations

import hmac
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__
from .config import AppConfig, load_config
from .fusion import FusionEngine, canonical_strategy
from .providers import ModelProvider
from .schema import FusionResult, OpenAIChatCompletionRequest, OpenAIModel

logger = logging.getLogger(__name__)

STRATEGY_MODEL_IDS = (
    "openfusion/adaptive",
    "openfusion/parallel-synthesis",
    "openfusion/panel-judge",  # backward-compatible alias
    "openfusion/critique-revision",
    "openfusion/layered-refinement",
    "openfusion/best-of-n",
    "openfusion/majority-vote",
    "openfusion/weighted-vote",
    "openfusion/fallback",
)


def create_app(config: AppConfig, providers: dict[str, ModelProvider] | None = None) -> FastAPI:
    engine = FusionEngine(config, providers=providers)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            await engine.aclose()

    app = FastAPI(title="OpenFusion", version=__version__, lifespan=lifespan)
    app.state.engine = engine
    if not config.server.resolved_api_key():
        logger.warning(
            "OpenFusion API authentication is disabled. Set OPENFUSION_API_KEY before exposing it."
        )

    async def require_auth(authorization: Annotated[str | None, Header()] = None) -> None:
        expected = config.server.resolved_api_key()
        if not expected:
            return
        supplied = authorization or ""
        if not hmac.compare_digest(supplied, f"Bearer {expected}"):
            raise HTTPException(status_code=401, detail="Missing or invalid bearer token")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "ok": True,
            "version": __version__,
            "providers": list(engine.providers.keys()),
            "strategies": list(engine.supported_strategies()),
        }

    @app.get("/v1/strategies")
    async def strategies(_: None = Depends(require_auth)) -> dict[str, object]:
        return {
            "object": "list",
            "data": [
                {"id": strategy, "object": "openfusion.strategy"}
                for strategy in engine.supported_strategies()
            ],
        }

    @app.get("/v1/models")
    async def models(_: None = Depends(require_auth)) -> dict[str, object]:
        data = [OpenAIModel(id=model_id).model_dump() for model_id in STRATEGY_MODEL_IDS]
        for provider_config in config.providers:
            if provider_config.enabled:
                data.append(
                    OpenAIModel(
                        id=f"provider/{provider_config.name}/{provider_config.model}",
                        owned_by=provider_config.name,
                    ).model_dump()
                )
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        request: OpenAIChatCompletionRequest,
        _: None = Depends(require_auth),
    ):
        try:
            result = await _run_completion(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if request.stream:
            return StreamingResponse(
                _stream_completion_response(request.model, result),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return _completion_response(request.model, result)

    async def _run_completion(request: OpenAIChatCompletionRequest) -> FusionResult:
        if request.model.startswith("provider/"):
            return await _run_direct_provider(request)

        strategy = request.fusion_strategy or _strategy_from_model(request.model)
        return await engine.run(
            messages=request.messages,
            strategy=strategy,
            panel=request.fusion_panel,
            judge_provider=request.fusion_judge,
            critic_provider=request.fusion_critic,
            reviser_provider=request.fusion_reviser,
            planner_provider=request.fusion_planner,
            temperature=request.temperature,
            max_tokens=request.effective_max_tokens(),
            extra_body=request.provider_extra_body(),
            samples_per_provider=request.fusion_samples_per_provider,
            refinement_rounds=request.fusion_refinement_rounds,
            max_total_calls=request.fusion_max_total_calls,
            vote_regex=request.fusion_vote_regex,
        )

    async def _run_direct_provider(request: OpenAIChatCompletionRequest) -> FusionResult:
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
            max_tokens=request.effective_max_tokens(),
            extra_body=request.provider_extra_body(),
        )
        if not result.final:
            detail = result.candidates[0].error if result.candidates else "Provider returned no content"
            raise HTTPException(status_code=502, detail=detail or "Provider returned no content")
        return result

    return app


def _strategy_from_model(model: str) -> str:
    if not model.startswith("openfusion/"):
        raise ValueError(
            "Unknown model ID. Use openfusion/{strategy} or provider/{provider_name}/{model}."
        )
    slug = model.split("/", 1)[1]
    return canonical_strategy(slug)


def _completion_payload(model: str, result: FusionResult) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-openfusion-{uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
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
        "system_fingerprint": f"openfusion-{__version__}",
    }


def _completion_response(model: str, result: FusionResult) -> JSONResponse:
    return JSONResponse(_completion_payload(model, result))


async def _stream_completion_response(model: str, result: FusionResult):
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
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(first_chunk)}\n\n"
    yield f"data: {json.dumps(stop_chunk)}\n\n"
    yield "data: [DONE]\n\n"


def app_from_config_path(config_path: str) -> FastAPI:
    return create_app(load_config(config_path))
