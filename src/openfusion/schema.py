from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


Role = Literal["system", "user", "assistant", "tool"]
MessageContent = str | list[dict[str, Any]]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Role
    content: MessageContent | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ProviderRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float = 0.2
    max_tokens: int | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class CandidateResult(BaseModel):
    provider: str
    model: str
    weight: float = 1.0
    content: str = ""
    ok: bool = True
    error: str | None = None
    latency_ms: int | None = None
    usage: Usage = Field(default_factory=Usage)


class FusionResult(BaseModel):
    strategy: str
    final: str
    judge_provider: str | None = None
    judge_analysis: str | None = None
    candidates: list[CandidateResult] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


class OpenAIChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "openfusion/panel-judge"
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    top_p: float | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None
    n: int | None = None
    # OpenFusion extensions. Standard OpenAI clients ignore or can pass these via extra_body.
    fusion_strategy: str | None = None
    fusion_panel: list[str] | None = None
    fusion_judge: str | None = None

    def provider_extra_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {}
        for field_name in (
            "top_p",
            "stop",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "user",
            "n",
        ):
            value = getattr(self, field_name)
            if value is not None:
                body[field_name] = value
        return body


class OpenAIModel(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "openfusion"
