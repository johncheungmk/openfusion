from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


Role = Literal["system", "developer", "user", "assistant", "tool"]
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
    stage: str = "draft"
    sample_index: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowStep(BaseModel):
    stage: str
    provider: str | None = None
    model: str | None = None
    status: Literal["ok", "error", "skipped", "fallback"] = "ok"
    latency_ms: int | None = None
    note: str | None = None


class OrchestrationPlan(BaseModel):
    strategy: str
    panel: list[str] = Field(default_factory=list)
    judge_provider: str | None = None
    critic_provider: str | None = None
    reviser_provider: str | None = None
    samples_per_provider: int = 1
    refinement_rounds: int = 0
    max_total_calls: int = 12
    estimated_calls: int = 1
    source: Literal["request", "heuristic", "model"] = "request"
    rationale: str = "Explicit user-selected workflow."


class FusionResult(BaseModel):
    strategy: str
    final: str
    judge_provider: str | None = None
    judge_analysis: str | None = None
    critic_provider: str | None = None
    reviser_provider: str | None = None
    candidates: list[CandidateResult] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    plan: OrchestrationPlan | None = None
    trace: list[WorkflowStep] = Field(default_factory=list)
    workflow_outputs: dict[str, str] = Field(default_factory=dict)


class OpenAIChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "openfusion/parallel-synthesis"
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False
    top_p: float | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None
    n: int | None = None
    response_format: dict[str, Any] | None = None

    # OpenFusion extensions. OpenAI SDK users can pass these through extra_body.
    fusion_strategy: str | None = None
    fusion_panel: list[str] | None = None
    fusion_judge: str | None = None
    fusion_critic: str | None = None
    fusion_reviser: str | None = None
    fusion_planner: str | None = None
    fusion_samples_per_provider: int | None = None
    fusion_refinement_rounds: int | None = None
    fusion_max_total_calls: int | None = None
    fusion_vote_regex: str | None = None

    def effective_max_tokens(self) -> int | None:
        if self.max_completion_tokens is not None:
            return self.max_completion_tokens
        return self.max_tokens

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
            "response_format",
        ):
            value = getattr(self, field_name)
            if value is not None:
                body[field_name] = value
        return body


class OpenAIModel(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "openfusion"
