from __future__ import annotations

import math
import os
from importlib import resources
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: Literal["openai_compatible"] = "openai_compatible"
    enabled: bool = True
    base_url: str
    api_key_env: str | None = None
    model: str
    timeout_seconds: float = 90
    weight: float = 1.0
    input_cost_per_million_tokens_usd: float | None = None
    output_cost_per_million_tokens_usd: float | None = None
    headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        if "/" in stripped:
            raise ValueError("must not contain '/' because provider names are used in model IDs")
        return stripped

    @field_validator("base_url")
    @classmethod
    def trim_slash(cls, value: str) -> str:
        value = value.rstrip("/")
        try:
            parsed = urlsplit(value)
        except ValueError as exc:
            raise ValueError("must be a valid URL") from exc
        if parsed.username is not None or parsed.password is not None:
            raise ValueError(
                "must not contain credentials; use api_key_env or configured headers"
            )
        if parsed.query or parsed.fragment:
            raise ValueError("must not contain a query string or fragment")
        return value

    @field_validator("timeout_seconds", "weight")
    @classmethod
    def require_positive_number(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("must be greater than zero")
        return value

    @field_validator(
        "input_cost_per_million_tokens_usd",
        "output_cost_per_million_tokens_usd",
    )
    @classmethod
    def require_nonnegative_cost(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("must be a finite number that is zero or greater")
        return value

    def resolved_api_key(self) -> str | None:
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        return None

    def estimate_cost_usd(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float | None:
        if (
            self.input_cost_per_million_tokens_usd is None
            or self.output_cost_per_million_tokens_usd is None
        ):
            return None
        return (
            max(0, prompt_tokens) * self.input_cost_per_million_tokens_usd
            + max(0, completion_tokens) * self.output_cost_per_million_tokens_usd
        ) / 1_000_000


class PanelRoleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    instruction: str

    @field_validator("name", "instruction")
    @classmethod
    def require_nonempty_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped


class FusionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Legacy panel_judge remains accepted. parallel_synthesis is the clearer v0.2 name.
    default_strategy: str = "parallel_synthesis"
    panel: list[str] = Field(default_factory=list)
    panel_roles: list[PanelRoleConfig] = Field(default_factory=list)
    judge_provider: str | None = None
    critic_provider: str | None = None
    reviser_provider: str | None = None
    planner_provider: str | None = None
    self_moa_provider: str | None = None
    ranker_provider: str | None = None
    fuser_provider: str | None = None
    vote_equivalence_provider: str | None = None
    cascade_providers: list[str] = Field(default_factory=list)

    max_parallel: int = 4
    max_total_calls: int = 12
    samples_per_provider: int = 1
    refinement_rounds: int = 1
    self_moa_samples: int = 3
    self_moa_batch_size: int = 4
    self_moa_seq_carry_max_chars: int = 12000
    rank_top_k: int = 3
    pairwise_rank_max_pairs: int = 12
    semantic_vote_max_pairs: int = 12
    cascade_consistency_samples: int = 1
    cascade_max_steps: int = 3

    temperature: float = 0.2
    judge_temperature: float = 0.1
    critique_temperature: float = 0.1
    self_moa_temperature: float = 0.7
    max_tokens: int | None = 256
    self_moa_mode: Literal["select", "synthesize"] = "synthesize"
    pairwise_rank_mode: Literal["pairwise", "score"] = "pairwise"
    semantic_vote_mode: Literal["rule_only", "llm_equivalence"] = "rule_only"
    cascade_confidence_threshold: float = 0.75
    cascade_escalate_on_disagreement: bool = True

    require_at_least_successes: int = 1
    include_candidate_outputs: bool = True
    include_workflow_outputs: bool = True
    structured_synthesis: bool = False
    judge_candidate_max_chars: int = 4000
    transcript_max_chars: int = 12000
    vote_answer_regex: str | None = None

    # When false, adaptive mode uses transparent local heuristics only. When true,
    # planner_provider may produce a constrained JSON plan before execution.
    adaptive_use_model_planner: bool = False

    @field_validator(
        "max_parallel",
        "max_total_calls",
        "samples_per_provider",
        "self_moa_samples",
        "self_moa_batch_size",
        "self_moa_seq_carry_max_chars",
        "rank_top_k",
        "pairwise_rank_max_pairs",
        "semantic_vote_max_pairs",
        "cascade_consistency_samples",
        "cascade_max_steps",
        "require_at_least_successes",
        "judge_candidate_max_chars",
        "transcript_max_chars",
    )
    @classmethod
    def require_positive_integer(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be at least 1")
        return value

    @field_validator("refinement_rounds")
    @classmethod
    def require_nonnegative_rounds(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be at least 0")
        return value

    @field_validator("cascade_confidence_threshold")
    @classmethod
    def require_probability(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("must be between 0 and 1")
        return value


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = 8000
    api_key_env: str | None = None

    def resolved_api_key(self) -> str | None:
        return os.getenv(self.api_key_env) if self.api_key_env else None


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    providers: list[ProviderConfig]
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @model_validator(mode="after")
    def validate_provider_references(self) -> "AppConfig":
        provider_names = [provider.name for provider in self.providers]
        duplicate_names = sorted(
            {name for name in provider_names if provider_names.count(name) > 1}
        )
        if duplicate_names:
            raise ValueError(f"Duplicate provider names: {', '.join(duplicate_names)}")

        enabled_names = {provider.name for provider in self.providers if provider.enabled}
        all_names = set(provider_names)
        unknown_panel = [name for name in self.fusion.panel if name not in all_names]
        if unknown_panel:
            raise ValueError(
                f"Fusion panel references unknown providers: {', '.join(unknown_panel)}"
            )
        disabled_panel = [name for name in self.fusion.panel if name not in enabled_names]
        if disabled_panel:
            raise ValueError(
                f"Fusion panel references disabled providers: {', '.join(disabled_panel)}"
            )
        unknown_cascade = [
            name for name in self.fusion.cascade_providers if name not in all_names
        ]
        if unknown_cascade:
            raise ValueError(
                f"Cascade providers reference unknown providers: {', '.join(unknown_cascade)}"
            )
        disabled_cascade = [
            name for name in self.fusion.cascade_providers if name not in enabled_names
        ]
        if disabled_cascade:
            raise ValueError(
                f"Cascade providers reference disabled providers: {', '.join(disabled_cascade)}"
            )

        role_references = {
            "Judge": self.fusion.judge_provider,
            "Critic": self.fusion.critic_provider,
            "Reviser": self.fusion.reviser_provider,
            "Planner": self.fusion.planner_provider,
            "Self-MoA": self.fusion.self_moa_provider,
            "Ranker": self.fusion.ranker_provider,
            "Fuser": self.fusion.fuser_provider,
            "Vote equivalence": self.fusion.vote_equivalence_provider,
        }
        for role, provider_name in role_references.items():
            if provider_name and provider_name not in all_names:
                raise ValueError(f"{role} provider is unknown: {provider_name}")
            if provider_name and provider_name not in enabled_names:
                raise ValueError(f"{role} provider is disabled: {provider_name}")

        return self

    def provider_map(self, enabled_only: bool = True) -> dict[str, ProviderConfig]:
        providers = self.providers
        if enabled_only:
            providers = [provider for provider in providers if provider.enabled]
        return {provider.name: provider for provider in providers}


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    load_dotenv(config_path.with_name(".env"))
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML object: {config_path}")
    return AppConfig.model_validate(raw)


def write_example_config(path: str | Path) -> None:
    template_path = Path(__file__).resolve().parents[2] / "config.example.yaml"
    destination = Path(path)
    if template_path.exists():
        destination.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")
        return
    packaged_template = resources.files("openfusion").joinpath("config.example.yaml")
    if packaged_template.is_file():
        destination.write_text(packaged_template.read_text(encoding="utf-8"), encoding="utf-8")
        return
    fallback: dict[str, Any] = {
        "providers": [
            {
                "name": "local-ollama",
                "type": "openai_compatible",
                "enabled": True,
                "base_url": "http://localhost:11434/v1",
                "api_key_env": "OLLAMA_API_KEY",
                "model": "llama3.2:3b",
                "timeout_seconds": 300,
                "weight": 1.0,
                "input_cost_per_million_tokens_usd": 0,
                "output_cost_per_million_tokens_usd": 0,
            }
        ],
        "fusion": {
            "default_strategy": "parallel_synthesis",
            "panel": ["local-ollama"],
            "panel_roles": [],
            "judge_provider": "local-ollama",
            "self_moa_samples": 3,
            "self_moa_mode": "synthesize",
            "rank_top_k": 3,
            "pairwise_rank_max_pairs": 12,
            "pairwise_rank_mode": "pairwise",
            "semantic_vote_max_pairs": 12,
            "semantic_vote_mode": "rule_only",
            "cascade_providers": ["local-ollama"],
            "cascade_confidence_threshold": 0.75,
            "cascade_consistency_samples": 1,
            "cascade_escalate_on_disagreement": True,
            "cascade_max_steps": 3,
            "structured_synthesis": False,
            "max_tokens": 256,
        },
    }
    destination.write_text(yaml.safe_dump(fallback, sort_keys=False), encoding="utf-8")
