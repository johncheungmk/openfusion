from __future__ import annotations

import os
from importlib import resources
from pathlib import Path
from typing import Any, Literal

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
    headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("base_url")
    @classmethod
    def trim_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("timeout_seconds", "weight")
    @classmethod
    def require_positive_number(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("must be greater than zero")
        return value

    def resolved_api_key(self) -> str | None:
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        return None


class FusionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Legacy panel_judge remains accepted. parallel_synthesis is the clearer v0.2 name.
    default_strategy: str = "parallel_synthesis"
    panel: list[str] = Field(default_factory=list)
    judge_provider: str | None = None
    critic_provider: str | None = None
    reviser_provider: str | None = None
    planner_provider: str | None = None

    max_parallel: int = 4
    max_total_calls: int = 12
    samples_per_provider: int = 1
    refinement_rounds: int = 1

    temperature: float = 0.2
    judge_temperature: float = 0.1
    critique_temperature: float = 0.1
    max_tokens: int | None = 256

    require_at_least_successes: int = 1
    include_candidate_outputs: bool = True
    include_workflow_outputs: bool = True
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

        known_names = set(provider_names)
        missing_panel = [name for name in self.fusion.panel if name not in known_names]
        if missing_panel:
            raise ValueError(f"Fusion panel references unknown providers: {', '.join(missing_panel)}")

        role_references = {
            "Judge": self.fusion.judge_provider,
            "Critic": self.fusion.critic_provider,
            "Reviser": self.fusion.reviser_provider,
            "Planner": self.fusion.planner_provider,
        }
        for role, provider_name in role_references.items():
            if provider_name and provider_name not in known_names:
                raise ValueError(f"{role} provider is unknown: {provider_name}")

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
            }
        ],
        "fusion": {
            "default_strategy": "parallel_synthesis",
            "panel": ["local-ollama"],
            "judge_provider": "local-ollama",
            "max_tokens": 256,
        },
    }
    destination.write_text(yaml.safe_dump(fallback, sort_keys=False), encoding="utf-8")
