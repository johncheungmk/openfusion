from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from openfusion.config import AppConfig, FusionConfig, ProviderConfig, load_config


def test_load_config(tmp_path: Path) -> None:
    path = tmp_path / "openfusion.yaml"
    path.write_text(
        """
providers:
  - name: local
    type: openai_compatible
    base_url: http://localhost:11434/v1/
    model: qwen
fusion:
  panel: [local]
""".strip(),
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.providers[0].base_url == "http://localhost:11434/v1"
    assert config.fusion.panel == ["local"]


def test_load_example_config() -> None:
    path = Path(__file__).resolve().parents[1] / "config.example.yaml"

    config = load_config(path)

    assert config.fusion.panel == ["local-ollama"]
    assert config.fusion.judge_provider == "local-ollama"


def test_inline_api_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "openfusion.yaml"
    path.write_text(
        """
providers:
  - name: local
    type: openai_compatible
    base_url: http://localhost:11434/v1
    api_key: do-not-put-secrets-here
    model: qwen
fusion:
  panel: [local]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(path)


def test_unknown_provider_reference_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "openfusion.yaml"
    path.write_text(
        """
providers:
  - name: local
    type: openai_compatible
    base_url: http://localhost:11434/v1
    model: qwen
fusion:
  panel: [missing]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(path)


def test_disabled_provider_reference_is_rejected() -> None:
    disabled = ProviderConfig(
        name="disabled",
        enabled=False,
        base_url="http://disabled",
        model="model",
    )

    with pytest.raises(ValidationError, match="disabled providers: disabled"):
        AppConfig(
            providers=[disabled],
            fusion=FusionConfig(panel=["disabled"]),
        )


def test_provider_names_and_pricing_are_validated() -> None:
    with pytest.raises(ValidationError, match="must not contain"):
        ProviderConfig(name="bad/name", base_url="http://local", model="model")
    with pytest.raises(ValidationError, match="zero or greater"):
        ProviderConfig(
            name="local",
            base_url="http://local",
            model="model",
            input_cost_per_million_tokens_usd=-1,
        )

    priced = ProviderConfig(
        name="local",
        base_url="http://local",
        model="model",
        input_cost_per_million_tokens_usd=1,
        output_cost_per_million_tokens_usd=2,
    )
    assert priced.estimate_cost_usd(prompt_tokens=2, completion_tokens=3) == 0.000008


def test_provider_base_url_rejects_embedded_credentials() -> None:
    with pytest.raises(ValidationError, match="must not contain credentials"):
        ProviderConfig(
            name="local",
            base_url="https://alice:supersecret@example.test/v1",
            model="model",
        )
    with pytest.raises(ValidationError, match="query string or fragment"):
        ProviderConfig(
            name="local",
            base_url="https://example.test/v1?api_key=supersecret",
            model="model",
        )


def test_load_config_reads_dotenv_next_to_config(tmp_path: Path) -> None:
    os.environ.pop("OPENFUSION_API_KEY", None)
    path = tmp_path / "openfusion.yaml"
    path.write_text(
        """
providers:
  - name: local
    type: openai_compatible
    base_url: http://localhost:11434/v1
    model: qwen
fusion:
  panel: [local]
server:
  api_key_env: OPENFUSION_API_KEY
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("OPENFUSION_API_KEY=local-test-key\n", encoding="utf-8")

    try:
        config = load_config(path)

        assert config.server.resolved_api_key() == "local-test-key"
    finally:
        os.environ.pop("OPENFUSION_API_KEY", None)
