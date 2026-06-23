from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from openfusion.config import load_config


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
