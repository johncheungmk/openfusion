from __future__ import annotations

import asyncio
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from .config import load_config, write_example_config
from .fusion import FusionEngine
from .schema import ChatMessage
from .server import create_app

app = typer.Typer(help="OpenFusion: open-source model fusion for OpenAI-compatible LLMs")
console = Console()


@app.command()
def init(path: str = typer.Option("openfusion.yaml", help="Where to write the example config.")) -> None:
    """Create an example OpenFusion config file."""
    destination = Path(path)
    if destination.exists():
        raise typer.BadParameter(f"Refusing to overwrite existing file: {destination}")
    write_example_config(destination)
    console.print(f"[green]Created[/green] {destination}")


@app.command()
def providers(config: str = typer.Option("openfusion.yaml", help="Path to config YAML.")) -> None:
    """List configured providers."""
    cfg = load_config(config)
    table = Table(title="OpenFusion providers")
    table.add_column("Name")
    table.add_column("Enabled")
    table.add_column("Model")
    table.add_column("Base URL")
    table.add_column("API key env")
    table.add_column("Key status")
    for provider in cfg.providers:
        if provider.api_key_env:
            key_status = "set" if provider.resolved_api_key() else "missing"
        else:
            key_status = "not required"
        table.add_row(
            provider.name,
            str(provider.enabled),
            provider.model,
            provider.base_url,
            provider.api_key_env or "-",
            key_status,
        )
    console.print(table)


@app.command()
def chat(
    prompt: str = typer.Argument(..., help="User prompt."),
    config: str = typer.Option("openfusion.yaml", help="Path to config YAML."),
    strategy: str | None = typer.Option(None, help="panel_judge or fallback."),
    panel: str | None = typer.Option(None, help="Comma-separated provider names."),
    judge: str | None = typer.Option(None, help="Provider name to use as judge."),
    max_tokens: int | None = typer.Option(None, help="Max output tokens."),
) -> None:
    """Run a single fused chat completion."""
    cfg = load_config(config)
    engine = FusionEngine(cfg)
    panel_names = [item.strip() for item in panel.split(",")] if panel else None

    async def _run() -> None:
        try:
            result = await engine.run(
                messages=[ChatMessage(role="user", content=prompt)],
                strategy=strategy,
                panel=panel_names,
                judge_provider=judge,
                max_tokens=max_tokens,
            )
            console.print("\n[bold]Final answer[/bold]\n")
            console.print(result.final)
            console.print("\n[bold]Candidates[/bold]")
            for candidate in result.candidates:
                status = "ok" if candidate.ok else f"error: {candidate.error}"
                console.print(f"- {candidate.provider} / {candidate.model}: {status}")
        finally:
            await engine.aclose()

    asyncio.run(_run())


@app.command()
def serve(
    config: str = typer.Option("openfusion.yaml", help="Path to config YAML."),
    host: str | None = typer.Option(None, help="Override host."),
    port: int | None = typer.Option(None, help="Override port."),
    reload: bool = typer.Option(False, help="Enable uvicorn reload for development."),
) -> None:
    """Start the OpenAI-compatible OpenFusion API server."""
    cfg = load_config(config)
    api = create_app(cfg)
    uvicorn.run(api, host=host or cfg.server.host, port=port or cfg.server.port, reload=reload)


if __name__ == "__main__":
    app()
