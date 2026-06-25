from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from .config import load_config, write_example_config
from .evaluation import evaluate_cases, load_jsonl
from .fusion import FusionEngine
from .schema import ChatMessage
from .server import create_app

app = typer.Typer(help="OpenFusion: open-source multi-model orchestration and fusion")
console = Console()

STRATEGY_HELP = {
    "fallback": "Try providers in order until one succeeds.",
    "parallel_synthesis": "Independent drafts followed by generative synthesis.",
    "best_of_n": "Generate alternatives and select the strongest unchanged answer.",
    "majority_vote": "Exact/regex-normalized consensus by vote count.",
    "weighted_vote": "Exact/regex-normalized consensus using provider weights.",
    "critique_revision": "Independent drafts, critic feedback, then a new revision.",
    "layered_refinement": "Mixture-of-agents-style refinement layers plus synthesis.",
    "adaptive": "Constrained heuristic or optional model-generated workflow plan.",
}


def _panel_names(panel: str | None) -> list[str] | None:
    return [item.strip() for item in panel.split(",") if item.strip()] if panel else None


@app.command()
def init(path: str = typer.Option("openfusion.yaml", help="Where to write the example config.")) -> None:
    """Create an example OpenFusion config file."""
    destination = Path(path)
    if destination.exists():
        raise typer.BadParameter(f"Refusing to overwrite existing file: {destination}")
    write_example_config(destination)
    console.print(f"[green]Created[/green] {destination}")
    console.print("Edit its model names to match `ollama list` or your cloud gateway.")


@app.command()
def providers(config: str = typer.Option("openfusion.yaml", help="Path to config YAML.")) -> None:
    """List configured providers."""
    cfg = load_config(config)
    table = Table(title="OpenFusion providers")
    for column in ("Name", "Enabled", "Model", "Base URL", "Weight", "API key env", "Key status"):
        table.add_column(column)
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
            f"{provider.weight:g}",
            provider.api_key_env or "-",
            key_status,
        )
    console.print(table)


@app.command()
def strategies() -> None:
    """List orchestration strategies."""
    table = Table(title="OpenFusion strategies")
    table.add_column("Strategy")
    table.add_column("Behavior")
    for name, description in STRATEGY_HELP.items():
        table.add_row(name, description)
    console.print(table)


@app.command("plan")
def plan_command(
    prompt: str = typer.Argument(..., help="User prompt to classify."),
    config: str = typer.Option("openfusion.yaml", help="Path to config YAML."),
    panel: str | None = typer.Option(None, help="Comma-separated provider names."),
    planner: str | None = typer.Option(None, help="Optional provider for model planning."),
    model_planner: bool = typer.Option(
        False,
        "--model-planner",
        help="Call the configured planner model instead of heuristics only.",
    ),
    max_total_calls: int | None = typer.Option(None, help="Workflow call budget."),
) -> None:
    """Preview the constrained adaptive workflow plan."""
    cfg = load_config(config)
    engine = FusionEngine(cfg)

    async def _run() -> None:
        try:
            plan, trace = await engine.plan(
                messages=[ChatMessage(role="user", content=prompt)],
                panel=_panel_names(panel),
                planner_provider=planner,
                max_total_calls=max_total_calls,
                use_model_planner=model_planner,
            )
            console.print_json(json.dumps(plan.model_dump()))
            if trace:
                console.print("\n[bold]Planning trace[/bold]")
                for step in trace:
                    console.print(
                        f"- {step.stage}: {step.provider or '-'} — {step.status}"
                        + (f" ({step.note})" if step.note else "")
                    )
        finally:
            await engine.aclose()

    asyncio.run(_run())


@app.command()
def chat(
    prompt: str = typer.Argument(..., help="User prompt."),
    config: str = typer.Option("openfusion.yaml", help="Path to config YAML."),
    strategy: str | None = typer.Option(None, help="See `openfusion strategies`."),
    panel: str | None = typer.Option(None, help="Comma-separated provider names."),
    judge: str | None = typer.Option(None, help="Provider used for selection/synthesis."),
    critic: str | None = typer.Option(None, help="Provider used for critique."),
    reviser: str | None = typer.Option(None, help="Provider used for revision."),
    planner: str | None = typer.Option(None, help="Provider used by adaptive model planning."),
    samples: int | None = typer.Option(None, help="Independent samples per provider."),
    rounds: int | None = typer.Option(None, help="Refinement rounds."),
    max_tokens: int | None = typer.Option(None, help="Maximum generated tokens per call."),
    max_total_calls: int | None = typer.Option(None, help="Hard model-call budget."),
    show_trace: bool = typer.Option(False, "--show-trace", help="Print workflow trace."),
) -> None:
    """Run one orchestrated chat completion."""
    cfg = load_config(config)
    engine = FusionEngine(cfg)

    async def _run() -> None:
        try:
            result = await engine.run(
                messages=[ChatMessage(role="user", content=prompt)],
                strategy=strategy,
                panel=_panel_names(panel),
                judge_provider=judge,
                critic_provider=critic,
                reviser_provider=reviser,
                planner_provider=planner,
                samples_per_provider=samples,
                refinement_rounds=rounds,
                max_tokens=max_tokens,
                max_total_calls=max_total_calls,
            )
            console.print("\n[bold]Final answer[/bold]\n")
            console.print(result.final)
            if result.plan:
                console.print(
                    f"\n[bold]Plan[/bold]: {result.plan.strategy} — {result.plan.rationale}"
                )
            console.print("\n[bold]Candidates[/bold]")
            for candidate in result.candidates:
                status = "ok" if candidate.ok else f"error: {candidate.error}"
                console.print(
                    f"- {candidate.stage}: {candidate.provider} / {candidate.model} "
                    f"sample={candidate.sample_index}: {status}"
                )
            if show_trace:
                console.print("\n[bold]Trace[/bold]")
                for step in result.trace:
                    console.print(
                        f"- {step.stage}: {step.provider or '-'} — {step.status}"
                        + (f" ({step.latency_ms} ms)" if step.latency_ms is not None else "")
                        + (f" — {step.note}" if step.note else "")
                    )
        finally:
            await engine.aclose()

    asyncio.run(_run())


@app.command()
def evaluate(
    dataset: str = typer.Argument(..., help="JSONL evaluation dataset."),
    config: str = typer.Option("openfusion.yaml", help="Path to config YAML."),
    strategy: str = typer.Option("fallback", help="Strategy to evaluate."),
    panel: str | None = typer.Option(None, help="Comma-separated provider names."),
    judge: str | None = typer.Option(None, help="Judge/selector provider."),
    max_tokens: int | None = typer.Option(None, help="Maximum generated tokens per call."),
    max_total_calls: int | None = typer.Option(None, help="Per-case model-call budget."),
    output: str | None = typer.Option(None, help="Optional JSON report path."),
) -> None:
    """Run an exact-match JSONL evaluation without claiming benchmark gains in advance."""
    cfg = load_config(config)
    engine = FusionEngine(cfg)
    cases = load_jsonl(dataset)

    async def _run() -> None:
        try:
            summary = await evaluate_cases(
                engine=engine,
                cases=cases,
                strategy=strategy,
                panel=_panel_names(panel),
                judge_provider=judge,
                max_tokens=max_tokens,
                max_total_calls=max_total_calls,
            )
            console.print(
                f"[bold]Accuracy[/bold]: {summary.correct}/{summary.total} "
                f"({summary.accuracy:.1%})"
            )
            report = summary.model_dump_json(indent=2)
            if output:
                Path(output).write_text(report, encoding="utf-8")
                console.print(f"[green]Wrote[/green] {output}")
            else:
                console.print_json(report)
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
