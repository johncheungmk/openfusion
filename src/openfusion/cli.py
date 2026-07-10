from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from .config import load_config, write_example_config
from .evaluation import compare_strategies, evaluate_cases, load_jsonl
from .fusion import FusionEngine
from .lab import (
    build_engine_plan,
    export_result_card,
    load_lab_config,
    load_lab_dataset,
    load_result_card,
    recommend_from_card,
    run_lab_experiment_sync,
    search_huggingface_models,
    write_generated_config,
)
from .schema import ChatMessage
from .server import create_app

app = typer.Typer(help="OpenFusion: open-source multi-model orchestration and fusion")
lab_app = typer.Typer(help="OpenFusion Lab experiment runner")
app.add_typer(lab_app, name="lab")
console = Console()

STRATEGY_HELP = {
    "fallback": "Try providers in order until one succeeds.",
    "parallel_synthesis": "Independent drafts followed by generative synthesis.",
    "self_moa": "Sample one provider multiple times, then select or synthesize.",
    "self_moa_seq": "Sequential batched Self-MoA with carry-forward aggregation.",
    "pairwise_rank_fuse": "Rank candidates by pairwise/score judging, then fuse top answers.",
    "semantic_vote": "Group concise semantically equivalent answers before voting.",
    "uncertainty_cascade": "Start cheap and escalate on low confidence, disagreement, or failure.",
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
    compare_strategies_option: str | None = typer.Option(
        None,
        "--compare-strategies",
        help="Comma-separated strategies to evaluate with the same call budget.",
    ),
    panel: str | None = typer.Option(None, help="Comma-separated provider names."),
    judge: str | None = typer.Option(None, help="Judge/selector provider."),
    grader: str = typer.Option(
        "exact_match",
        help=(
            "Grader: exact_match, regex, llm_pairwise, llm_pairwise_swap, or llm_rubric."
        ),
    ),
    grader_provider: str | None = typer.Option(None, help="Provider used by LLM graders."),
    max_tokens: int | None = typer.Option(None, help="Maximum generated tokens per call."),
    max_total_calls: int | None = typer.Option(None, help="Per-case model-call budget."),
    output: str | None = typer.Option(None, help="Optional JSON report path."),
) -> None:
    """Run JSONL evaluation without claiming benchmark gains in advance."""
    cfg = load_config(config)
    engine = FusionEngine(cfg)
    cases = load_jsonl(dataset)

    async def _run() -> None:
        try:
            if grader not in {
                "exact_match",
                "regex",
                "llm_pairwise",
                "llm_pairwise_swap",
                "llm_rubric",
            }:
                raise typer.BadParameter(
                    "grader must be exact_match, regex, llm_pairwise, "
                    "llm_pairwise_swap, or llm_rubric"
                )
            if compare_strategies_option:
                report_model = await compare_strategies(
                    engine=engine,
                    cases=cases,
                    strategies=_panel_names(compare_strategies_option) or [],
                    panel=_panel_names(panel),
                    judge_provider=judge,
                    max_tokens=max_tokens,
                    max_total_calls=max_total_calls,
                    grader=grader,  # type: ignore[arg-type]
                    grader_provider=grader_provider,
                )
                console.print(
                    f"[bold]Compared[/bold] {len(report_model.summaries)} strategies "
                    f"with baseline {report_model.baseline_strategy}"
                )
            else:
                report_model = await evaluate_cases(
                    engine=engine,
                    cases=cases,
                    strategy=strategy,
                    panel=_panel_names(panel),
                    judge_provider=judge,
                    max_tokens=max_tokens,
                    max_total_calls=max_total_calls,
                    grader=grader,  # type: ignore[arg-type]
                    grader_provider=grader_provider,
                )
                console.print(
                    f"[bold]Accuracy[/bold]: {report_model.correct}/{report_model.total} "
                    f"({report_model.accuracy:.1%})"
                )
            report = report_model.model_dump_json(indent=2)
            if output:
                Path(output).write_text(report, encoding="utf-8")
                console.print(f"[green]Wrote[/green] {output}")
            else:
                console.print_json(report)
        finally:
            await engine.aclose()

    asyncio.run(_run())


@lab_app.command("validate")
def lab_validate(lab_yaml: str = typer.Argument(..., help="Path to lab.yaml.")) -> None:
    """Validate an OpenFusion Lab experiment file."""
    cfg = load_lab_config(lab_yaml)
    configured_dataset_path = Path(cfg.dataset.path)
    dataset_path = (
        configured_dataset_path
        if configured_dataset_path.exists()
        else Path(lab_yaml).resolve().parent / cfg.dataset.path
    )
    examples = load_lab_dataset(
        dataset_path,
        max_examples=cfg.experiment.max_examples,
        seed=cfg.experiment.seed,
    )
    console.print(f"[bold]Experiment[/bold]: {cfg.experiment.name}")
    if cfg.experiment.description:
        console.print(cfg.experiment.description)
    console.print(f"[bold]Dataset[/bold]: {cfg.dataset.name} ({cfg.dataset.path})")
    console.print(f"[bold]Examples[/bold]: {len(examples)}")

    engine_table = Table(title="Lab engines")
    for column in ("Name", "Type", "Base URL", "Launch"):
        engine_table.add_column(column)
    for engine in cfg.engines:
        engine_table.add_row(engine.name, engine.type, engine.base_url, engine.launch)
    console.print(engine_table)

    model_table = Table(title="Lab models")
    for column in ("Provider", "Engine", "Model", "Weight", "Timeout"):
        model_table.add_column(column)
    for model in cfg.models:
        model_table.add_row(
            model.provider_name,
            model.engine,
            model.model,
            f"{model.weight:g}",
            f"{model.timeout_seconds:g}s",
        )
    console.print(model_table)
    console.print("[bold]Strategies[/bold]: " + ", ".join(strategy.name for strategy in cfg.strategies))


@lab_app.command("generate-config")
def lab_generate_config(
    lab_yaml: str = typer.Argument(..., help="Path to lab.yaml."),
    out: str = typer.Option("openfusion.lab.generated.yaml", help="Generated config path."),
) -> None:
    """Generate an OpenFusion runtime config from lab.yaml."""
    cfg = load_lab_config(lab_yaml)
    write_generated_config(cfg, out)
    console.print(f"[green]Wrote[/green] {out}")


@lab_app.command("run")
def lab_run(
    lab_yaml: str = typer.Argument(..., help="Path to lab.yaml."),
    out: str = typer.Option("results.json", help="Result card path."),
    include_samples: bool = typer.Option(
        False,
        "--include-samples",
        help="Include per-example result IDs and metrics. Raw prompts and references are never included.",
    ),
) -> None:
    """Run a lab experiment and save a result-card JSON file."""
    cfg = load_lab_config(lab_yaml)
    card = run_lab_experiment_sync(cfg, lab_path=lab_yaml, include_samples=include_samples)
    Path(out).write_text(card.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"[green]Wrote[/green] {out}")
    _print_lab_report(card)


@lab_app.command("recommend")
def lab_recommend(results_json: str = typer.Argument(..., help="Path to result card JSON.")) -> None:
    """Print strategy recommendations from a result card."""
    card = load_result_card(results_json)
    recommendation = recommend_from_card(card)
    card.recommendations = recommendation
    _print_lab_report(card)


@lab_app.command("export")
def lab_export(
    results_json: str = typer.Argument(..., help="Path to result card JSON."),
    out: str = typer.Option("result-card.json", help="Normalized result-card path."),
) -> None:
    """Normalize and write a shareable result card."""
    export_result_card(results_json, out)
    console.print(f"[green]Wrote[/green] {out}")


@lab_app.command("search-models")
def lab_search_models(
    query: str | None = typer.Option(None, "--query", help="Search text."),
    limit: int = typer.Option(10, "--limit", min=1, max=100, help="Maximum models."),
    license: str | None = typer.Option(None, "--license", help="License filter."),
    sort: str = typer.Option(
        "downloads",
        "--sort",
        help="Sort by downloads, likes, or lastModified.",
    ),
) -> None:
    """Search the Hugging Face model catalog without downloading models."""
    if sort not in {"downloads", "likes", "lastModified"}:
        raise typer.BadParameter("sort must be downloads, likes, or lastModified")
    models = search_huggingface_models(
        query=query,
        limit=limit,
        license=license,
        sort=sort,  # type: ignore[arg-type]
    )
    table = Table(title="Hugging Face model candidates")
    for column in ("Model", "Downloads", "Likes", "Last Modified", "Pipeline", "License"):
        table.add_column(column)
    for model in models:
        table.add_row(
            str(model.get("modelId") or "-"),
            str(model.get("downloads") or 0),
            str(model.get("likes") or 0),
            str(model.get("lastModified") or "-"),
            str(model.get("pipeline_tag") or "-"),
            str(model.get("license") or "-"),
        )
    console.print(table)
    console.print("Advisory only: no models were downloaded and trust_remote_code was not used.")


@lab_app.command("engine-plan")
def lab_engine_plan(lab_yaml: str = typer.Argument(..., help="Path to lab.yaml.")) -> None:
    """Print manual engine launch guidance."""
    cfg = load_lab_config(lab_yaml)
    console.print(build_engine_plan(cfg))


def _print_lab_report(card) -> None:
    if card.baselines:
        table = Table(title="Single-model baselines")
        for column in (
            "Provider",
            "Model",
            "Accuracy",
            "Correct/Total",
            "Avg Latency",
            "Total Calls",
            "Total Tokens",
            "Accuracy/Call",
            "Accuracy/1k Tokens",
        ):
            table.add_column(column)
        for baseline in card.baselines:
            metrics = baseline.metrics
            table.add_row(
                baseline.provider,
                baseline.model,
                _format_percent(metrics.accuracy),
                f"{metrics.correct}/{metrics.total_examples}",
                _format_ms(metrics.avg_latency_ms),
                str(metrics.total_calls),
                str(metrics.total_tokens),
                _format_float(metrics.accuracy_per_call, digits=4),
                _format_float(metrics.accuracy_per_1k_tokens),
            )
        console.print(table)

    if card.strategy_comparisons:
        summaries = {summary.strategy: summary for summary in card.strategies}
        table = Table(title="Strategy comparison")
        for column in (
            "Strategy",
            "Accuracy",
            "Delta vs Fallback",
            "Delta vs Best Single",
            "Avg Latency",
            "Latency Ratio vs Best Single",
            "Avg Calls",
            "Call Ratio vs Best Single",
            "Total Tokens",
            "Recommendation Hint",
        ):
            table.add_column(column)
        for comparison in card.strategy_comparisons:
            summary = summaries.get(comparison.strategy)
            if summary is None:
                continue
            metrics = summary.metrics
            table.add_row(
                comparison.strategy,
                _format_percent(metrics.accuracy),
                _format_pp(comparison.accuracy_delta_vs_fallback_pp),
                _format_pp(comparison.accuracy_delta_vs_best_single_pp),
                _format_ms(metrics.avg_latency_ms),
                _format_ratio(comparison.latency_ratio_vs_best_single),
                _format_float(metrics.avg_calls_per_example),
                _format_ratio(comparison.calls_ratio_vs_best_single),
                str(metrics.total_tokens),
                comparison.recommendation_hint or "-",
            )
        console.print(table)

    _print_recommendation(card.recommendations)


def _print_recommendation(recommendation) -> None:
    console.print("[bold]Recommendations[/bold]")
    console.print(
        f"configured_{recommendation.configured_objective}: "
        f"{recommendation.recommended_strategy}"
    )
    by_objective = recommendation.by_objective or {
        "best_accuracy": recommendation.best_accuracy,
        "best_latency": recommendation.best_latency,
        "best_efficiency": recommendation.best_efficiency,
        "best_balanced": recommendation.best_balanced,
    }
    for objective in ("best_accuracy", "best_latency", "best_efficiency", "best_balanced"):
        console.print(f"{objective}: {by_objective.get(objective)}")
    if recommendation.explanations_by_objective:
        console.print("\n[bold]Objective explanations[/bold]")
        for objective in ("best_accuracy", "best_latency", "best_efficiency", "best_balanced"):
            explanations = recommendation.explanations_by_objective.get(objective) or []
            for explanation in explanations:
                console.print(f"- {objective}: {explanation}")
    if recommendation.explanations:
        console.print("\n[bold]Explanations[/bold]")
        for explanation in recommendation.explanations:
            console.print(f"- {explanation}")
    if recommendation.warnings:
        console.print("\n[bold yellow]Warnings[/bold yellow]")
        for warning in recommendation.warnings:
            console.print(f"- {warning}")


def _format_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _format_pp(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.1f} pp"


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}x"


def _format_ms(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{value:.0f} ms"


def _format_float(value: float | int | None, *, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


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
