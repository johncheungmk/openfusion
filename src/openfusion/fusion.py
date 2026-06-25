from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .config import AppConfig
from .providers import ModelProvider, ProviderClientPool, make_provider
from .schema import (
    CandidateResult,
    ChatMessage,
    FusionResult,
    OrchestrationPlan,
    ProviderRequest,
    Usage,
    WorkflowStep,
)


PARALLEL_SYNTHESIS_SYSTEM_PROMPT = """You are OpenFusion's synthesis agent.
Use the supplied independent candidate answers as evidence, not as authorities.
Identify consensus, contradictions, missing points, and likely errors, then write a new,
self-contained final answer. Do not merely select or concatenate a candidate. Do not reveal
hidden chain-of-thought or private reasoning. Return only the useful user-facing answer.
"""

SELECTOR_SYSTEM_PROMPT = """You are OpenFusion's best-of-N evaluator.
Choose the candidate that most accurately and completely answers the user's request.
Do not rewrite the answer and do not expose hidden chain-of-thought. Return strict JSON only:
{"winner": 1, "reason": "one brief user-visible reason"}
"""

CRITIC_SYSTEM_PROMPT = """You are OpenFusion's critic.
Inspect independent candidate answers for factual errors, unsupported claims, contradictions,
omissions, and instruction-following problems. Produce concise, actionable, user-visible
feedback for a reviser. Do not reveal hidden chain-of-thought.
"""

REVISION_SYSTEM_PROMPT = """You are OpenFusion's revision agent.
Write a new final answer using the original request, independent drafts, and the critic's
feedback. Correct errors, preserve useful complementary details, and follow the user's format.
Do not mention the workflow or reveal hidden chain-of-thought. Return only the final answer.
"""

REFINEMENT_SYSTEM_PROMPT = """You are one agent in an OpenFusion refinement layer.
Review the previous layer's candidate answers, independently check their weaknesses, and produce
one improved answer. Do not simply vote or concatenate. Do not mention candidate labels and do
not reveal hidden chain-of-thought. Return only the improved answer.
"""

PLANNER_SYSTEM_PROMPT = """You are OpenFusion's constrained workflow planner.
Select a safe, cost-bounded workflow from the allowed strategies and providers. You may not
invent providers, tools, or strategies. Return strict JSON only and give one brief operational
rationale, not hidden chain-of-thought.
"""

SUPPORTED_STRATEGIES = (
    "fallback",
    "parallel_synthesis",
    "best_of_n",
    "majority_vote",
    "weighted_vote",
    "critique_revision",
    "layered_refinement",
    "adaptive",
)

STRATEGY_ALIASES = {
    "fastest": "fallback",
    "panel_judge": "parallel_synthesis",
    "panel-judge": "parallel_synthesis",
    "parallel_judge": "parallel_synthesis",
    "parallel-judge": "parallel_synthesis",
    "parallel-synthesis": "parallel_synthesis",
    "fusion": "parallel_synthesis",
    "best-of-n": "best_of_n",
    "majority-vote": "majority_vote",
    "weighted-vote": "weighted_vote",
    "critique-revision": "critique_revision",
    "layered-refinement": "layered_refinement",
}


@dataclass
class CallBudget:
    limit: int
    used: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def reserve(self) -> bool:
        if self.used >= self.limit:
            return False
        self.used += 1
        return True


def canonical_strategy(strategy: str) -> str:
    normalized = strategy.strip().lower().replace(" ", "_")
    normalized = STRATEGY_ALIASES.get(normalized, normalized.replace("-", "_"))
    if normalized not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Unknown fusion strategy: {strategy}. Supported: {', '.join(SUPPORTED_STRATEGIES)}"
        )
    return normalized


def _render_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        rendered_parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    rendered_parts.append(part["text"])
                elif part.get("type") == "image_url":
                    rendered_parts.append("[image]")
                else:
                    rendered_parts.append(str(part))
            else:
                rendered_parts.append(str(part))
        return "\n".join(rendered_parts)
    return str(content)


def _latest_user_message(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return _render_message_content(message.content)
    return ""


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else stripped
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(candidate[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


class FusionEngine:
    def __init__(self, config: AppConfig, providers: dict[str, ModelProvider] | None = None):
        self.config = config
        self._client_pool = ProviderClientPool() if providers is None else None
        self.providers = providers or {
            provider_config.name: make_provider(provider_config, client_pool=self._client_pool)
            for provider_config in config.providers
            if provider_config.enabled
        }

    @staticmethod
    def supported_strategies() -> tuple[str, ...]:
        return SUPPORTED_STRATEGIES

    async def aclose(self) -> None:
        for provider in self.providers.values():
            await provider.aclose()
        if self._client_pool is not None:
            await self._client_pool.aclose()

    async def run(
        self,
        messages: list[ChatMessage],
        strategy: str | None = None,
        panel: Iterable[str] | None = None,
        judge_provider: str | None = None,
        critic_provider: str | None = None,
        reviser_provider: str | None = None,
        planner_provider: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
        samples_per_provider: int | None = None,
        refinement_rounds: int | None = None,
        max_total_calls: int | None = None,
        vote_regex: str | None = None,
    ) -> FusionResult:
        selected_strategy = canonical_strategy(strategy or self.config.fusion.default_strategy)
        panel_names = self._panel_names(panel)
        samples = max(1, samples_per_provider or self.config.fusion.samples_per_provider)
        rounds = (
            self.config.fusion.refinement_rounds
            if refinement_rounds is None
            else max(0, refinement_rounds)
        )
        call_limit = max(1, max_total_calls or self.config.fusion.max_total_calls)
        budget = CallBudget(call_limit)
        trace: list[WorkflowStep] = []

        planning_usage = Usage()
        if selected_strategy == "adaptive":
            plan, planning_usage = await self._adaptive_plan(
                messages=messages,
                panel=panel_names,
                judge_provider=judge_provider,
                critic_provider=critic_provider,
                reviser_provider=reviser_provider,
                planner_provider=planner_provider,
                samples_per_provider=samples,
                refinement_rounds=rounds,
                budget=budget,
                trace=trace,
            )
        else:
            plan = self._request_plan(
                strategy=selected_strategy,
                panel=panel_names,
                judge_provider=judge_provider,
                critic_provider=critic_provider,
                reviser_provider=reviser_provider,
                samples_per_provider=samples,
                refinement_rounds=rounds,
                max_total_calls=call_limit,
            )

        result = await self._execute(
            strategy=plan.strategy,
            messages=messages,
            panel=plan.panel,
            judge_provider=plan.judge_provider,
            critic_provider=plan.critic_provider,
            reviser_provider=plan.reviser_provider,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
            samples_per_provider=plan.samples_per_provider,
            refinement_rounds=plan.refinement_rounds,
            vote_regex=vote_regex or self.config.fusion.vote_answer_regex,
            budget=budget,
            trace=trace,
        )
        result.usage = result.usage + planning_usage
        result.plan = plan
        result.trace = list(trace)
        return result

    async def plan(
        self,
        messages: list[ChatMessage],
        panel: Iterable[str] | None = None,
        planner_provider: str | None = None,
        max_total_calls: int | None = None,
        use_model_planner: bool | None = None,
    ) -> tuple[OrchestrationPlan, list[WorkflowStep]]:
        panel_names = self._panel_names(panel)
        call_limit = max(1, max_total_calls or self.config.fusion.max_total_calls)
        budget = CallBudget(call_limit)
        trace: list[WorkflowStep] = []
        plan, _planning_usage = await self._adaptive_plan(
            messages=messages,
            panel=panel_names,
            judge_provider=None,
            critic_provider=None,
            reviser_provider=None,
            planner_provider=planner_provider,
            samples_per_provider=self.config.fusion.samples_per_provider,
            refinement_rounds=self.config.fusion.refinement_rounds,
            budget=budget,
            trace=trace,
            force_model_planner=use_model_planner,
        )
        return plan, trace

    async def run_provider(
        self,
        provider_name: str,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> FusionResult:
        trace: list[WorkflowStep] = []
        budget = CallBudget(1)
        request = self._provider_request(messages, temperature, max_tokens, extra_body)
        result = await self._call_provider(
            provider_name,
            request,
            asyncio.Semaphore(1),
            budget,
            trace,
            stage="direct",
            sample_index=1,
        )
        plan = OrchestrationPlan(
            strategy="direct_provider",
            panel=[provider_name],
            max_total_calls=1,
            estimated_calls=1,
            source="request",
            rationale="Direct provider route requested by model ID.",
        )
        return FusionResult(
            strategy="direct_provider",
            final=result.content if result.ok and result.content.strip() else "",
            candidates=self._visible_candidates([result]),
            usage=result.usage,
            plan=plan,
            trace=trace,
        )

    async def _execute(
        self,
        strategy: str,
        messages: list[ChatMessage],
        panel: list[str],
        judge_provider: str | None,
        critic_provider: str | None,
        reviser_provider: str | None,
        temperature: float | None,
        max_tokens: int | None,
        extra_body: dict[str, Any] | None,
        samples_per_provider: int,
        refinement_rounds: int,
        vote_regex: str | None,
        budget: CallBudget,
        trace: list[WorkflowStep],
    ) -> FusionResult:
        if strategy == "fallback":
            return await self._fallback(
                messages,
                panel,
                temperature,
                max_tokens,
                extra_body,
                budget,
                trace,
            )
        if strategy == "parallel_synthesis":
            return await self._parallel_synthesis(
                messages,
                panel,
                judge_provider,
                temperature,
                max_tokens,
                extra_body,
                samples_per_provider,
                budget,
                trace,
            )
        if strategy == "best_of_n":
            return await self._best_of_n(
                messages,
                panel,
                judge_provider,
                temperature,
                max_tokens,
                extra_body,
                samples_per_provider,
                budget,
                trace,
            )
        if strategy in {"majority_vote", "weighted_vote"}:
            return await self._vote(
                strategy,
                messages,
                panel,
                temperature,
                max_tokens,
                extra_body,
                samples_per_provider,
                vote_regex,
                budget,
                trace,
            )
        if strategy == "critique_revision":
            return await self._critique_revision(
                messages,
                panel,
                critic_provider,
                reviser_provider,
                judge_provider,
                temperature,
                max_tokens,
                extra_body,
                samples_per_provider,
                budget,
                trace,
            )
        if strategy == "layered_refinement":
            return await self._layered_refinement(
                messages,
                panel,
                judge_provider,
                temperature,
                max_tokens,
                extra_body,
                samples_per_provider,
                refinement_rounds,
                budget,
                trace,
            )
        raise ValueError(f"Unsupported executable strategy: {strategy}")

    async def _call_provider(
        self,
        provider_name: str,
        request: ProviderRequest,
        semaphore: asyncio.Semaphore,
        budget: CallBudget,
        trace: list[WorkflowStep],
        stage: str,
        sample_index: int,
    ) -> CandidateResult:
        provider = self.providers.get(provider_name)
        if provider is None:
            result = CandidateResult(
                provider=provider_name,
                model="unknown",
                ok=False,
                error=f"Provider not found or not enabled: {provider_name}",
                stage=stage,
                sample_index=sample_index,
            )
            trace.append(
                WorkflowStep(
                    stage=stage,
                    provider=provider_name,
                    model="unknown",
                    status="error",
                    note=result.error,
                )
            )
            return result
        if not budget.reserve():
            result = CandidateResult(
                provider=provider_name,
                model=provider.config.model,
                weight=provider.config.weight,
                ok=False,
                error=f"Call budget exhausted ({budget.limit} calls)",
                stage=stage,
                sample_index=sample_index,
            )
            trace.append(
                WorkflowStep(
                    stage=stage,
                    provider=provider_name,
                    model=provider.config.model,
                    status="skipped",
                    note=result.error,
                )
            )
            return result
        async with semaphore:
            result = await provider.chat(request)
        result.weight = provider.config.weight
        result.model = provider.config.model
        result.stage = stage
        result.sample_index = sample_index
        trace.append(
            WorkflowStep(
                stage=stage,
                provider=provider_name,
                model=provider.config.model,
                status="ok" if result.ok and result.content.strip() else "error",
                latency_ms=result.latency_ms,
                note=None if result.ok else (result.error or "empty response")[:300],
            )
        )
        return result

    async def _generate_candidates(
        self,
        provider_names: list[str],
        request: ProviderRequest,
        samples_per_provider: int,
        budget: CallBudget,
        trace: list[WorkflowStep],
        stage: str,
    ) -> list[CandidateResult]:
        semaphore = asyncio.Semaphore(max(1, self.config.fusion.max_parallel))
        tasks = []
        total_requested = len(provider_names) * samples_per_provider
        for provider_name in provider_names:
            for sample_index in range(1, samples_per_provider + 1):
                sample_request = request
                if total_requested > 1:
                    sample_request = request.model_copy(
                        update={
                            "messages": self._sampling_messages(request.messages, sample_index),
                        },
                        deep=True,
                    )
                tasks.append(
                    self._call_provider(
                        provider_name,
                        sample_request,
                        semaphore,
                        budget,
                        trace,
                        stage=stage,
                        sample_index=sample_index,
                    )
                )
        if not tasks:
            return []
        return list(await asyncio.gather(*tasks))

    async def _parallel_synthesis(
        self,
        messages: list[ChatMessage],
        panel: list[str],
        judge_provider: str | None,
        temperature: float | None,
        max_tokens: int | None,
        extra_body: dict[str, Any] | None,
        samples_per_provider: int,
        budget: CallBudget,
        trace: list[WorkflowStep],
    ) -> FusionResult:
        request = self._provider_request(messages, temperature, max_tokens, extra_body)
        candidates = await self._generate_candidates(
            panel, request, samples_per_provider, budget, trace, stage="draft"
        )
        successes = self._successes(candidates)
        if len(successes) < self.config.fusion.require_at_least_successes:
            return self._no_success_result("parallel_synthesis", candidates)

        selected_judge = self._role_provider(
            judge_provider,
            self.config.fusion.judge_provider,
            successes,
            panel,
        )
        judge_result = await self._call_synthesizer(
            messages,
            successes,
            selected_judge,
            max_tokens,
            budget,
            trace,
            stage="synthesis",
        )
        usage = self._sum_usage(candidates) + judge_result.usage
        if judge_result.ok and judge_result.content.strip():
            return FusionResult(
                strategy="parallel_synthesis",
                final=judge_result.content,
                judge_provider=selected_judge,
                judge_analysis=f"Synthesized {len(successes)} independent candidate(s).",
                candidates=self._visible_candidates(candidates),
                usage=usage,
            )

        best = self._deterministic_best(successes)
        return FusionResult(
            strategy="parallel_synthesis",
            final=best.content,
            judge_provider=selected_judge,
            judge_analysis=f"Synthesis failed: {judge_result.error or 'empty response'}",
            candidates=self._visible_candidates(candidates),
            usage=usage,
        )

    async def _best_of_n(
        self,
        messages: list[ChatMessage],
        panel: list[str],
        judge_provider: str | None,
        temperature: float | None,
        max_tokens: int | None,
        extra_body: dict[str, Any] | None,
        samples_per_provider: int,
        budget: CallBudget,
        trace: list[WorkflowStep],
    ) -> FusionResult:
        effective_samples = samples_per_provider
        if len(panel) * effective_samples < 2:
            effective_samples = 2
        request = self._provider_request(messages, temperature, max_tokens, extra_body)
        candidates = await self._generate_candidates(
            panel, request, effective_samples, budget, trace, stage="candidate"
        )
        successes = self._successes(candidates)
        if not successes:
            return self._no_success_result("best_of_n", candidates)
        if len(successes) == 1:
            return FusionResult(
                strategy="best_of_n",
                final=successes[0].content,
                judge_analysis="Only one usable candidate was available.",
                candidates=self._visible_candidates(candidates),
                usage=self._sum_usage(candidates),
            )

        selected_judge = self._role_provider(
            judge_provider,
            self.config.fusion.judge_provider,
            successes,
            panel,
        )
        selector_request = ProviderRequest(
            messages=[
                ChatMessage(role="system", content=SELECTOR_SYSTEM_PROMPT),
                ChatMessage(role="user", content=self._build_selector_prompt(messages, successes)),
            ],
            temperature=0.0,
            max_tokens=220,
        )
        selector = await self._call_provider(
            selected_judge,
            selector_request,
            asyncio.Semaphore(1),
            budget,
            trace,
            stage="selection",
            sample_index=1,
        )
        usage = self._sum_usage(candidates) + selector.usage
        selection = self._parse_selection(selector.content, len(successes)) if selector.ok else None
        if selection is not None:
            winner_index, reason = selection
            winner = successes[winner_index]
            return FusionResult(
                strategy="best_of_n",
                final=winner.content,
                judge_provider=selected_judge,
                judge_analysis=reason,
                candidates=self._visible_candidates(candidates),
                usage=usage,
            )

        winner = self._deterministic_best(successes)
        detail = selector.error or "selector returned invalid JSON"
        return FusionResult(
            strategy="best_of_n",
            final=winner.content,
            judge_provider=selected_judge,
            judge_analysis=f"Deterministic fallback used because {detail}.",
            candidates=self._visible_candidates(candidates),
            usage=usage,
        )

    async def _vote(
        self,
        strategy: str,
        messages: list[ChatMessage],
        panel: list[str],
        temperature: float | None,
        max_tokens: int | None,
        extra_body: dict[str, Any] | None,
        samples_per_provider: int,
        vote_regex: str | None,
        budget: CallBudget,
        trace: list[WorkflowStep],
    ) -> FusionResult:
        minimum = 3 if strategy == "majority_vote" else 2
        effective_samples = samples_per_provider
        if len(panel) * effective_samples < minimum:
            effective_samples = max(1, (minimum + max(1, len(panel)) - 1) // max(1, len(panel)))
        request = self._provider_request(messages, temperature, max_tokens, extra_body)
        candidates = await self._generate_candidates(
            panel, request, effective_samples, budget, trace, stage="vote_candidate"
        )
        successes = self._successes(candidates)
        if not successes:
            return self._no_success_result(strategy, candidates)

        groups: dict[str, list[CandidateResult]] = defaultdict(list)
        for candidate in successes:
            groups[self._vote_key(candidate.content, vote_regex)].append(candidate)

        def score(item: tuple[str, list[CandidateResult]]) -> tuple[float, int, float, int]:
            _, group = item
            weighted = sum(candidate.weight for candidate in group)
            primary = float(len(group)) if strategy == "majority_vote" else weighted
            return primary, len(group), weighted, max(len(candidate.content) for candidate in group)

        winning_key, winning_group = max(groups.items(), key=score)
        representative = self._deterministic_best(winning_group)
        weighted_score = sum(candidate.weight for candidate in winning_group)
        summary = {
            "winning_key": winning_key[:300],
            "votes": len(winning_group),
            "weighted_score": weighted_score,
            "usable_candidates": len(successes),
            "groups": len(groups),
        }
        outputs = (
            {"vote_summary": json.dumps(summary, ensure_ascii=False)}
            if self.config.fusion.include_workflow_outputs
            else {}
        )
        return FusionResult(
            strategy=strategy,
            final=representative.content,
            judge_analysis=(
                f"Winning consensus group: {len(winning_group)}/{len(successes)} usable "
                f"candidate(s), weighted score {weighted_score:g}."
            ),
            candidates=self._visible_candidates(candidates),
            usage=self._sum_usage(candidates),
            workflow_outputs=outputs,
        )

    async def _critique_revision(
        self,
        messages: list[ChatMessage],
        panel: list[str],
        critic_provider: str | None,
        reviser_provider: str | None,
        judge_provider: str | None,
        temperature: float | None,
        max_tokens: int | None,
        extra_body: dict[str, Any] | None,
        samples_per_provider: int,
        budget: CallBudget,
        trace: list[WorkflowStep],
    ) -> FusionResult:
        request = self._provider_request(messages, temperature, max_tokens, extra_body)
        candidates = await self._generate_candidates(
            panel, request, samples_per_provider, budget, trace, stage="draft"
        )
        successes = self._successes(candidates)
        if len(successes) < self.config.fusion.require_at_least_successes:
            return self._no_success_result("critique_revision", candidates)

        selected_critic = self._role_provider(
            critic_provider,
            self.config.fusion.critic_provider or self.config.fusion.judge_provider,
            successes,
            panel,
        )
        critic_request = ProviderRequest(
            messages=[
                ChatMessage(role="system", content=CRITIC_SYSTEM_PROMPT),
                ChatMessage(role="user", content=self._build_critic_prompt(messages, successes)),
            ],
            temperature=self.config.fusion.critique_temperature,
            max_tokens=max_tokens if max_tokens is not None else self.config.fusion.max_tokens,
        )
        critique = await self._call_provider(
            selected_critic,
            critic_request,
            asyncio.Semaphore(1),
            budget,
            trace,
            stage="critique",
            sample_index=1,
        )

        selected_reviser = self._role_provider(
            reviser_provider,
            self.config.fusion.reviser_provider
            or judge_provider
            or self.config.fusion.judge_provider,
            successes,
            panel,
        )
        critique_text = (
            critique.content
            if critique.ok and critique.content.strip()
            else f"Critic unavailable: {critique.error or 'empty response'}. Independently verify drafts."
        )
        revision_request = ProviderRequest(
            messages=[
                ChatMessage(role="system", content=REVISION_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=self._build_revision_prompt(messages, successes, critique_text),
                ),
            ],
            temperature=self.config.fusion.judge_temperature,
            max_tokens=max_tokens if max_tokens is not None else self.config.fusion.max_tokens,
        )
        revision = await self._call_provider(
            selected_reviser,
            revision_request,
            asyncio.Semaphore(1),
            budget,
            trace,
            stage="revision",
            sample_index=1,
        )
        usage = self._sum_usage(candidates) + critique.usage + revision.usage
        if revision.ok and revision.content.strip():
            final = revision.content
        else:
            final = self._deterministic_best(successes).content

        outputs: dict[str, str] = {}
        if self.config.fusion.include_workflow_outputs:
            outputs["critique"] = critique_text
            if not revision.ok:
                outputs["revision_error"] = revision.error or "empty response"
        return FusionResult(
            strategy="critique_revision",
            final=final,
            critic_provider=selected_critic,
            reviser_provider=selected_reviser,
            candidates=self._visible_candidates(candidates),
            usage=usage,
            workflow_outputs=outputs,
        )

    async def _layered_refinement(
        self,
        messages: list[ChatMessage],
        panel: list[str],
        judge_provider: str | None,
        temperature: float | None,
        max_tokens: int | None,
        extra_body: dict[str, Any] | None,
        samples_per_provider: int,
        refinement_rounds: int,
        budget: CallBudget,
        trace: list[WorkflowStep],
    ) -> FusionResult:
        base_request = self._provider_request(messages, temperature, max_tokens, extra_body)
        all_candidates = await self._generate_candidates(
            panel, base_request, samples_per_provider, budget, trace, stage="layer_0"
        )
        current = self._successes(all_candidates)
        if not current:
            return self._no_success_result("layered_refinement", all_candidates)

        for round_index in range(1, refinement_rounds + 1):
            refinement_request = ProviderRequest(
                messages=[
                    ChatMessage(role="system", content=REFINEMENT_SYSTEM_PROMPT),
                    ChatMessage(
                        role="user",
                        content=self._build_refinement_prompt(messages, current, round_index),
                    ),
                ],
                temperature=temperature
                if temperature is not None
                else self.config.fusion.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.config.fusion.max_tokens,
            )
            layer = await self._generate_candidates(
                panel,
                refinement_request,
                1,
                budget,
                trace,
                stage=f"layer_{round_index}",
            )
            all_candidates.extend(layer)
            layer_successes = self._successes(layer)
            if layer_successes:
                current = layer_successes
            if budget.remaining <= 0:
                break

        selected_judge = self._role_provider(
            judge_provider,
            self.config.fusion.judge_provider,
            current,
            panel,
        )
        synthesis = await self._call_synthesizer(
            messages,
            current,
            selected_judge,
            max_tokens,
            budget,
            trace,
            stage="final_synthesis",
        )
        usage = self._sum_usage(all_candidates) + synthesis.usage
        if synthesis.ok and synthesis.content.strip():
            final = synthesis.content
            note = f"Synthesized the final refinement layer ({len(current)} candidate(s))."
        else:
            final = self._deterministic_best(current).content
            note = f"Final synthesis failed: {synthesis.error or 'empty response'}"
        return FusionResult(
            strategy="layered_refinement",
            final=final,
            judge_provider=selected_judge,
            judge_analysis=note,
            candidates=self._visible_candidates(all_candidates),
            usage=usage,
        )

    async def _fallback(
        self,
        messages: list[ChatMessage],
        panel: list[str],
        temperature: float | None,
        max_tokens: int | None,
        extra_body: dict[str, Any] | None,
        budget: CallBudget,
        trace: list[WorkflowStep],
    ) -> FusionResult:
        request = self._provider_request(messages, temperature, max_tokens, extra_body)
        candidates: list[CandidateResult] = []
        for provider_name in panel:
            result = await self._call_provider(
                provider_name,
                request,
                asyncio.Semaphore(1),
                budget,
                trace,
                stage="fallback",
                sample_index=1,
            )
            candidates.append(result)
            if result.ok and result.content.strip():
                return FusionResult(
                    strategy="fallback",
                    final=result.content,
                    candidates=self._visible_candidates(candidates),
                    usage=self._sum_usage(candidates),
                )
        return FusionResult(
            strategy="fallback",
            final="No provider produced a usable answer.",
            candidates=self._visible_candidates(candidates),
            usage=self._sum_usage(candidates),
        )

    async def _call_synthesizer(
        self,
        messages: list[ChatMessage],
        candidates: list[CandidateResult],
        provider_name: str,
        max_tokens: int | None,
        budget: CallBudget,
        trace: list[WorkflowStep],
        stage: str,
    ) -> CandidateResult:
        request = ProviderRequest(
            messages=[
                ChatMessage(role="system", content=PARALLEL_SYNTHESIS_SYSTEM_PROMPT),
                ChatMessage(role="user", content=self._build_synthesis_prompt(messages, candidates)),
            ],
            temperature=self.config.fusion.judge_temperature,
            max_tokens=max_tokens if max_tokens is not None else self.config.fusion.max_tokens,
        )
        return await self._call_provider(
            provider_name,
            request,
            asyncio.Semaphore(1),
            budget,
            trace,
            stage=stage,
            sample_index=1,
        )

    async def _adaptive_plan(
        self,
        messages: list[ChatMessage],
        panel: list[str],
        judge_provider: str | None,
        critic_provider: str | None,
        reviser_provider: str | None,
        planner_provider: str | None,
        samples_per_provider: int,
        refinement_rounds: int,
        budget: CallBudget,
        trace: list[WorkflowStep],
        force_model_planner: bool | None = None,
    ) -> tuple[OrchestrationPlan, Usage]:
        heuristic = self._heuristic_plan(
            messages,
            panel,
            judge_provider,
            critic_provider,
            reviser_provider,
            samples_per_provider,
            refinement_rounds,
            budget,
        )
        use_model = (
            self.config.fusion.adaptive_use_model_planner
            if force_model_planner is None
            else force_model_planner
        )
        if planner_provider is not None:
            use_model = True
        if not use_model or budget.remaining <= 1:
            return heuristic, Usage()

        selected_planner = (
            planner_provider
            or self.config.fusion.planner_provider
            or self.config.fusion.judge_provider
            or (panel[0] if panel else None)
        )
        if not selected_planner or selected_planner not in self.providers:
            return heuristic, Usage()

        planner_request = ProviderRequest(
            messages=[
                ChatMessage(role="system", content=PLANNER_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=self._build_planner_prompt(messages, panel, budget.remaining - 1),
                ),
            ],
            temperature=0.0,
            max_tokens=400,
        )
        planner_result = await self._call_provider(
            selected_planner,
            planner_request,
            asyncio.Semaphore(1),
            budget,
            trace,
            stage="planning",
            sample_index=1,
        )
        if not planner_result.ok:
            heuristic.rationale = (
                f"Heuristic plan used because model planner failed: "
                f"{planner_result.error or 'empty response'}."
            )
            return self._fit_plan_to_budget(heuristic, budget.remaining), planner_result.usage

        parsed = self._parse_model_plan(planner_result.content, panel, budget)
        if parsed is None:
            trace.append(
                WorkflowStep(
                    stage="planning_validation",
                    provider=selected_planner,
                    model=self.providers[selected_planner].config.model,
                    status="fallback",
                    note="Planner output was not valid constrained JSON; heuristic plan used.",
                )
            )
            heuristic.rationale = "Heuristic plan used because model planner output was invalid."
            return self._fit_plan_to_budget(heuristic, budget.remaining), planner_result.usage
        return parsed, planner_result.usage

    def _heuristic_plan(
        self,
        messages: list[ChatMessage],
        panel: list[str],
        judge_provider: str | None,
        critic_provider: str | None,
        reviser_provider: str | None,
        samples_per_provider: int,
        refinement_rounds: int,
        budget: CallBudget,
    ) -> OrchestrationPlan:
        prompt = _latest_user_message(messages).casefold()
        multiple_choice = bool(
            re.search(r"\bmultiple[- ]choice\b|\bchoose (?:one|the best)\b|(?:^|\n)\s*[a-d][.)]", prompt)
        )
        code_or_math = bool(
            re.search(
                r"\b(debug|implement|code|program|algorithm|calculate|equation|proof|solve|unit test)\b",
                prompt,
            )
        )
        explicit_refinement = bool(
            re.search(r"\b(debate|critique|review alternatives|refine|challenge the answers)\b", prompt)
        )
        complex_analysis = bool(
            re.search(
                r"\b(compare|research|analy[sz]e|architecture|risk|policy|deployment|plan|recommend|evidence)\b",
                prompt,
            )
        )

        samples = max(1, samples_per_provider)
        rounds = max(0, refinement_rounds)
        if explicit_refinement:
            strategy = "layered_refinement"
            rounds = max(1, rounds)
            rationale = "The request explicitly benefits from critique and iterative refinement."
        elif multiple_choice:
            strategy = "weighted_vote"
            if len(panel) * samples < 3:
                samples = max(1, (3 + max(1, len(panel)) - 1) // max(1, len(panel)))
            rationale = "A concise or multiple-choice task is suitable for consensus voting."
        elif code_or_math:
            strategy = "best_of_n"
            if len(panel) * samples < 2:
                samples = 2
            rationale = "Independent attempts plus selection are useful for code or verifiable reasoning."
        elif complex_analysis and len(prompt) >= 80:
            strategy = "critique_revision"
            rationale = "The task benefits from independent drafts followed by critique and revision."
        elif len(prompt) < 180:
            strategy = "fallback"
            rationale = "The request appears simple, so a single successful provider minimizes latency."
        else:
            strategy = "parallel_synthesis"
            rationale = "The request benefits from complementary independent answers and synthesis."

        plan = OrchestrationPlan(
            strategy=strategy,
            panel=panel,
            judge_provider=judge_provider or self.config.fusion.judge_provider,
            critic_provider=critic_provider or self.config.fusion.critic_provider,
            reviser_provider=reviser_provider or self.config.fusion.reviser_provider,
            samples_per_provider=samples,
            refinement_rounds=rounds,
            max_total_calls=budget.limit,
            estimated_calls=self._estimate_calls(strategy, panel, samples, rounds),
            source="heuristic",
            rationale=rationale,
        )
        return self._fit_plan_to_budget(plan, budget.remaining)

    def _parse_model_plan(
        self,
        text: str,
        default_panel: list[str],
        budget: CallBudget,
    ) -> OrchestrationPlan | None:
        data = _extract_json_object(text)
        if data is None:
            return None
        try:
            strategy = canonical_strategy(str(data.get("strategy", "fallback")))
        except ValueError:
            return None
        if strategy == "adaptive":
            return None
        requested_panel = data.get("panel")
        if not isinstance(requested_panel, list):
            requested_panel = default_panel
        panel = [name for name in requested_panel if isinstance(name, str) and name in self.providers]
        if not panel:
            panel = default_panel
        samples = self._bounded_int(data.get("samples_per_provider"), 1, 3, 1)
        rounds = self._bounded_int(data.get("refinement_rounds"), 0, 2, 0)
        rationale = str(data.get("rationale") or "Model planner selected this workflow.")[:500]
        plan = OrchestrationPlan(
            strategy=strategy,
            panel=panel,
            judge_provider=self._valid_provider_name(data.get("judge_provider")),
            critic_provider=self._valid_provider_name(data.get("critic_provider")),
            reviser_provider=self._valid_provider_name(data.get("reviser_provider")),
            samples_per_provider=samples,
            refinement_rounds=rounds,
            max_total_calls=budget.limit,
            estimated_calls=self._estimate_calls(strategy, panel, samples, rounds),
            source="model",
            rationale=rationale,
        )
        fitted = self._fit_plan_to_budget(plan, budget.remaining)
        return fitted.model_copy(update={"estimated_calls": fitted.estimated_calls + 1})

    def _fit_plan_to_budget(
        self,
        plan: OrchestrationPlan,
        remaining_calls: int,
    ) -> OrchestrationPlan:
        remaining_calls = max(1, remaining_calls)
        samples = plan.samples_per_provider
        rounds = plan.refinement_rounds
        estimated = self._estimate_calls(plan.strategy, plan.panel, samples, rounds)
        while estimated > remaining_calls and samples > 1:
            samples -= 1
            estimated = self._estimate_calls(plan.strategy, plan.panel, samples, rounds)
        while estimated > remaining_calls and rounds > 0:
            rounds -= 1
            estimated = self._estimate_calls(plan.strategy, plan.panel, samples, rounds)
        strategy = plan.strategy
        rationale = plan.rationale
        if estimated > remaining_calls:
            strategy = "fallback"
            samples = 1
            rounds = 0
            estimated = min(max(1, len(plan.panel)), remaining_calls)
            rationale = f"{rationale} Reduced to fallback to respect the call budget."
        return plan.model_copy(
            update={
                "strategy": strategy,
                "samples_per_provider": samples,
                "refinement_rounds": rounds,
                "estimated_calls": estimated,
                "rationale": rationale,
            }
        )

    def _request_plan(
        self,
        strategy: str,
        panel: list[str],
        judge_provider: str | None,
        critic_provider: str | None,
        reviser_provider: str | None,
        samples_per_provider: int,
        refinement_rounds: int,
        max_total_calls: int,
    ) -> OrchestrationPlan:
        samples_per_provider = self._effective_samples_for_strategy(
            strategy, panel, samples_per_provider
        )
        return OrchestrationPlan(
            strategy=strategy,
            panel=panel,
            judge_provider=judge_provider or self.config.fusion.judge_provider,
            critic_provider=critic_provider or self.config.fusion.critic_provider,
            reviser_provider=reviser_provider or self.config.fusion.reviser_provider,
            samples_per_provider=samples_per_provider,
            refinement_rounds=refinement_rounds,
            max_total_calls=max_total_calls,
            estimated_calls=self._estimate_calls(
                strategy, panel, samples_per_provider, refinement_rounds
            ),
            source="request",
            rationale="Explicit user-selected workflow.",
        )

    @staticmethod
    def _estimate_calls(
        strategy: str,
        panel: list[str],
        samples_per_provider: int,
        refinement_rounds: int,
    ) -> int:
        panel_size = max(1, len(panel))
        samples_per_provider = FusionEngine._effective_samples_for_strategy(
            strategy, panel, samples_per_provider
        )
        drafts = panel_size * max(1, samples_per_provider)
        if strategy == "fallback":
            return panel_size
        if strategy in {"parallel_synthesis", "best_of_n"}:
            return drafts + 1
        if strategy in {"majority_vote", "weighted_vote"}:
            return drafts
        if strategy == "critique_revision":
            return drafts + 2
        if strategy == "layered_refinement":
            return drafts + panel_size * max(0, refinement_rounds) + 1
        return 1

    @staticmethod
    def _effective_samples_for_strategy(
        strategy: str,
        panel: list[str],
        samples_per_provider: int,
    ) -> int:
        panel_size = max(1, len(panel))
        minimum_total = 1
        if strategy in {"best_of_n", "weighted_vote"}:
            minimum_total = 2
        elif strategy == "majority_vote":
            minimum_total = 3
        minimum_per_provider = (minimum_total + panel_size - 1) // panel_size
        return max(1, samples_per_provider, minimum_per_provider)

    def _provider_request(
        self,
        messages: list[ChatMessage],
        temperature: float | None,
        max_tokens: int | None,
        extra_body: dict[str, Any] | None,
    ) -> ProviderRequest:
        return ProviderRequest(
            messages=messages,
            temperature=temperature if temperature is not None else self.config.fusion.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.config.fusion.max_tokens,
            extra_body=extra_body or {},
        )

    def _panel_names(self, panel: Iterable[str] | None) -> list[str]:
        selected = list(panel) if panel is not None else list(self.config.fusion.panel)
        if not selected:
            selected = list(self.providers.keys())
        # Stable de-duplication protects budgets from accidental repeated names.
        return list(dict.fromkeys(selected))

    @staticmethod
    def _successes(candidates: list[CandidateResult]) -> list[CandidateResult]:
        return [candidate for candidate in candidates if candidate.ok and candidate.content.strip()]

    def _role_provider(
        self,
        requested: str | None,
        configured: str | None,
        successes: list[CandidateResult],
        panel: list[str],
    ) -> str:
        for candidate in (requested, configured):
            if candidate and candidate in self.providers:
                return candidate
        if successes:
            return successes[0].provider
        if panel:
            return panel[0]
        raise ValueError("No enabled provider is available for the requested workflow role")

    def _sampling_messages(
        self,
        messages: list[ChatMessage],
        sample_index: int,
    ) -> list[ChatMessage]:
        instruction = ChatMessage(
            role="system",
            content=(
                "OpenFusion independent-sampling instruction: produce a fresh solution without "
                f"assuming another sample's approach. This is sample {sample_index}; do not mention "
                "sample numbers in the answer."
            ),
        )
        split = 0
        while split < len(messages) and messages[split].role in {"system", "developer"}:
            split += 1
        return [*messages[:split], instruction, *messages[split:]]

    def _build_synthesis_prompt(
        self,
        messages: list[ChatMessage],
        candidates: list[CandidateResult],
    ) -> str:
        parts = [
            "Conversation transcript:",
            self._conversation_transcript(messages),
            "",
            "Latest user request:",
            _latest_user_message(messages),
            "",
            "Independent candidate answers. Weights are advisory reliability hints:",
        ]
        parts.extend(self._render_candidates(candidates))
        parts.append(
            "Write a new final answer that resolves contradictions and combines complementary "
            "strengths. Do not expose candidate labels or hidden reasoning."
        )
        return "\n".join(parts)

    def _build_selector_prompt(
        self,
        messages: list[ChatMessage],
        candidates: list[CandidateResult],
    ) -> str:
        parts = [
            "Conversation transcript:",
            self._conversation_transcript(messages),
            "",
            "Choose the single best candidate:",
        ]
        parts.extend(self._render_candidates(candidates))
        return "\n".join(parts)

    def _build_critic_prompt(
        self,
        messages: list[ChatMessage],
        candidates: list[CandidateResult],
    ) -> str:
        parts = [
            "Conversation transcript:",
            self._conversation_transcript(messages),
            "",
            "Draft answers to audit:",
        ]
        parts.extend(self._render_candidates(candidates))
        parts.append("Return concise correction and coverage guidance for the reviser.")
        return "\n".join(parts)

    def _build_revision_prompt(
        self,
        messages: list[ChatMessage],
        candidates: list[CandidateResult],
        critique: str,
    ) -> str:
        parts = [
            "Conversation transcript:",
            self._conversation_transcript(messages),
            "",
            "Independent drafts:",
        ]
        parts.extend(self._render_candidates(candidates))
        parts.extend(["", "Critic feedback:", self._truncate_for_judge(critique)])
        return "\n".join(parts)

    def _build_refinement_prompt(
        self,
        messages: list[ChatMessage],
        candidates: list[CandidateResult],
        round_index: int,
    ) -> str:
        parts = [
            f"Refinement round {round_index}.",
            "Conversation transcript:",
            self._conversation_transcript(messages),
            "",
            "Previous layer outputs:",
        ]
        parts.extend(self._render_candidates(candidates))
        return "\n".join(parts)

    def _build_planner_prompt(
        self,
        messages: list[ChatMessage],
        panel: list[str],
        remaining_calls: int,
    ) -> str:
        providers = []
        for name in panel:
            provider = self.providers.get(name)
            if provider:
                providers.append(
                    {
                        "name": name,
                        "model": provider.config.model,
                        "weight": provider.config.weight,
                    }
                )
        schema = {
            "strategy": "one of fallback, parallel_synthesis, best_of_n, majority_vote, "
            "weighted_vote, critique_revision, layered_refinement",
            "panel": ["enabled provider names only"],
            "judge_provider": "enabled provider name or null",
            "critic_provider": "enabled provider name or null",
            "reviser_provider": "enabled provider name or null",
            "samples_per_provider": "integer 1 to 3",
            "refinement_rounds": "integer 0 to 2",
            "rationale": "one brief operational sentence",
        }
        return (
            f"Available providers: {json.dumps(providers)}\n"
            f"Remaining model-call budget after planning: {remaining_calls}\n"
            f"Required JSON shape: {json.dumps(schema)}\n\n"
            f"User conversation:\n{self._conversation_transcript(messages)}"
        )

    def _render_candidates(self, candidates: list[CandidateResult]) -> list[str]:
        rendered: list[str] = []
        for index, candidate in enumerate(candidates, start=1):
            rendered.append(
                "\n"
                f"--- Candidate {index}: provider={candidate.provider}, model={candidate.model}, "
                f"weight={candidate.weight:g}, stage={candidate.stage}, "
                f"sample={candidate.sample_index} ---\n"
                f"{self._truncate_for_judge(candidate.content.strip())}"
            )
        return rendered

    def _conversation_transcript(self, messages: list[ChatMessage]) -> str:
        lines: list[str] = []
        for index, message in enumerate(messages, start=1):
            content = _render_message_content(message.content).strip()
            name = f" name={message.name}" if message.name else ""
            lines.append(f"{index}. {message.role}{name}: {content}")
        transcript = "\n".join(lines) if lines else "(empty)"
        limit = max(500, self.config.fusion.transcript_max_chars)
        if len(transcript) <= limit:
            return transcript
        omitted = len(transcript) - limit
        return f"[truncated {omitted} earlier characters]\n{transcript[-limit:]}"

    def _truncate_for_judge(self, content: str) -> str:
        limit = max(200, self.config.fusion.judge_candidate_max_chars)
        if len(content) <= limit:
            return content
        omitted = len(content) - limit
        return f"{content[:limit]}\n[truncated {omitted} characters before orchestration]"

    @staticmethod
    def _parse_selection(content: str, candidate_count: int) -> tuple[int, str] | None:
        data = _extract_json_object(content)
        if data is None:
            return None
        try:
            winner = int(data.get("winner")) - 1
        except (TypeError, ValueError):
            return None
        if winner < 0 or winner >= candidate_count:
            return None
        reason = str(data.get("reason") or "Selected by evaluator.")[:500]
        return winner, reason

    @staticmethod
    def _vote_key(content: str, answer_regex: str | None) -> str:
        extracted = content.strip()[:20000]
        if answer_regex:
            if len(answer_regex) > 256:
                raise ValueError("fusion vote regex must be 256 characters or fewer")
            try:
                matches = list(re.finditer(answer_regex, extracted, flags=re.IGNORECASE | re.MULTILINE))
            except re.error as exc:
                raise ValueError(f"Invalid fusion vote regex: {exc}") from exc
            if matches:
                match = matches[-1]
                extracted = match.group(1) if match.lastindex else match.group(0)
        else:
            matches = list(
                re.finditer(
                    r"^(?:final\s+answer|answer|choice)\s*[:\-]\s*(.+)$",
                    extracted,
                    flags=re.IGNORECASE | re.MULTILINE,
                )
            )
            if matches:
                extracted = matches[-1].group(1)
        extracted = re.sub(r"[`*_>#]", "", extracted)
        extracted = re.sub(r"\s+", " ", extracted).strip().casefold()
        return extracted.strip(" .,:;!?()[]{}\"'")

    @staticmethod
    def _deterministic_best(candidates: list[CandidateResult]) -> CandidateResult:
        return max(
            candidates,
            key=lambda candidate: (
                candidate.weight,
                len(candidate.content),
                -(candidate.latency_ms or 0),
            ),
        )

    def _visible_candidates(self, candidates: list[CandidateResult]) -> list[CandidateResult]:
        if self.config.fusion.include_candidate_outputs:
            return candidates
        return [candidate.model_copy(update={"content": ""}) for candidate in candidates]

    def _no_success_result(
        self,
        strategy: str,
        candidates: list[CandidateResult],
    ) -> FusionResult:
        return FusionResult(
            strategy=strategy,
            final="No model produced a usable answer.",
            candidates=self._visible_candidates(candidates),
            usage=self._sum_usage(candidates),
        )

    @staticmethod
    def _sum_usage(candidates: list[CandidateResult]) -> Usage:
        total = Usage()
        for candidate in candidates:
            total += candidate.usage
        return total

    def _valid_provider_name(self, value: Any) -> str | None:
        if isinstance(value, str) and value in self.providers:
            return value
        return None

    @staticmethod
    def _bounded_int(value: Any, minimum: int, maximum: int, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return max(minimum, min(maximum, parsed))
