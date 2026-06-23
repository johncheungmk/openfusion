from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

from .config import AppConfig
from .providers import ModelProvider, ProviderClientPool, make_provider
from .schema import CandidateResult, ChatMessage, FusionResult, ProviderRequest, Usage


JUDGE_SYSTEM_PROMPT = """You are OpenFusion, an impartial judge and synthesis model.
You will receive answers from multiple LLMs. Your job is to:
1. identify consensus,
2. identify contradictions and likely errors,
3. identify missing points,
4. produce one final answer that is more reliable than any single candidate.
Do not reveal hidden chain-of-thought. Give concise, user-visible reasoning only when helpful.
"""


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


class FusionEngine:
    def __init__(self, config: AppConfig, providers: dict[str, ModelProvider] | None = None):
        self.config = config
        self._client_pool = ProviderClientPool() if providers is None else None
        self.providers = providers or {
            provider_config.name: make_provider(provider_config, client_pool=self._client_pool)
            for provider_config in config.providers
            if provider_config.enabled
        }

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
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> FusionResult:
        selected_strategy = strategy or self.config.fusion.default_strategy
        if selected_strategy in {"panel_judge", "parallel_judge", "fusion"}:
            return await self._panel_judge(
                messages=messages,
                panel=list(panel) if panel is not None else None,
                judge_provider=judge_provider,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
        if selected_strategy in {"fastest", "fallback"}:
            return await self._fallback(
                messages=messages,
                panel=list(panel) if panel is not None else None,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
        raise ValueError(f"Unknown fusion strategy: {selected_strategy}")

    async def run_provider(
        self,
        provider_name: str,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> FusionResult:
        request = self._provider_request(messages, temperature, max_tokens, extra_body)
        result = await self._call_provider(provider_name, request, asyncio.Semaphore(1))
        candidates = self._visible_candidates([result])
        return FusionResult(
            strategy="direct_provider",
            final=result.content if result.ok and result.content.strip() else "",
            candidates=candidates,
            usage=result.usage,
        )

    async def _call_provider(
        self,
        provider_name: str,
        request: ProviderRequest,
        semaphore: asyncio.Semaphore,
    ) -> CandidateResult:
        provider = self.providers.get(provider_name)
        if provider is None:
            return CandidateResult(
                provider=provider_name,
                model="unknown",
                ok=False,
                error=f"Provider not found or not enabled: {provider_name}",
            )
        async with semaphore:
            result = await provider.chat(request)
            result.weight = provider.config.weight
            result.model = provider.config.model
            return result

    async def _call_many(
        self,
        provider_names: list[str],
        request: ProviderRequest,
    ) -> list[CandidateResult]:
        semaphore = asyncio.Semaphore(max(1, self.config.fusion.max_parallel))
        tasks = [self._call_provider(name, request, semaphore) for name in provider_names]
        return list(await asyncio.gather(*tasks))

    async def _panel_judge(
        self,
        messages: list[ChatMessage],
        panel: list[str] | None,
        judge_provider: str | None,
        temperature: float | None,
        max_tokens: int | None,
        extra_body: dict[str, Any] | None,
    ) -> FusionResult:
        panel_names = panel or self.config.fusion.panel or list(self.providers.keys())
        if not panel_names:
            raise ValueError("No providers available for fusion panel")

        request = self._provider_request(messages, temperature, max_tokens, extra_body)
        candidates = await self._call_many(panel_names, request)
        successes = [candidate for candidate in candidates if candidate.ok and candidate.content.strip()]
        if len(successes) < self.config.fusion.require_at_least_successes:
            return FusionResult(
                strategy="panel_judge",
                final="No model produced a usable answer.",
                candidates=self._visible_candidates(candidates),
                usage=self._sum_usage(candidates),
            )

        selected_judge = judge_provider or self.config.fusion.judge_provider or successes[0].provider
        judge_prompt = self._build_judge_prompt(messages, successes)
        judge_request = ProviderRequest(
            messages=[
                ChatMessage(role="system", content=JUDGE_SYSTEM_PROMPT),
                ChatMessage(role="user", content=judge_prompt),
            ],
            temperature=0.1,
            max_tokens=max_tokens if max_tokens is not None else self.config.fusion.max_tokens,
        )
        judge_result = await self._call_provider(
            selected_judge,
            judge_request,
            asyncio.Semaphore(1),
        )

        if judge_result.ok and judge_result.content.strip():
            usage = self._sum_usage(candidates) + judge_result.usage
            return FusionResult(
                strategy="panel_judge",
                final=judge_result.content,
                judge_provider=selected_judge,
                judge_analysis=judge_result.content,
                candidates=self._visible_candidates(candidates),
                usage=usage,
            )

        # Graceful degradation: return the longest successful candidate.
        best = max(successes, key=lambda item: len(item.content))
        return FusionResult(
            strategy="panel_judge",
            final=best.content,
            judge_provider=selected_judge,
            judge_analysis=f"Judge failed: {judge_result.error or 'empty response'}",
            candidates=self._visible_candidates(candidates),
            usage=self._sum_usage(candidates),
        )

    async def _fallback(
        self,
        messages: list[ChatMessage],
        panel: list[str] | None,
        temperature: float | None,
        max_tokens: int | None,
        extra_body: dict[str, Any] | None,
    ) -> FusionResult:
        panel_names = panel or self.config.fusion.panel or list(self.providers.keys())
        request = self._provider_request(messages, temperature, max_tokens, extra_body)
        candidates: list[CandidateResult] = []
        for provider_name in panel_names:
            result = await self._call_provider(provider_name, request, asyncio.Semaphore(1))
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

    def _build_judge_prompt(self, messages: list[ChatMessage], candidates: list[CandidateResult]) -> str:
        original_question = _latest_user_message(messages)
        parts = [
            "Conversation transcript:",
            self._conversation_transcript(messages),
            "",
            "Latest user request:",
            original_question,
            "",
            "Candidate answers. Provider weights are advisory confidence/routing hints from config:",
        ]
        for idx, candidate in enumerate(candidates, start=1):
            content = self._truncate_for_judge(candidate.content.strip())
            parts.append(
                "\n"
                f"--- Candidate {idx}: provider={candidate.provider}, model={candidate.model}, "
                f"weight={candidate.weight:g} ---\n"
                f"{content}"
            )
        parts.append(
            """
Return the final response with these sections when useful:
- Answer
- Important agreements
- Differences or caveats
Do not include raw candidate labels unless needed. Do not reveal hidden chain-of-thought.
""".strip()
        )
        return "\n".join(parts)

    @staticmethod
    def _conversation_transcript(messages: list[ChatMessage]) -> str:
        lines: list[str] = []
        for index, message in enumerate(messages, start=1):
            content = _render_message_content(message.content).strip()
            name = f" name={message.name}" if message.name else ""
            lines.append(f"{index}. {message.role}{name}: {content}")
        return "\n".join(lines) if lines else "(empty)"

    def _truncate_for_judge(self, content: str) -> str:
        limit = max(200, self.config.fusion.judge_candidate_max_chars)
        if len(content) <= limit:
            return content
        omitted = len(content) - limit
        return f"{content[:limit]}\n[truncated {omitted} characters before judge synthesis]"

    def _visible_candidates(self, candidates: list[CandidateResult]) -> list[CandidateResult]:
        if self.config.fusion.include_candidate_outputs:
            return candidates
        return [candidate.model_copy(update={"content": ""}) for candidate in candidates]

    @staticmethod
    def _sum_usage(candidates: list[CandidateResult]) -> Usage:
        total = Usage()
        for candidate in candidates:
            total += candidate.usage
        return total
