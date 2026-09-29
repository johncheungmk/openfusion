from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from openfusion.config import AppConfig, DecisionModelConfig, FusionConfig, ProviderConfig
from openfusion.decisions import Decision, DecisionClient
from openfusion.fusion import CallBudget, FusionEngine
from openfusion.providers import StaticProvider
from openfusion.schema import ChatMessage


@pytest.mark.asyncio
async def test_system_one_wire_format_and_usage(monkeypatch):
    monkeypatch.setenv("TEST_DECISION_KEY", "secret")

    def respond(request):
        assert request.url == "http://kev/v1/systemone"
        assert request.headers["Authorization"] == "Bearer secret"
        body = json.loads(request.content)
        assert body["state"]["candidates"] == {"candidate_1": "A", "candidate_2": "B"}
        assert body["questions"]["selection"]["type"] == "choice"
        return httpx.Response(200, json={
            "answers": {"selection": {
                "choice": "candidate_2", "probabilities": {"candidate_1": .2, "candidate_2": .8},
            }}, "usage": {"input_tokens": 20, "output_tokens": 5},
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await DecisionClient(DecisionModelConfig(
            base_url="http://kev/v1/", api_key_env="TEST_DECISION_KEY",
        ), client).select("Question", ["A", "B"])
    assert (result.index, result.probability, result.input_tokens) == (1, .8, 20)


@pytest.mark.asyncio
@pytest.mark.parametrize("choice,probs", [
    ("bogus", {"candidate_1": .2, "candidate_2": .8}),
    ("candidate_1", {"candidate_1": .2, "candidate_2": .8}),
    ("candidate_1", {"candidate_1": .9, "candidate_2": .9}),
    ("candidate_1", {"candidate_1": True, "candidate_2": 0}),
    ("candidate_1", {"candidate_1": 1}),
])
async def test_invalid_decisions_rejected(choice, probs):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(
        200, json={"answers": {"selection": {"choice": choice, "probabilities": probs}}},
    ))) as client:
        with pytest.raises(ValueError):
            await DecisionClient(DecisionModelConfig(), client).select("Q", ["A", "B"])


def engine(limit=4, threshold=0, visible=True):
    configs = [ProviderConfig(name=n, base_url=f"http://{n}", model=n) for n in "abc"]
    config = AppConfig(
        providers=configs,
        decision_model=DecisionModelConfig(min_probability=threshold),
        fusion=FusionConfig(panel=list("abc"), max_total_calls=limit,
                            include_candidate_outputs=visible, include_workflow_outputs=visible),
    )
    return FusionEngine(config, providers={c.name: StaticProvider(c, c.name) for c in configs})


@pytest.mark.asyncio
@pytest.mark.parametrize("limit,expected,calls", [(1, "a", 1), (2, "a", 2),
                                               (3, "b", 3), (4, "b", 4)])
async def test_budget_includes_selector(monkeypatch, limit, expected, calls):
    async def select(self, request, candidates):
        return Decision(1, .8, {})

    monkeypatch.setattr(DecisionClient, "select", select)
    result = await engine(limit).run([ChatMessage(role="user", content="Q")],
                                    strategy="decision-select", max_total_calls=100)
    assert result.final == expected
    assert sum(step.model_call for step in result.trace) == calls


@pytest.mark.asyncio
async def test_failure_falls_back_without_leaking_exception(monkeypatch):
    async def select(*args):
        raise RuntimeError("secret credential and private endpoint")

    monkeypatch.setattr(DecisionClient, "select", select)
    result = await engine().run([ChatMessage(role="user", content="Q")], strategy="decision_select")
    assert result.final == "a"
    assert result.failed_model_calls == 1
    assert "secret credential" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_threshold_and_visibility(monkeypatch):
    async def select(*args):
        return Decision(1, .6, {}, 10, 4)

    monkeypatch.setattr(DecisionClient, "select", select)
    result = await engine(threshold=.8, visible=False).run(
        [ChatMessage(role="user", content="Q")], strategy="decision_select",
    )
    assert result.final == "a"
    assert result.trace[-1].status == "fallback"
    assert result.trace[-1].total_tokens == 14
    assert result.judge_analysis is None
    assert all(not c.content for c in result.candidates)


@pytest.mark.parametrize("url", ["ftp://example.org", "relative", "http://u:p@example.org",
                                 "http://example.org?key=secret"])
def test_decision_url_validation(url):
    with pytest.raises(ValidationError):
        DecisionModelConfig(base_url=url)


@pytest.mark.asyncio
async def test_missing_configuration_fails_before_generation():
    app = engine()
    app.config.decision_model = None
    with pytest.raises(ValueError, match="decision_model configuration"):
        await app.run([ChatMessage(role="user", content="Q")], strategy="decision_select")


@pytest.mark.asyncio
@pytest.mark.parametrize("contents,expected", [(["", "", ""], ""), (["", "b", ""], "b")])
async def test_empty_candidates_skip_selector(monkeypatch, contents, expected):
    async def select(*args):
        pytest.fail("selector must not run without two usable candidates")

    monkeypatch.setattr(DecisionClient, "select", select)
    app = engine()
    app.providers = {name: StaticProvider(app.providers[name].config, text)
                     for name, text in zip("abc", contents)}
    result = await app.run([ChatMessage(role="user", content="Q")], strategy="decision_select")
    if expected:
        assert result.final == expected
    else:
        assert result.error
    assert result.ok == bool(expected)


@pytest.mark.asyncio
async def test_timeout_is_propagated_to_engine_boundary():
    def timeout(request):
        raise httpx.ReadTimeout("test", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(timeout)) as client:
        with pytest.raises(httpx.ReadTimeout):
            await DecisionClient(DecisionModelConfig(), client).select("Q", ["A", "B"])


@pytest.mark.asyncio
async def test_exhausted_planner_budget_cannot_be_expanded():
    app = engine()
    budget = CallBudget(1, used=1)
    trace = []
    result = await app._decision_select(
        [ChatMessage(role="user", content="Q")], list("abc"), None, None, None, 1,
        budget, trace,
    )
    assert not result.ok
    assert budget.used == budget.limit == 1
    assert not any(step.model_call for step in trace)
