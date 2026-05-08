from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from valuecell.core.super_agent import core as super_agent_mod
from valuecell.core.super_agent.core import (
    SuperAgent,
    SuperAgentDecision,
    SuperAgentOutcome,
)
from valuecell.core.super_agent.service import SuperAgentService
from valuecell.core.types import UserInput, UserInputMetadata


@pytest.mark.asyncio
async def test_super_agent_run_uses_underlying_agent(monkeypatch: pytest.MonkeyPatch):
    fake_response = SimpleNamespace(
        content=SuperAgentOutcome(
            decision=SuperAgentDecision.ANSWER,
            answer_content="Here is a quick reply",
            enriched_query=None,
        ),
        content_type="outcome",
    )

    agent_instance_holder: dict[str, object] = {}

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            # Provide minimal model info for error formatting paths
            self.model = SimpleNamespace(id="fake-model", provider="fake-provider")
            agent_instance_holder["instance"] = self

        async def arun(self, *args, **kwargs):
            yield fake_response

    monkeypatch.setattr(super_agent_mod, "Agent", FakeAgent)
    # Patch model creation to avoid real provider/model access
    monkeypatch.setattr(
        super_agent_mod.model_utils_mod,
        "get_model_for_agent",
        lambda *args, **kwargs: "stub-model",
    )
    monkeypatch.setattr(super_agent_mod, "agent_debug_mode_enabled", lambda: False)

    sa = SuperAgent()

    user_input = UserInput(
        query="answer this",
        target_agent_name=sa.name,
        meta=UserInputMetadata(conversation_id="conv-sa", user_id="user-sa"),
    )

    # Consume async iterator: should yield final outcome
    outcomes = [item async for item in sa.run(user_input) if not isinstance(item, str)]
    assert outcomes and isinstance(outcomes[-1], SuperAgentOutcome)
    assert outcomes[-1].answer_content == "Here is a quick reply"


def test_super_agent_prompts_are_non_empty():
    from valuecell.core.super_agent.prompts import (
        SUPER_AGENT_EXPECTED_OUTPUT,
        SUPER_AGENT_INSTRUCTION,
    )

    assert "<purpose>" in SUPER_AGENT_INSTRUCTION
    assert '"decision"' in SUPER_AGENT_EXPECTED_OUTPUT


@pytest.mark.asyncio
async def test_super_agent_service_delegates_to_underlying_agent():
    async def _run(user_input):
        yield "result"

    fake_agent = SimpleNamespace(
        name="Helper",
        run=_run,
    )
    service = SuperAgentService(super_agent=fake_agent)
    user_input = UserInput(
        query="test",
        target_agent_name="Helper",
        meta=UserInputMetadata(conversation_id="conv", user_id="user"),
    )

    assert service.name == "Helper"
    # Service.run is async iterator passthrough
    items = [item async for item in service.run(user_input)]
    assert items == ["result"]


@pytest.mark.asyncio
async def test_super_agent_run_handles_malformed_response(
    monkeypatch: pytest.MonkeyPatch,
):
    """When underlying agent returns non-SuperAgentOutcome, SuperAgent falls back to ANSWER with explanatory text."""

    # Return a malformed content (not a SuperAgentOutcome instance)
    fake_response = SimpleNamespace(
        content=SimpleNamespace(raw="oops"), content_type="malformed"
    )

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            # Minimal model attributes used in error formatting
            self.model = SimpleNamespace(id="fake-model", provider="fake-provider")

        async def arun(self, *args, **kwargs):
            yield fake_response

    monkeypatch.setattr(super_agent_mod, "Agent", FakeAgent)
    monkeypatch.setattr(
        super_agent_mod.model_utils_mod,
        "get_model_for_agent",
        lambda *args, **kwargs: "stub-model",
    )
    monkeypatch.setattr(super_agent_mod, "agent_debug_mode_enabled", lambda: False)

    sa = SuperAgent()
    user_input = UserInput(
        query="give answer",
        target_agent_name=sa.name,
        meta=UserInputMetadata(conversation_id="conv", user_id="user"),
    )

    # Fallback path should return an ANSWER decision with helpful message
    outcomes = [item async for item in sa.run(user_input) if not isinstance(item, str)]
    assert outcomes and outcomes[-1].decision == SuperAgentDecision.ANSWER
    assert "malformed response" in outcomes[-1].answer_content
    assert "malformed structured output" in (outcomes[-1].reason or "")


@pytest.mark.asyncio
async def test_super_agent_lazy_init_failure_handoff_to_planner(
    monkeypatch: pytest.MonkeyPatch,
):
    """When SuperAgent cannot initialize, it hands off directly to Planner."""

    def _raise(*_args, **_kwargs):
        raise RuntimeError("no model")

    monkeypatch.setattr(super_agent_mod.model_utils_mod, "get_model_for_agent", _raise)
    monkeypatch.setattr(super_agent_mod, "agent_debug_mode_enabled", lambda: False)

    sa = SuperAgent()

    user_input = UserInput(
        query="please plan",
        target_agent_name=sa.name,
        meta=UserInputMetadata(conversation_id="conv-fallback", user_id="user-x"),
    )

    outcomes = [item async for item in sa.run(user_input) if not isinstance(item, str)]
    assert outcomes and outcomes[-1].decision == SuperAgentDecision.ANSWER
    assert outcomes[-1].enriched_query is None
    assert outcomes[-1].answer_content and "arun" in outcomes[-1].answer_content
    assert outcomes[-1].reason == "SuperAgent execution failed"


@pytest.mark.asyncio
async def test_super_agent_malformed_response_unknown_provider(
    monkeypatch: pytest.MonkeyPatch,
):
    """Malformed response with missing model info uses 'unknown model/provider' label."""

    # Return a malformed content (not a SuperAgentOutcome instance)
    fake_response = SimpleNamespace(
        content=SimpleNamespace(raw="oops"), content_type="malformed"
    )

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            # No model attribute to trigger unknown path
            # self.model = missing
            pass

        async def arun(self, *args, **kwargs):
            yield fake_response

    monkeypatch.setattr(super_agent_mod, "Agent", FakeAgent)
    monkeypatch.setattr(
        super_agent_mod.model_utils_mod,
        "get_model_for_agent",
        lambda *args, **kwargs: "stub-model",
    )
    monkeypatch.setattr(super_agent_mod, "agent_debug_mode_enabled", lambda: False)

    sa = SuperAgent()
    user_input = UserInput(
        query="give answer",
        target_agent_name=sa.name,
        meta=UserInputMetadata(conversation_id="conv", user_id="user"),
    )

    outcomes = [item async for item in sa.run(user_input) if not isinstance(item, str)]
    assert outcomes and outcomes[-1].decision == SuperAgentDecision.ANSWER
    assert outcomes[-1].answer_content is not None
    assert outcomes[-1].reason is not None
    assert "unknown model/provider" in outcomes[-1].answer_content


@pytest.mark.asyncio
async def test_super_agent_timeout_falls_back_to_planner(
    monkeypatch: pytest.MonkeyPatch,
):
    """Timeout while waiting for structured output should hand off to planner."""

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            self.model = SimpleNamespace(id="fake-model", provider="fake-provider")

        async def arun(self, *args, **kwargs):
            if False:
                yield None

    monkeypatch.setattr(super_agent_mod, "Agent", FakeAgent)
    monkeypatch.setattr(
        super_agent_mod.model_utils_mod,
        "get_model_for_agent",
        lambda *args, **kwargs: "stub-model",
    )
    monkeypatch.setattr(super_agent_mod, "agent_debug_mode_enabled", lambda: False)

    async def fake_wait_for(awaitable, timeout):
        raise TimeoutError()

    monkeypatch.setattr(super_agent_mod.asyncio, "wait_for", fake_wait_for)

    sa = SuperAgent()
    user_input = UserInput(
        query="hello",
        target_agent_name=sa.name,
        meta=UserInputMetadata(conversation_id="conv-timeout", user_id="user"),
    )

    outcomes = [item async for item in sa.run(user_input) if not isinstance(item, str)]
    assert outcomes
    assert outcomes[-1].decision == SuperAgentDecision.HANDOFF_TO_PLANNER
    assert outcomes[-1].enriched_query == "hello"
    assert "timed out" in (outcomes[-1].reason or "")
