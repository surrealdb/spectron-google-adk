"""Tool wrapping and return-shape tests using a fake Agent Memory client.

No network or credentials are needed: a stub client returns canned response
dataclasses (and, in one case, raises) so the wrappers can be checked in
isolation.
"""

from __future__ import annotations

import pytest
from surrealdb.memory import (
    ForgetResponse,
    MemoryAPIError,
    RecallHit,
    RecallResponse,
    ReflectResponse,
    RememberResponse,
)

from agent_memory_google_adk import DEFAULT_VERBS, build_tools, get_agent_memory_tools


class FakeClient:
    """Minimal async stand-in for AsyncMemory used by build_tools."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, tuple, dict]] = []

    async def remember(self, text, **kwargs):
        self.calls.append(("remember", (text,), kwargs))
        return RememberResponse(mode="full", session_id="sess:1", turn_id="turn:1")

    async def recall(self, query, **kwargs):
        self.calls.append(("recall", (query,), kwargs))
        if self.fail:
            raise MemoryAPIError(500, "boom", trace_id="trace:xyz")
        return RecallResponse(
            hits=[RecallHit(id="fact:1", score=0.9, source="fact", text="hi")]
        )

    async def forget(self, query, **kwargs):
        self.calls.append(("forget", (query,), kwargs))
        return ForgetResponse(deleted=3)

    async def reflect(self, query, **kwargs):
        self.calls.append(("reflect", (query,), kwargs))
        return ReflectResponse(reflection="a summary", evidence=["fact:1"])


def _tool_by_name(tools, name):
    for tool in tools:
        if tool.name == name:
            return tool
    raise AssertionError(f"tool {name!r} not found")


def test_build_tools_returns_all_default_verbs() -> None:
    tools = build_tools(FakeClient())
    assert [tool.name for tool in tools] == list(DEFAULT_VERBS)


def test_every_tool_has_a_docstring() -> None:
    for tool in build_tools(FakeClient()):
        assert tool.func.__doc__, tool.name
        assert tool.description


def test_include_selects_a_subset() -> None:
    tools = build_tools(FakeClient(), include=["remember", "recall"])
    assert [tool.name for tool in tools] == ["remember", "recall"]


def test_unknown_verb_raises() -> None:
    with pytest.raises(ValueError):
        build_tools(FakeClient(), include=["nope"])


async def test_remember_shape_and_binding() -> None:
    client = FakeClient()
    tools = build_tools(client, session_id="user-123", scope={"org": "acme"})
    result = await _tool_by_name(tools, "remember").func("I work at Acme")
    assert result["status"] == "success"
    assert result["session_id"] == "sess:1"
    assert result["turn_id"] == "turn:1"
    # session_id and scope are bound at build time, not model-chosen.
    _, _, kwargs = client.calls[-1]
    assert kwargs["session_id"] == "user-123"
    assert kwargs["scope"] == {"org": "acme"}


async def test_recall_shape() -> None:
    tools = build_tools(FakeClient())
    result = await _tool_by_name(tools, "recall").func("what do I do")
    assert result["status"] == "success"
    assert result["count"] == 1
    assert result["hits"][0] == {
        "text": "hi",
        "score": 0.9,
        "source": "fact",
        "id": "fact:1",
    }


async def test_forget_and_reflect_shapes() -> None:
    tools = build_tools(FakeClient())
    forget = await _tool_by_name(tools, "forget").func("old job")
    assert forget == {"status": "success", "deleted": 3}
    reflect = await _tool_by_name(tools, "reflect").func("my preferences")
    assert reflect["status"] == "success"
    assert reflect["reflection"] == "a summary"
    assert reflect["evidence"] == ["fact:1"]


async def test_errors_map_to_error_dict() -> None:
    tools = build_tools(FakeClient(fail=True))
    result = await _tool_by_name(tools, "recall").func("anything")
    assert result["status"] == "error"
    assert result["message"] == "boom"
    assert result["status_code"] == 500
    assert result["trace_id"] == "trace:xyz"


def test_get_agent_memory_tools_requires_credentials() -> None:
    with pytest.raises(ValueError):
        get_agent_memory_tools()


def test_get_agent_memory_tools_accepts_a_client() -> None:
    tools = get_agent_memory_tools(client=FakeClient(), include=["recall"])
    assert [tool.name for tool in tools] == ["recall"]
