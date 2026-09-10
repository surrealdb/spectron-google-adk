"""The public API imports and exposes what it promises."""

from __future__ import annotations

import agent_memory_google_adk


def test_public_api() -> None:
    for name in (
        "AgentMemoryToolset",
        "get_agent_memory_tools",
        "AgentMemoryConfig",
        "DEFAULT_VERBS",
        "build_tools",
    ):
        assert hasattr(agent_memory_google_adk, name), name


def test_version_is_a_string() -> None:
    assert isinstance(agent_memory_google_adk.__version__, str)


def test_default_verbs_cover_the_expected_set() -> None:
    assert set(agent_memory_google_adk.DEFAULT_VERBS) == {
        "remember",
        "recall",
        "forget",
        "reflect",
        "chat",
        "consolidate",
        "elaborate",
        "query_context",
        "inspect",
        "state",
    }
