"""Per-session memory isolation.

Binding a tool set to a session_id keeps one user's or one tenant's memory
separate from everyone else's. Here two independent agents share the same
session_id, so the second one recalls what the first stored, while a third
agent on a different session_id sees nothing.

    python examples/sessionized_memory.py
"""

from __future__ import annotations

import asyncio

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner

from spectron_google_adk import SpectronConfig, SpectronToolset


def _print_final(label: str, events) -> None:
    for event in events:
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if part.text:
                    print(f"[{label}] {part.text}")


def _build_agent(toolset: SpectronToolset) -> Agent:
    return Agent(
        model="gemini-2.5-flash",
        name="assistant",
        description="An assistant with session-scoped memory.",
        instruction=(
            "You are a helpful assistant. Store durable facts with the "
            "remember tool and look things up with the recall tool."
        ),
        tools=[toolset],
    )


async def _run(toolset: SpectronToolset, message: str, label: str) -> None:
    agent = _build_agent(toolset)
    runner = InMemoryRunner(agent=agent)
    try:
        _print_final(label, await runner.run_debug(message))
    finally:
        await runner.close()


async def main() -> None:
    config = SpectronConfig.from_env()

    # First agent, session user-123: store something.
    store = SpectronToolset(config=config, session_id="user-123")
    try:
        await _run(store, "I'm working on the authentication service.", "store")
    finally:
        await store.close()

    # Second agent, same session: the fact is still there.
    same = SpectronToolset(config=config, session_id="user-123")
    try:
        await _run(same, "What was I working on?", "same-session")
    finally:
        await same.close()

    # Third agent, different session: isolated, should not see it.
    other = SpectronToolset(config=config, session_id="user-999")
    try:
        await _run(other, "What was I working on?", "other-session")
    finally:
        await other.close()


if __name__ == "__main__":
    asyncio.run(main())
