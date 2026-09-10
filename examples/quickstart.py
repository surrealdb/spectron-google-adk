"""Quickstart: an ADK agent with Agent Memory.

The agent gets the full Agent Memory tool set. It stores a fact on the first turn
and recalls it on the second. Run it after filling in .env (see .env.example).

    python examples/quickstart.py
"""

from __future__ import annotations

import asyncio

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner

from agent_memory_google_adk import AgentMemoryConfig, AgentMemoryToolset


def _print_final(label: str, events) -> None:
    for event in events:
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if part.text:
                    print(f"[{label}] {part.text}")


async def main() -> None:
    config = AgentMemoryConfig.from_env()
    toolset = AgentMemoryToolset(config=config)

    agent = Agent(
        model="gemini-2.5-flash",
        name="assistant",
        description="An assistant with persistent memory backed by AgentMemory.",
        instruction=(
            "You are a helpful assistant with a long-term memory. "
            "Use the remember tool to store durable facts the user shares, "
            "and the recall tool to look things up before answering."
        ),
        tools=[toolset],
    )

    runner = InMemoryRunner(agent=agent)
    try:
        _print_final(
            "store",
            await runner.run_debug(
                "Remember: Acme Corp, healthcare sector, 1.2M dollar contract."
            ),
        )
        _print_final(
            "recall",
            await runner.run_debug("What healthcare contracts do we have?"),
        )
    finally:
        await runner.close()
        await toolset.close()


if __name__ == "__main__":
    asyncio.run(main())
