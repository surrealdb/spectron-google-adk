"""Two agents sharing one memory context.

A collector agent can only write (remember). A researcher agent can only read
(recall, reflect). Both point at the same AgentMemory context, so knowledge the
collector stores is available to the researcher. Splitting the verbs this way
keeps each agent's job narrow while the memory stays shared.

    python examples/multi_agent.py
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


async def _run(agent: Agent, message: str, label: str) -> None:
    runner = InMemoryRunner(agent=agent)
    try:
        _print_final(label, await runner.run_debug(message))
    finally:
        await runner.close()


async def main() -> None:
    config = AgentMemoryConfig.from_env()

    collector_tools = AgentMemoryToolset(config=config, include=["remember"])
    researcher_tools = AgentMemoryToolset(config=config, include=["recall", "reflect"])

    collector = Agent(
        model="gemini-2.5-flash",
        name="data_collector",
        description="Collects and stores information.",
        instruction=(
            "You collect important information and store it "
            "with the remember tool."
        ),
        tools=[collector_tools],
    )
    researcher = Agent(
        model="gemini-2.5-flash",
        name="researcher",
        description="Searches and analyzes stored information.",
        instruction=(
            "You answer questions using the recall and reflect tools "
            "over the shared knowledge base."
        ),
        tools=[researcher_tools],
    )

    try:
        await _run(
            collector,
            "Note that Beta Ltd renewed for two years at 800K dollars.",
            "collector",
        )
        await _run(
            researcher,
            "Which customers have renewed, and for how much?",
            "researcher",
        )
    finally:
        await collector_tools.close()
        await researcher_tools.close()


if __name__ == "__main__":
    asyncio.run(main())
