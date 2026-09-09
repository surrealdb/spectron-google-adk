# agent-memory-google-adk

Give [Google ADK](https://github.com/google/adk-python) agents persistent
memory backed by [AgentMemory](https://surrealdb.com/agent-memory),
SurrealDB's agent-memory layer.
 
An agent can store facts, search them back, forget them, and reason over them, with memory that
survives restarts and separate conversations. AgentMemory handles entity
extraction, knowledge-graph storage, temporal facts, and hybrid retrieval; this
package is the thin layer that hands those verbs to an ADK agent.

## Install

```bash
pip install agent-memory-google-adk
```

This pulls in `google-adk` and `surrealdb`. The AgentMemory client ships in the
`surrealdb` package (3.0.0a1 and later), which is installed for you from PyPI.

## Configuration

AgentMemory needs a context id, an endpoint, and an API key. The AgentMemory SDK does
not read environment variables itself, so you pass these in explicitly. For
scripts, `AgentMemoryConfig.from_env()` reads them from the environment for you:

```bash
export AGENT_MEMORY_CONTEXT="acme-prod"
export AGENT_MEMORY_ENDPOINT="https://api.agent_memory.example"
export AGENT_MEMORY_API_KEY="sk-spec-..."
export GOOGLE_API_KEY="your-google-api-key"   # used by the ADK model
```

See `.env.example` for the full list.

## Quickstart

```python
import asyncio
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from agent_memory_google_adk import AgentMemoryToolset

async def main():
    toolset = AgentMemoryToolset(
        context="acme-prod",
        endpoint="https://api.agent_memory.example",
        api_key="sk-spec-...",
    )

    agent = Agent(
        model="gemini-2.5-flash",
        name="assistant",
        description="An assistant with persistent memory.",
        instruction="Store durable facts with remember and look things up with recall.",
        tools=[toolset],
    )

    runner = InMemoryRunner(agent=agent)
    try:
        await runner.run_debug("Remember: Acme Corp, healthcare, 1.2M dollar contract.")
        events = await runner.run_debug("What healthcare contracts do we have?")
        for event in events:
            if event.is_final_response() and event.content:
                for part in event.content.parts:
                    if part.text:
                        print(part.text)
    finally:
        await runner.close()
        await toolset.close()

asyncio.run(main())
```

`AgentMemoryToolset` extends ADK's `BaseToolset`, so an ADK `Runner` closes it on
shutdown. When the toolset creates the client, `close()` closes that client too.

## Two ways to build tools

### AgentMemoryToolset (recommended)

Owns the client and manages its lifecycle. Add it to an agent as a single item
in the `tools` list:

```python
toolset = AgentMemoryToolset(config=AgentMemoryConfig.from_env())
agent = Agent(model="gemini-2.5-flash", name="assistant", tools=[toolset])
```

### get_agent_memory_tools (for quick scripts)

Returns a plain list of tools. Pass your own `client` when you want to control
its lifecycle; otherwise the client it creates lives for the process:

```python
from surrealdb.memory import AsyncMemory
from agent_memory_google_adk import get_agent_memory_tools

client = AsyncMemory(context="acme-prod", endpoint="...", api_key="sk-...")
tools = get_agent_memory_tools(client=client)
agent = Agent(model="gemini-2.5-flash", name="assistant", tools=tools)
```

## Session and tenant isolation

Bind a `session_id` (and optionally a `scope`) when you build the tools. The
values are fixed at build time and are not exposed to the model, so an agent
cannot read or write outside its slice of memory:

```python
toolset = AgentMemoryToolset(config=config, session_id="user-123")
```

Two agents built with the same `session_id` share memory; agents on different
session ids stay isolated. See `examples/sessionized_memory.py`.

## Choosing which verbs to expose

By default all of the verbs below are available. Pass `include=[...]` to expose
a subset, for example a collector agent that can only write and a researcher
agent that can only read (see `examples/multi_agent.py`):

```python
collector = AgentMemoryToolset(config=config, include=["remember"])
researcher = AgentMemoryToolset(config=config, include=["recall", "reflect"])
```

## Tools

| Tool | Argument | What the agent uses it for |
| --- | --- | --- |
| `remember` | `text` | Store a fact, preference, or piece of information. |
| `recall` | `query` | Search memory and get back ranked matching passages. |
| `forget` | `query` | Remove information that matches a description. |
| `reflect` | `query` | Get a written summary synthesized from memory, with evidence. |
| `chat` | `message` | Ask memory a question and get a grounded natural-language answer. |
| `consolidate` | (none) | Pool recent facts into durable observations. |
| `elaborate` | `entity_ref` | Expand an entity's relationships from stored facts. |
| `query_context` | `query` | Build a composed context string for grounding a response. |
| `inspect` | `ref` | Inspect a single memory object by reference. |
| `state` | (none) | Get a snapshot of current working memory. |

Every tool returns a JSON-safe dict with a `status` key that is `"success"` or
`"error"`. A failed AgentMemory request becomes
`{"status": "error", "message": ..., "status_code": ..., "trace_id": ...}` so
the model sees it as data rather than the agent turn failing.

## Examples

- `examples/quickstart.py` - one agent that stores a fact and recalls it.
- `examples/sessionized_memory.py` - memory isolation across sessions.
- `examples/multi_agent.py` - a writer agent and a reader agent sharing a context.

Fill in `.env` from `.env.example`, then run any example, for example
`python examples/quickstart.py`.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests use a fake client, so they need no network access or credentials.

## Links

- AgentMemory: https://surrealdb.com/agent-memory
- AgentMemory docs: https://surrealdb.com/docs/learn/agent_memory
- Google ADK: https://github.com/google/adk-python

## License

Apache 2.0. See [LICENSE](LICENSE).
