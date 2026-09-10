"""ADK entry points: the AgentMemoryToolset and the get_agent_memory_tools factory."""

from __future__ import annotations

from collections.abc import Sequence

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools import FunctionTool
from google.adk.tools.base_toolset import BaseToolset
from surrealdb.memory import AsyncMemory, ScopeArg

from agent_memory_google_adk._config import AgentMemoryConfig
from agent_memory_google_adk._tools import build_tools


def _resolve_client(
    context: str | None,
    endpoint: str | None,
    api_key: str | None,
    client: AsyncMemory | None,
    config: AgentMemoryConfig | None,
    timeout: float,
    max_retries: int,
) -> tuple[AsyncMemory, bool]:
    """Return a client and whether this call owns (and must close) it."""

    if client is not None:
        return client, False
    if config is not None:
        context, endpoint, api_key = config.context, config.endpoint, config.api_key
        timeout, max_retries = config.timeout, config.max_retries
    if not (context and endpoint and api_key):
        raise ValueError(
            "provide either client=, config=, or all of "
            "context/endpoint/api_key"
        )
    owned = AsyncMemory(
        context,
        endpoint=endpoint,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )
    return owned, True


class AgentMemoryToolset(BaseToolset):
    """An Agent Memory-backed toolset for Google ADK agents.

    Wraps an ``AsyncMemory`` client and exposes its memory verbs as ADK
    tools. This is the recommended entry point: an ADK ``Runner`` calls
    ``close`` on shutdown, which closes the client if the toolset created it.

    Add it to an agent directly::

        toolset = AgentMemoryToolset(
            context="acme-prod",
            endpoint="https://api.agent-memory.example",
            api_key="sk-...",
        )
        agent = Agent(model="gemini-2.5-flash", name="assistant", tools=[toolset])

    Pass ``session_id`` (and optionally ``scope``) to bind the toolset to one
    session or tenant so its memory stays isolated from others.
    """

    def __init__(
        self,
        context: str | None = None,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        client: AsyncMemory | None = None,
        config: AgentMemoryConfig | None = None,
        session_id: str | None = None,
        scope: ScopeArg = None,
        include: Sequence[str] | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__()
        self._client, self._owns_client = _resolve_client(
            context, endpoint, api_key, client, config, timeout, max_retries
        )
        self._tools = build_tools(
            self._client,
            session_id=session_id,
            scope=scope,
            include=include,
        )

    @property
    def client(self) -> AsyncMemory:
        """The underlying Agent Memory client."""

        return self._client

    async def get_tools(
        self, readonly_context: ReadonlyContext | None = None
    ) -> list[FunctionTool]:
        return list(self._tools)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()


def get_agent_memory_tools(
    context: str | None = None,
    *,
    endpoint: str | None = None,
    api_key: str | None = None,
    client: AsyncMemory | None = None,
    config: AgentMemoryConfig | None = None,
    session_id: str | None = None,
    scope: ScopeArg = None,
    include: Sequence[str] | None = None,
    timeout: float = 30.0,
    max_retries: int = 3,
) -> list[FunctionTool]:
    """Build a list of Agent Memory-backed ADK tools.

    A convenience for scripts that want a plain tool list rather than a managed
    toolset. Provide either an existing ``client``, a ``config``, or the
    ``context`` / ``endpoint`` / ``api_key`` triple. When this function creates
    the client, that client is not closed automatically; pass your own
    ``client`` or use ``AgentMemoryToolset`` when you need deterministic cleanup.

    Args:
        context: Agent Memory context id, for example "acme-prod".
        endpoint: Full URL of the Agent Memory host.
        api_key: Bearer token for the Agent Memory API.
        client: An existing ``AsyncMemory`` to reuse instead of the triple.
        config: A ``AgentMemoryConfig`` to build the client from.
        session_id: Optional session id bound to the session-aware tools.
        scope: Optional scope bound to the write tools.
        include: Which verbs to expose. Defaults to all of them.
        timeout: Per-request timeout in seconds when creating a client.
        max_retries: Retry budget when creating a client.

    Returns:
        A list of ``FunctionTool`` ready to pass to an ADK ``Agent``.
    """

    resolved, _ = _resolve_client(
        context, endpoint, api_key, client, config, timeout, max_retries
    )
    return build_tools(
        resolved, session_id=session_id, scope=scope, include=include
    )
