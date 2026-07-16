"""ADK entry points: the SpectronToolset and the get_spectron_tools factory."""

from __future__ import annotations

from collections.abc import Sequence

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools import FunctionTool
from google.adk.tools.base_toolset import BaseToolset
from surrealdb.spectron import AsyncSpectron
from surrealdb.spectron._scope import ScopeArg

from spectron_google_adk._config import SpectronConfig
from spectron_google_adk._tools import build_tools


def _resolve_client(
    context: str | None,
    endpoint: str | None,
    api_key: str | None,
    client: AsyncSpectron | None,
    config: SpectronConfig | None,
    timeout: float,
    max_retries: int,
) -> tuple[AsyncSpectron, bool]:
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
    owned = AsyncSpectron(
        context,
        endpoint=endpoint,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )
    return owned, True


class SpectronToolset(BaseToolset):
    """A Spectron-backed toolset for Google ADK agents.

    Wraps an ``AsyncSpectron`` client and exposes its memory verbs as ADK
    tools. This is the recommended entry point: an ADK ``Runner`` calls
    ``close`` on shutdown, which closes the client if the toolset created it.

    Add it to an agent directly::

        toolset = SpectronToolset(
            context="acme-prod",
            endpoint="https://api.spectron.example",
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
        client: AsyncSpectron | None = None,
        config: SpectronConfig | None = None,
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
    def client(self) -> AsyncSpectron:
        """The underlying Spectron client."""

        return self._client

    async def get_tools(
        self, readonly_context: ReadonlyContext | None = None
    ) -> list[FunctionTool]:
        return list(self._tools)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()


def get_spectron_tools(
    context: str | None = None,
    *,
    endpoint: str | None = None,
    api_key: str | None = None,
    client: AsyncSpectron | None = None,
    config: SpectronConfig | None = None,
    session_id: str | None = None,
    scope: ScopeArg = None,
    include: Sequence[str] | None = None,
    timeout: float = 30.0,
    max_retries: int = 3,
) -> list[FunctionTool]:
    """Build a list of Spectron-backed ADK tools.

    A convenience for scripts that want a plain tool list rather than a managed
    toolset. Provide either an existing ``client``, a ``config``, or the
    ``context`` / ``endpoint`` / ``api_key`` triple. When this function creates
    the client, that client is not closed automatically; pass your own
    ``client`` or use ``SpectronToolset`` when you need deterministic cleanup.

    Args:
        context: Spectron context id, for example "acme-prod".
        endpoint: Full URL of the Spectron host.
        api_key: Bearer token for the Spectron API.
        client: An existing ``AsyncSpectron`` to reuse instead of the triple.
        config: A ``SpectronConfig`` to build the client from.
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
