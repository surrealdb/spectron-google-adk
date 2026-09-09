"""Build Google ADK function tools backed by a AgentMemory client.

Each AgentMemory memory verb is wrapped as an async function whose docstring and
type hints ADK turns into the schema the model sees. The wrappers return plain
JSON-safe dicts and follow ADK's ``{"status": "success" | "error", ...}``
convention, so a failed request surfaces to the model as data instead of
crashing the agent turn.

``session_id`` and ``scope`` are bound when the tools are built, not exposed to
the model. That keeps each agent (or each user session) inside its own slice of
memory: the model cannot widen its own scope by choosing a different argument.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from google.adk.tools import FunctionTool
from surrealdb.memory import AsyncMemory, MemoryServiceError, ScopeArg

# The verbs exposed as tools by default, in a sensible order for the model.
DEFAULT_VERBS: tuple[str, ...] = (
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
)


def _error(exc: MemoryServiceError) -> dict[str, Any]:
    """Turn a AgentMemory exception into the ADK error dict shape."""

    result: dict[str, Any] = {
        "status": "error",
        "message": getattr(exc, "message", None) or str(exc),
    }
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        result["status_code"] = status_code
    trace_id = getattr(exc, "trace_id", None)
    if trace_id is not None:
        result["trace_id"] = trace_id
    return result


def _build_remember(
    client: AsyncMemory, session_id: str | None, scope: ScopeArg
) -> Callable:
    async def remember(text: str) -> dict:
        """Store a fact, preference, or piece of information in long-term memory.

        Call this whenever the user shares something worth keeping for later:
        who they are, what they are working on, decisions, or any durable fact.
        AgentMemory extracts the entities and relationships automatically.

        Args:
            text: The information to remember, written in plain language.

        Returns:
            A dict with a "status" key. On success it also carries the
            "session_id" and "turn_id" the fact was recorded under.
        """
        try:
            response = await client.remember(
                text, session_id=session_id, scope=scope
            )
        except MemoryServiceError as exc:
            return _error(exc)
        result: dict[str, Any] = {
            "status": "success",
            "session_id": response.session_id,
            "mode": response.mode,
        }
        if response.turn_id is not None:
            result["turn_id"] = response.turn_id
        if response.extraction is not None:
            result["extraction"] = response.extraction.to_dict()
        return result

    return remember


def _build_recall(
    client: AsyncMemory, session_id: str | None, scope: ScopeArg
) -> Callable:
    async def recall(query: str) -> dict:
        """Search long-term memory for information relevant to a query.

        Use this before answering questions that may depend on earlier facts.
        Results are ranked by relevance and drawn from stored facts and any
        uploaded documents.

        Args:
            query: A natural-language description of what to look for.

        Returns:
            A dict with a "status" key and, on success, a "hits" list. Each hit
            has "text", "score", "source", and "id".
        """
        try:
            response = await client.recall(query, session_id=session_id)
        except MemoryServiceError as exc:
            return _error(exc)
        hits = [
            {
                "text": hit.text,
                "score": hit.score,
                "source": hit.source,
                "id": hit.id,
            }
            for hit in response.hits
        ]
        return {"status": "success", "count": len(hits), "hits": hits}

    return recall


def _build_forget(
    client: AsyncMemory, session_id: str | None, scope: ScopeArg
) -> Callable:
    async def forget(query: str) -> dict:
        """Remove information from memory that matches a description.

        Use this when the user asks to delete or retract something. The match
        is semantic, so describe what should be forgotten in plain language.

        Args:
            query: A description of the information to forget.

        Returns:
            A dict with a "status" key and, on success, the "deleted" count.
        """
        try:
            response = await client.forget(query)
        except MemoryServiceError as exc:
            return _error(exc)
        return {"status": "success", "deleted": response.deleted}

    return forget


def _build_reflect(
    client: AsyncMemory, session_id: str | None, scope: ScopeArg
) -> Callable:
    async def reflect(query: str) -> dict:
        """Summarize what memory knows about a topic and synthesize an answer.

        Unlike recall, which returns raw matching passages, reflect runs a pass
        over the retrieved context and returns a written summary with the
        evidence it drew on.

        Args:
            query: The topic or question to reflect on.

        Returns:
            A dict with a "status" key and, on success, a "reflection" string
            and an "evidence" list.
        """
        try:
            response = await client.reflect(query)
        except MemoryServiceError as exc:
            return _error(exc)
        return {
            "status": "success",
            "reflection": response.reflection,
            "evidence": response.evidence,
        }

    return reflect


def _build_chat(
    client: AsyncMemory, session_id: str | None, scope: ScopeArg
) -> Callable:
    async def chat(message: str) -> dict:
        """Ask memory a question and get an answer grounded in stored facts.

        This runs AgentMemory's own retrieval-and-answer pipeline over the
        context. Prefer recall when you want raw passages to reason over
        yourself, and chat when you want a ready-made grounded reply.

        Args:
            message: The question to ask.

        Returns:
            A dict with a "status" key and, on success, the "reply" text.
        """
        try:
            response = await client.chat(
                message, session_id=session_id, scope=scope
            )
        except MemoryServiceError as exc:
            return _error(exc)
        return {
            "status": "success",
            "reply": response.reply,
            "session_id": response.session_id,
            "trace_id": response.trace_id,
        }

    return chat


def _build_consolidate(
    client: AsyncMemory, session_id: str | None, scope: ScopeArg
) -> Callable:
    async def consolidate() -> dict:
        """Consolidate recent facts into durable, higher-level observations.

        Run this occasionally to let memory pool related facts into stable
        observations. It takes no arguments.

        Returns:
            A dict with a "status" key and, on success, counts of observations
            created, updated, and superseded.
        """
        try:
            response = await client.consolidate()
        except MemoryServiceError as exc:
            return _error(exc)
        return {
            "status": "success",
            "created": response.created,
            "updated": response.updated,
            "superseded": response.superseded,
        }

    return consolidate


def _build_elaborate(
    client: AsyncMemory, session_id: str | None, scope: ScopeArg
) -> Callable:
    async def elaborate(entity_ref: str) -> dict:
        """Expand an entity's relationships from what is already in memory.

        Use this to enrich a known entity with relations inferred from stored
        facts.

        Args:
            entity_ref: The entity reference, for example
                "entity:person/stu".

        Returns:
            A dict with a "status" key and, on success, the number of relations
            emitted.
        """
        try:
            response = await client.elaborate(entity_ref=entity_ref)
        except MemoryServiceError as exc:
            return _error(exc)
        return {
            "status": "success",
            "relations_emitted": response.relations_emitted,
        }

    return elaborate


def _build_query_context(
    client: AsyncMemory, session_id: str | None, scope: ScopeArg
) -> Callable:
    async def query_context(query: str) -> dict:
        """Build a composed context string relevant to a query.

        Returns a single block of context assembled from memory, suitable for
        grounding a longer answer.

        Args:
            query: What the context should be about.

        Returns:
            A dict with a "status" key and, on success, the "context" string.
        """
        try:
            response = await client.query_context(query)
        except MemoryServiceError as exc:
            return _error(exc)
        return {
            "status": "success",
            "context": response.context,
            "tier": response.tier,
        }

    return query_context


def _build_inspect(
    client: AsyncMemory, session_id: str | None, scope: ScopeArg
) -> Callable:
    async def inspect(ref: str) -> dict:
        """Inspect a single memory object by reference.

        The reference grammar is "entity:<type>/<name>",
        "attribute:<type>/<name>/<key>",
        "relation:<subject>/<label>/<object>", or "trace:<id>".

        Args:
            ref: The reference of the object to inspect.

        Returns:
            A dict with a "status" key and, on success, the raw "object"
            payload.
        """
        try:
            payload = await client.inspect(ref)
        except MemoryServiceError as exc:
            return _error(exc)
        return {"status": "success", "object": payload}

    return inspect


def _build_state(
    client: AsyncMemory, session_id: str | None, scope: ScopeArg
) -> Callable:
    async def state() -> dict:
        """Get a snapshot of the context's current working memory.

        Useful for grounding on what is active right now: recent context,
        identity, and known facts. Takes no arguments.

        Returns:
            A dict with a "status" key and, on success, the working-memory
            snapshot under "state".
        """
        try:
            response = await client.state()
        except MemoryServiceError as exc:
            return _error(exc)
        return {"status": "success", "state": response.to_dict()}

    return state


_BUILDERS: dict[str, Callable[[AsyncMemory, str | None, ScopeArg], Callable]] = {
    "remember": _build_remember,
    "recall": _build_recall,
    "forget": _build_forget,
    "reflect": _build_reflect,
    "chat": _build_chat,
    "consolidate": _build_consolidate,
    "elaborate": _build_elaborate,
    "query_context": _build_query_context,
    "inspect": _build_inspect,
    "state": _build_state,
}


def build_tools(
    client: AsyncMemory,
    *,
    session_id: str | None = None,
    scope: ScopeArg = None,
    include: Sequence[str] | None = None,
) -> list[FunctionTool]:
    """Wrap AgentMemory verbs as ADK FunctionTools bound to one client.

    Args:
        client: An open ``AsyncMemory`` instance.
        session_id: Optional session id bound to the ``remember``, ``recall``,
            and ``chat`` tools for per-session isolation.
        scope: Optional scope bound to the ``remember`` and ``chat`` tools.
        include: Which verbs to expose. Defaults to ``DEFAULT_VERBS``.

    Returns:
        A list of ``FunctionTool`` ready to pass to an ADK ``Agent``.
    """

    verbs = tuple(include) if include is not None else DEFAULT_VERBS
    unknown = [verb for verb in verbs if verb not in _BUILDERS]
    if unknown:
        known = ", ".join(_BUILDERS)
        raise ValueError(
            f"unknown AgentMemory verb(s): {', '.join(unknown)}. Known verbs: {known}"
        )

    tools: list[FunctionTool] = []
    for verb in verbs:
        func = _BUILDERS[verb](client, session_id, scope)
        tools.append(FunctionTool(func=func))
    return tools
