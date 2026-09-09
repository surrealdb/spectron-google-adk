"""AgentMemory memory as tools for Google Agent Development Kit (ADK) agents.

Public API:

- ``AgentMemoryToolset``: an ADK ``BaseToolset`` that owns a AgentMemory client and
  exposes its memory verbs as tools, with lifecycle cleanup.
- ``get_agent_memory_tools``: a factory returning a plain list of tools.
- ``AgentMemoryConfig``: connection settings, with a ``from_env`` helper.
- ``DEFAULT_VERBS``: the verbs exposed by default.
"""

from __future__ import annotations

from agent_memory_google_adk._config import AgentMemoryConfig
from agent_memory_google_adk._tools import DEFAULT_VERBS, build_tools
from agent_memory_google_adk._toolset import AgentMemoryToolset, get_agent_memory_tools

__version__ = "0.3.0"

__all__ = [
    "AgentMemoryToolset",
    "get_agent_memory_tools",
    "AgentMemoryConfig",
    "DEFAULT_VERBS",
    "build_tools",
    "__version__",
]
