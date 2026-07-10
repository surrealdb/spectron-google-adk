"""Spectron memory as tools for Google Agent Development Kit (ADK) agents.

Public API:

- ``SpectronToolset``: an ADK ``BaseToolset`` that owns a Spectron client and
  exposes its memory verbs as tools, with lifecycle cleanup.
- ``get_spectron_tools``: a factory returning a plain list of tools.
- ``SpectronConfig``: connection settings, with a ``from_env`` helper.
- ``DEFAULT_VERBS``: the verbs exposed by default.
"""

from __future__ import annotations

from spectron_google_adk._config import SpectronConfig
from spectron_google_adk._tools import DEFAULT_VERBS, build_tools
from spectron_google_adk._toolset import SpectronToolset, get_spectron_tools

__version__ = "0.1.0"

__all__ = [
    "SpectronToolset",
    "get_spectron_tools",
    "SpectronConfig",
    "DEFAULT_VERBS",
    "build_tools",
    "__version__",
]
