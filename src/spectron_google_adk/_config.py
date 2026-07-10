"""Connection configuration for the Spectron client used by the ADK tools."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class SpectronConfig:
    """Everything needed to open a Spectron client.

    The Spectron SDK never reads environment variables, so credentials are
    always passed in explicitly. ``from_env`` is a convenience for scripts and
    examples that keep their secrets in the process environment.
    """

    context: str
    endpoint: str
    api_key: str
    timeout: float = 30.0
    max_retries: int = 3

    @classmethod
    def from_env(
        cls,
        *,
        context_var: str = "SPECTRON_CONTEXT",
        endpoint_var: str = "SPECTRON_ENDPOINT",
        api_key_var: str = "SPECTRON_API_KEY",
    ) -> "SpectronConfig":
        """Build a config from environment variables.

        Raises ``ValueError`` if any of the required variables is missing so
        the failure is obvious at startup rather than on the first request.
        """

        values = {
            "context": os.environ.get(context_var),
            "endpoint": os.environ.get(endpoint_var),
            "api_key": os.environ.get(api_key_var),
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            wanted = {
                "context": context_var,
                "endpoint": endpoint_var,
                "api_key": api_key_var,
            }
            names = ", ".join(wanted[key] for key in missing)
            raise ValueError(f"missing required environment variables: {names}")

        return cls(
            context=values["context"],  # type: ignore[arg-type]
            endpoint=values["endpoint"],  # type: ignore[arg-type]
            api_key=values["api_key"],  # type: ignore[arg-type]
        )
