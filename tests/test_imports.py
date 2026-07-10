"""The public API imports and exposes what it promises."""

from __future__ import annotations

import spectron_google_adk


def test_public_api() -> None:
    for name in (
        "SpectronToolset",
        "get_spectron_tools",
        "SpectronConfig",
        "DEFAULT_VERBS",
        "build_tools",
    ):
        assert hasattr(spectron_google_adk, name), name


def test_version_is_a_string() -> None:
    assert isinstance(spectron_google_adk.__version__, str)


def test_default_verbs_cover_the_expected_set() -> None:
    assert set(spectron_google_adk.DEFAULT_VERBS) == {
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
    }
