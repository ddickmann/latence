"""Shared naive language auto-detection heuristic (reused by Source + Parse)."""

from __future__ import annotations

from latence_core.language import UNKNOWN_LANGUAGE, detect_language


def test_detects_english() -> None:
    assert detect_language("the framework reads the folder and exports the records") == "en"


def test_detects_german() -> None:
    assert detect_language("der Rahmen ist nicht das und ein mit") == "de"


def test_empty_text_is_undetermined() -> None:
    assert detect_language("") == UNKNOWN_LANGUAGE
    assert detect_language("   \n\t ") == UNKNOWN_LANGUAGE


def test_no_signal_is_undetermined() -> None:
    # No stop-words from any signature.
    assert detect_language("xyzzy plugh frobnicate quux") == UNKNOWN_LANGUAGE


def test_is_deterministic() -> None:
    text = "the and of to is in that it"
    assert detect_language(text) == detect_language(text) == "en"
