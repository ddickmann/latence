"""Typed error taxonomy regression tests (#73).

Pins the two properties the Hardening-bar taxonomy must hold, either of which a revert of
``errors.py`` (or the re-homing of the existing exceptions) would break:

1. Each category maps to a **stable lowercase tag** and to the **builtin it replaces**, so
   both a taxonomy-aware ``except LatenceError`` *and* every pre-taxonomy ``except
   ValueError`` / ``except RuntimeError`` / ``except TypeError`` call site keep working.
2. The framework's previously-bare exceptions (``SchemeNotAllowedError``,
   ``CorpusStoreError``, ``EndpointError``) now **subclass the taxonomy** while remaining
   subclasses of their original builtin — the additive-not-breaking guarantee.
"""

from __future__ import annotations

import pytest
from latence_core.endpoint import EndpointError
from latence_core.errors import (
    CATEGORY_UNKNOWN,
    ConfigError,
    ContractError,
    LatenceError,
    ProviderError,
    ScreeningQuarantineError,
    StorageError,
    error_category,
)
from latence_core.state import CorpusStoreError
from latence_core.storage import SchemeNotAllowedError


def test_categories_are_the_stable_lowercase_tags() -> None:
    assert ContractError("x").category == "contract"
    assert ConfigError("x").category == "config"
    assert ProviderError("x").category == "provider"
    assert StorageError("x").category == "storage"
    assert ScreeningQuarantineError("x", record_id="r", reason="why").category == "screening"
    assert LatenceError("x").category == CATEGORY_UNKNOWN


def test_error_category_maps_taxonomy_and_off_taxonomy() -> None:
    # A taxonomy error maps to its own category…
    assert error_category(ContractError("x")) == "contract"
    assert error_category(ConfigError("x")) == "config"
    # …a stray builtin (a real bug, not a framework-typed error) maps to "unknown", so the
    # Runner can label ANY caught exception without importing every provider's error type.
    assert error_category(KeyError("boom")) == CATEGORY_UNKNOWN
    assert error_category(RuntimeError("boom")) == CATEGORY_UNKNOWN


def test_taxonomy_is_additive_not_breaking() -> None:
    # Every category is a LatenceError (the one catch-all root) …
    for exc in (
        ContractError("x"),
        ConfigError("x"),
        ProviderError("x"),
        StorageError("x"),
        ScreeningQuarantineError("x", record_id="r", reason="why"),
    ):
        assert isinstance(exc, LatenceError)
    # … AND each still IS the builtin the pre-taxonomy code caught, so old ``except`` clauses
    # keep working. A revert that drops the multiple-inheritance breaks exactly this.
    assert isinstance(ContractError("x"), TypeError)
    assert isinstance(ConfigError("x"), ValueError)
    assert isinstance(ProviderError("x"), RuntimeError)
    assert isinstance(StorageError("x"), RuntimeError)


def test_rehomed_exceptions_subclass_the_taxonomy_and_the_old_builtin() -> None:
    # SchemeNotAllowedError: a ConfigError now, still a ValueError (the ADR-0009 guard's
    # pre-taxonomy contract).
    assert issubclass(SchemeNotAllowedError, ConfigError)
    assert issubclass(SchemeNotAllowedError, ValueError)
    # CorpusStoreError: a StorageError now, still a RuntimeError (the S11 store's contract).
    assert issubclass(CorpusStoreError, StorageError)
    assert issubclass(CorpusStoreError, RuntimeError)
    # EndpointError: a ProviderError now, still a RuntimeError (the S12 retry loop's contract).
    assert issubclass(EndpointError, ProviderError)
    assert issubclass(EndpointError, RuntimeError)


def test_endpoint_error_retryable_agrees_with_status_classing() -> None:
    # The re-homing must not change the status-derived retry classing the ported retry relies on:
    # 429/5xx/transport retryable, other 4xx terminal — and the base ProviderError.retryable
    # seeded from it must agree.
    assert EndpointError("rate limited", status=429).retryable is True
    assert EndpointError("overloaded", status=503).retryable is True
    assert EndpointError("dropped connection", status=None).retryable is True
    assert EndpointError("bad request", status=400).retryable is False
    assert EndpointError("unauthorized", status=401).retryable is False


def test_screening_quarantine_error_carries_id_and_reason_no_content() -> None:
    exc = ScreeningQuarantineError(
        "record quarantined", record_id="doc-42", reason="zip_bomb"
    )
    assert exc.record_id == "doc-42"
    assert exc.reason == "zip_bomb"
    # It is a typed disposition, catchable as the taxonomy root but NOT as a builtin surprise.
    with pytest.raises(LatenceError):
        raise exc
