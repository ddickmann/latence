"""EndpointEmbedder: the Embedder performance path over an OpenAI-compatible endpoint (S12).

The real ``openai`` client is intentionally NOT installed (ADR-0007/0016: offline, deterministic).
A fake ``openai`` module returns canned embedding responses so the Provider's batching, input-order
preservation, dimension validation, retry accounting, and lease handling are exercised offline.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from latence_core.endpoint import EndpointError
from latence_embedder_endpoint import EndpointEmbedder
from latence_embedder_endpoint.provider import _POOL


class _FakeEmbeddings:
    dimension: int = 4
    error: Exception | None = None
    fail_times: int = 0
    calls: int = 0
    shuffle: bool = False

    def create(self, *, model: str, input: list[str]) -> Any:  # noqa: A002 - openai API name
        type(self).calls += 1
        if type(self).error is not None and type(self).calls <= type(self).fail_times:
            raise type(self).error
        data = []
        for i, text in enumerate(input):
            # A deterministic vector keyed by text length so order is observable.
            vec = [float(len(text))] * type(self).dimension
            data.append(types.SimpleNamespace(index=i, embedding=vec))
        if type(self).shuffle:
            data = list(reversed(data))  # endpoint may return out of order; Provider must re-sort.
        return types.SimpleNamespace(data=data)


class _FakeOpenAI:
    def __init__(self, **kwargs: Any) -> None:
        self.embeddings = _FakeEmbeddings()


class _StatusError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.fixture()
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("openai")
    module.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    _FakeEmbeddings.dimension = 4
    _FakeEmbeddings.error = None
    _FakeEmbeddings.fail_times = 0
    _FakeEmbeddings.calls = 0
    _FakeEmbeddings.shuffle = False
    _POOL._leases.clear()  # noqa: SLF001


def test_requires_positive_dimension() -> None:
    with pytest.raises(ValueError, match="dimension must be >= 1"):
        EndpointEmbedder({"dimension": 0})


def test_embeds_in_input_order(fake_openai: None) -> None:
    provider = EndpointEmbedder({"dimension": 4, "base_url": "http://vllm/v1"})
    out = list(provider.embed(["a", "bb", "ccc"]))
    assert [v[0] for v in out] == [1.0, 2.0, 3.0]  # keyed by text length, preserved order
    assert all(len(v) == 4 for v in out)


def test_reorders_out_of_order_endpoint_response(fake_openai: None) -> None:
    _FakeEmbeddings.shuffle = True
    provider = EndpointEmbedder({"dimension": 4})
    out = list(provider.embed(["a", "bb", "ccc"]))
    # Even though the endpoint returned them reversed, index-sorting restores input order.
    assert [v[0] for v in out] == [1.0, 2.0, 3.0]


def test_batches_bound_requests(fake_openai: None) -> None:
    provider = EndpointEmbedder({"dimension": 4, "batch_size": 2})
    out = list(provider.embed(["a", "b", "c", "d", "e"]))
    assert len(out) == 5
    assert _FakeEmbeddings.calls == 3  # ceil(5/2)


def test_dimension_mismatch_surfaces(fake_openai: None) -> None:
    provider = EndpointEmbedder({"dimension": 8})  # endpoint returns dim 4
    with pytest.raises(EndpointError, match="configured dimension is 8"):
        list(provider.embed(["a"]))


def test_transient_failure_retried_and_counted(fake_openai: None) -> None:
    _FakeEmbeddings.error = _StatusError("overloaded", status_code=503)
    _FakeEmbeddings.fail_times = 2
    provider = EndpointEmbedder({"dimension": 4, "retry_attempts": 5})
    out = list(provider.embed(["a"]))
    assert len(out) == 1
    assert provider.endpoint_retries == 2


def test_lease_released_after_embed(fake_openai: None) -> None:
    provider = EndpointEmbedder({"dimension": 4, "base_url": "http://vllm/v1"})
    list(provider.embed(["a", "b"]))
    assert provider._lease.refcount == 0  # noqa: SLF001
    assert provider._lease.live is False  # noqa: SLF001


def test_empty_input_makes_no_call(fake_openai: None) -> None:
    provider = EndpointEmbedder({"dimension": 4})
    assert list(provider.embed([])) == []
    assert _FakeEmbeddings.calls == 0


def test_missing_openai_dependency_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)
    _POOL._leases.clear()  # noqa: SLF001
    provider = EndpointEmbedder({"dimension": 4})
    with pytest.raises(ImportError, match="pip install latence-embedder-endpoint"):
        list(provider.embed(["a"]))
