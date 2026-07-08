"""Shared helpers to prove resume *equality* — not just a ``from_checkpoint`` flag.

H-E1 Part 1 (audit THEME E). The pre-existing resume tests asserted only
``all(m.from_checkpoint)`` (+ contract validity): a resume that dropped, duplicated,
or reordered a record passed them (the exact class of the P3-F1 sticky-resume bug).
These helpers capture the run's **output artifacts** — the exported corpus bytes
(``records.jsonl`` / ``records.parquet``) and every Stage checkpoint — so a test can
run to completion, resume the same ``run_id``, and assert the resumed output is
**byte-identical** to the first run's.

Why these artifacts and not the whole run tree:

* ``export/records.{jsonl,parquet}`` — the AI-ready output an adopter consumes; a
  corrupt resume shows up here as different/dropped/duplicated/reordered rows.
* ``checkpoints/*.jsonl`` — the newline-delimited record-set per Stage; byte order is
  the deterministic ``model_dump_json`` line order (the G1 fingerprint bar), so a
  legitimately-resumed run reproduces them byte-for-byte.

Deliberately EXCLUDED from the equality set (they legitimately differ across a
resume, and comparing them would make the test lie the other way):

* ``quality-report.json`` — carries per-Stage ``duration_seconds`` (wall-clock) and
  ``from_checkpoint`` flags that differ between the fresh and the resumed run.
* ``wal.log`` — an append-only log; the resume appends new ``resume``/``stage_batch``
  events, so it grows.
* ``manifest.json`` — written once on the first run and not rewritten, so it is stable,
  but it is run metadata, not output; excluded to keep the set "the output".

The record-set of a Stage is also exposed (``checkpoint_record_lines``) so a test can
assert *record-set* identity where byte order is legitimately nondeterministic — here
it is not (deterministic dumps), so byte-identity is the stronger, correct assertion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from latence_core.storage import Storage


def run_base(store: str, run_id: str) -> str:
    """The run-scoped artifact root ``<store>/_latence/runs/<run_id>``."""
    return f"{store}/_latence/runs/{run_id}"


def output_artifacts(storage: Storage, store: str, run_id: str) -> dict[str, bytes]:
    """Every output artifact of a run, keyed by its path relative to the run root.

    The map is the exported corpus (``records.jsonl`` + ``records.parquet``) plus every
    Stage checkpoint (``checkpoints/*.jsonl``) — the bytes that MUST be identical after a
    correct resume. Excludes ``quality-report.json`` / ``wal.log`` / ``manifest.json``
    (see module docstring: they legitimately differ or are not "output").

    Keying by relative path (not absolute) lets two runs under different store roots be
    compared directly — used by the determinism check as well as the resume check.
    """
    base = run_base(store, run_id)
    # ``walk_files`` now yields scheme-BEARING URIs on every backend (file://, memory://, s3://),
    # so a result can be fed straight back into ``read_bytes``. To take the run-relative path we
    # normalise both the walked URI and ``base`` (which also carries the scheme) to a bare,
    # leading-slash-free key, so a run under a ``file://`` root and one under an ``s3://`` root
    # compare like with like. The leading-slash strip matters: ``s3fs`` yields bare ``bucket/key``
    # but ``memory``/some object stores normalise to an absolute ``/bucket/key`` — without it the
    # prefix would never match on those backends and nothing would be captured (a silent false pass).
    prefix = _rel_key(base) + "/"
    captured: dict[str, bytes] = {}
    for uri in storage.walk_files(base):
        bare = _rel_key(uri)
        rel = bare[len(prefix) :] if bare.startswith(prefix) else bare
        if rel.startswith("checkpoints/") or rel.startswith("export/"):
            captured[rel] = storage.read_bytes(uri)
    return captured


def _strip_scheme(uri: str) -> str:
    """Drop a leading ``<scheme>://`` so local (bare) and object-store (scheme-bearing) paths
    compare like with like."""
    return uri.split("://", 1)[1] if "://" in uri else uri


def _rel_key(uri: str) -> str:
    """A scheme- and leading-slash-free key for prefix/relative-path comparison.

    Different fsspec backends disagree on whether a bare path keeps a leading slash
    (``s3fs`` -> ``bucket/key``; ``memory`` -> ``/bucket/key``). Normalising both sides here makes
    the run-relative keying backend-agnostic.
    """
    return _strip_scheme(uri).lstrip("/")


def checkpoint_record_lines(storage: Storage, store: str, run_id: str, stage: str) -> list[str]:
    """The newline-delimited record lines of a Stage's checkpoint (its record-set).

    Each line is one ``model_dump_json`` record. Returned as a list (order-preserving) so a
    caller can assert either byte/line identity or, where order were legitimately
    nondeterministic, ``sorted(...)`` set identity.
    """
    ckpt = f"{run_base(store, run_id)}/checkpoints/{stage}.jsonl"
    return [line for line in storage.read_text(ckpt).splitlines() if line.strip()]
