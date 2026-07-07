# H-D1 — GDPR Purge integrity: no residue, no idempotency break

**Source: hostile 360 audit, THEME D. Security S2 (residue) + correctness S3 (idempotency), both traced/reproduced.**
The delta engine's Purge is a headline enterprise/GDPR claim ("nothing derived survives", real
physical erasure — S11/ADR-0018). The audit found two independent holes in
`packages/latence-core/src/latence_core/state/corpus_store.py::expunge_records` (and its WAL
interaction). Both must close, each with a regression test.

## Gap 1 — Abandoned WAL transactions leave recoverable plaintext (security S2)
`expunge_records` sweeps only `self._wal.committed_versions()` — versions with a `MANIFEST.json`.
But the WAL by design leaves a `wal/v<N>/` **staging directory** with a fully-staged
`records.jsonl` and **no manifest** after a crash mid-commit (an *abandoned* transaction;
`WriteAheadLog.recover()` reports these in `.abandoned` and never deletes them). Those staged
`records.jsonl` files can contain a purged document's canonical entities/relations/Evidence in
**cleartext**, sitting at `_latence/corpus/wal/v<N>/records.jsonl` — invisible to `read_records`
(so the corpus *looks* erased) but recoverable by anyone with filesystem/backup access. That is
exactly the GDPR-erasure threat model, and it violates the S11 "nothing derived survives" criterion.

**Fix:** Purge must also physically neutralise abandoned staging dirs. Two acceptable shapes
(pick the simpler that's provably complete):
- **(a)** Expunge each abandoned version's staged `records.jsonl` the same way as committed ones
  (rewrite atomically, dropping every record whose `_record_reach` touches a purged doc); **or**
- **(b)** Physically delete the entire abandoned `wal/v<N>/` staging directory during Purge
  (they are incomplete, never-readable transactions — GC'ing them is also correct hygiene), after
  confirming none is the in-flight transaction of a *concurrent* run.
Prefer (a) if a bare directory delete could race a live recovery; document the choice. Either way,
after Purge **no** `records.jsonl` anywhere under the corpus root — committed or abandoned — may
contain a record reaching a purged document.

## Gap 2 — `expunge_records` leaves stale `version.json` → idempotency break (correctness S3)
`expunge_records` rewrites each version's `records.jsonl` but **not** its `version.json`, so the
stored `CorpusVersion.fingerprint` and `record_count` still describe the *pre-purge* record set.
The docstring calls this intentional ("erased, not re-derived"), but the auditor reproduced the
consequence: after a Purge, re-running the **same** delta over the identical post-purge corpus hits
`commit()`'s idempotency check (`current.fingerprint == fingerprint`), which now compares the new
live set's *real* fingerprint against the head's **stale** stored fingerprint → mismatch → a
**spurious duplicate Version** is minted instead of the documented no-op. Audit reads of
`record_count` also lie.

**Fix:** After rewriting a version's `records.jsonl`, **re-fingerprint and re-count** that
version's `version.json` to match its post-purge records (erasure already trumps immutability —
the manifest must reflect the erased reality, not the pre-erasure one). At minimum the **head**
version (whose fingerprint drives the idempotency check) must be corrected; correcting every
rewritten version keeps all audit metadata consistent with its records — do that unless there's a
concrete reason not to, and state it. Reuse the existing `_fingerprint(records)` helper so the
recomputed value matches what `commit()` computes for an identical live set.

## Definition of Done (Gate bar — security + correctness)
- **No residue:** a regression that (i) stages an abandoned transaction whose records reach doc D
  (crash mid-commit), (ii) commits a head version also deriving from D, (iii) `purge(D)`, then
  **walks every `records.jsonl` under the corpus root** (committed AND abandoned) and asserts no
  record reaches D, and that D's secret surface strings appear in **no** file on disk.
- **Idempotent after purge:** a regression that commits, purges D, then re-applies the identical
  delta over the post-purge live set and asserts `commit()` returns the **existing head Version**
  (a no-op), not a new duplicate; and that `read_version(n).record_count` / `.fingerprint` match
  the true post-purge records for every rewritten version.
- Purge remains **no-collateral** (the existing no-collateral test still passes — erasing D must
  not drop unrelated records) and atomic (each rewrite via `write_atomic`; WAL crash-safety intact).
- Concurrency note: document/guard the abandoned-dir handling against a concurrent run's in-flight
  transaction (don't delete a staging dir a live run is mid-commit on).
- `mypy --strict` clean over src+tests; `ruff` clean; deterministic seeded tests; CI green.
- Correct the `expunge_records` docstring (it currently states the stale-metadata behaviour as
  intentional) and add an ADR-0018/0029 note reconciling "immutable audit history" with
  "erasure re-writes the manifest to the erased reality."

## Reviewer should check
- The residue test genuinely walks the **whole** corpus tree (not just committed versions) and
  would FAIL against `main` (the abandoned `records.jsonl` currently survives).
- The idempotency test would FAIL against `main` (a duplicate Version is currently minted).
- No-collateral + crash-recovery (WAL replay) still hold — the fix doesn't tear resume/recovery.
- The abandoned-dir handling can't race or corrupt a legitimately concurrent run's transaction.
- Re-fingerprint uses the same `_fingerprint` as `commit()` so the idempotency compare is exact.
