# R1 — the SharePoint connector's headline Delta win is false under the real DeltaRunner

- **Severity:** major
- **Dimension:** connectors & ingestion / anti-false-green (hollow test hiding a real cost)
- **Branch:** feat/release-fleet-integration
- **Tickets:** #178–#199 (release-fleet integration), #182 (the SharePoint Source connector)
- **Status:** fixed

## What is wrong

The connector's headline claim — "an unchanged document is skipped **without transferring a
byte**", the thing that makes an incremental run over a 50k-document library cheap
(`packages/latence-source-sharepoint/src/latence_source_sharepoint/provider.py:26-31`,
`packages/latence-source-sharepoint/README.md:99-103`) — did not hold end to end.

`DeltaRunner.apply_delta` must know which document ids the source currently holds before it can
intersect them with the parent Corpus Version and decide what is unchanged. It obtained that set
in `DeltaRunner._current_document_ids` (`delta_runner.py:526`) by instantiating each Source
Provider on the **unmodified** stage config and iterating `provider.produce(storage)` — and
`SharePointSource.produce` (`provider.py:236`) downloads every item's bytes in order to yield a
`ParserInput`. That discovery pass runs *before* `exclude_document_ids` is computed, so no
exclusion could ever prevent it: **every delta run re-downloaded the entire library**, and the
first run downloaded each document twice (once in discovery, once in the run).

The suite could not see it. `test_excluded_document_is_skipped_without_transferring_a_byte`
(`packages/latence-source-sharepoint/tests/test_sharepoint_source.py:180`) — the test whose name
asserts the claim — hand-injects `exclude_document_ids` straight into the Provider config,
bypassing the runner entirely. It proves the Provider *honours* an exclusion; it says nothing
about whether the runner ever gets to ask for one, or what the runner itself transfers first.

## Failure scenario

A 4-stage pipeline rooted at `source.sharepoint` (source → parse → chunk → export) over a
3-document fake drive, run twice through `DeltaRunner.apply_delta`, counting the
byte-transferring calls on one shared `FakeGraphTransport`:

- **Run 1 (first run):** 6 downloads for 3 documents — every document fetched TWICE (discovery
  pass + the run).
- **Run 2, nothing changed:** 3 more downloads — 100% of the library re-transferred, despite all
  3 documents being classified UNCHANGED and their records reused (`documents_reused == 3`,
  `documents_extracted == 0`).
- **Run 2 with one of three documents edited:** 4 more downloads — all three in discovery, plus
  the changed one again in the run. The "incremental" run transfers **more** bytes than a full
  run.

On the 50k-document library named at `provider.py:28-30` the documented saving is therefore
exactly zero, and the buyer-facing README promises it anyway.

## What to build

1. **A cheap-enumeration seam on the Source Capability.** `latence_core.capability` gains an
   OPTIONAL `DocumentEnumerator` protocol — `document_ids(storage) -> Iterator[str]` — for a
   Source that can name its documents more cheaply than it can fetch them. The ids MUST be the
   same content-addressed ids `produce` stamps on `Provenance.document_id`.
2. **`DeltaRunner._current_document_ids` uses it.** Prefer `document_ids` when the Provider
   implements it; fall back to `produce` otherwise (correct, just not free). Push the durable
   prior deletions into the Source's `exclude_document_ids` *before* the discovery pass, so even
   the fallback never fetches a document the run is going to drop.
3. **`SharePointSource.document_ids` answers from the listing.** Graph reports a content hash per
   `driveItem` in the listing, so the id is computable from metadata alone; only a hash-less
   drive still has to fetch (the same unavoidable fallback `produce` makes).
4. **A non-vacuous regression test** that drives the REAL `DeltaRunner` over a shared transport
   and counts byte transfers, rather than hand-injecting exclusions.
5. **Align the claims** in `provider.py` and the package README with the mechanism that now
   actually delivers them.

## Acceptance criteria

- [x] A delta run over an **unchanged** SharePoint library transfers **zero** bytes, measured on
      the transport through `DeltaRunner.apply_delta`, not through a hand-configured Provider.
- [x] A **first** run fetches each document exactly once (no discovery/run double-download).
- [x] A delta run over a library with 1 of 3 documents changed transfers exactly 1 document —
      strictly less than a full run.
- [x] The enumeration seam is optional: a Source that does not implement `DocumentEnumerator`
      (e.g. the in-core `source.local_folder`) still works via the `produce` fallback, with the
      prior deletions already excluded.
- [x] The regression test
      (`packages/latence-source-sharepoint/tests/test_sharepoint_delta_transfer.py`) **fails on
      the pre-fix code** (`assert 6 == 3` on the first run; `assert (10 - 6) == 1` on the
      incremental run) and **passes on the fix**.
- [x] `make test` stays green at ≥ 2985 passed / 28 skipped.
