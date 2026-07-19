# R3 — on SharePoint Online (quickXorHash) every delta run downloads the library TWICE

- **Severity:** critical
- **Dimension:** connectors & ingestion / anti-false-green (the honest-witness test measures the
  one backend on which the claim is true)
- **Branch:** feat/release-fleet-integration
- **Tickets:** #178–#199 (release-fleet integration), #182 (the SharePoint Source connector)
- **Status:** fixed

## What is wrong

`SharePointSource.document_ids` — the `DocumentEnumerator` half of the Delta win — falls back to
`_content_document_id(self._client.download(item))` for **every** item whose listing carries no
reproducible `sha256Hash`. That is precisely the content fetch
`DeltaRunner._current_document_ids` forbids in its own docstring ("**This pass must not fetch
content.** It runs BEFORE the reuse exclusions are known, so anything it transfers is transferred
unconditionally"). `produce()` then evaluates the exclusion only against `listed_id`
(`if listed_id is not None and listed_id in self._excluded: continue`), so when the listing gave
no id the guard is skipped and the bytes are downloaded a **second** time before
`document_id in self._excluded` can be tested.

Graph reports `sha256Hash` only on **OneDrive Personal**. **SharePoint Online — the backend this
connector is named for — reports `quickXorHash`** (`graph.py:_HASH_FIELDS`, and the connector's
own `test_document_id_is_one_namespace_across_every_hash_facet_state`). So on the target product
the fallback is not the exception, it is the whole library, on every run.

The green suite cannot see it: `FakeFile.hash_algorithm` **defaults** to `"sha256Hash"`, so
`tests/test_sharepoint_delta_transfer.py` — written to be "the only honest witness" that the skip
holds end to end — exercises the one backend on which the claim is true and never the one the
connector targets. `test_a_drive_without_hashes_falls_back_to_hashing_the_downloaded_bytes`
asserts only id stability and concedes in a comment that it "cannot avoid the transfer", never
measuring the 2x cost.

Ticket #182's AC "Delta runs work: unchanged documents are skipped" is therefore unmet on the
target product; `README.md:16` states the headline unconditionally and `README.md:184` sends the
operator into a live smoke ("the second run must … transfer no document bytes") that fails
against a real SharePoint Online library.

## Failure scenario

The REAL `DeltaRunner` over one shared `FakeGraphTransport`, 4-document library, counting calls
to `DOWNLOAD_HOST` (source → parse → chunk → export):

| run | `hash_algorithm="quickXorHash"` (SharePoint Online) | `"sha256Hash"` (the fake's default) |
| --- | --- | --- |
| first run | **8** transfers for 4 documents (2x) | 4 |
| delta, nothing changed | **8** (reused=4, extracted=0) | 0 |
| delta, 1 of 4 changed | **8** (reused=3, extracted=1) | 1 |

A no-op incremental run over an unchanged SharePoint Online library moves **twice** the bytes of
a full run.

## What to build

1. **Discovery is listing-only.** `SharePointSource.document_ids` never downloads. An item the
   listing cannot content-address (`quickXorHash` / `sha1Hash` / no facet yet) is simply not
   enumerated — it is not in `current_doc_ids`, so the runner classifies it as new and `produce`
   fetches it exactly **once**. Under-reporting the enumeration is safe (it can only *lose* a
   reuse, never invent one); fetching in discovery is not.
2. **The accepted trade-off, stated openly.** On a hash-less backend a delta run costs **one**
   pass (never two), but those documents are re-extracted rather than reused. That is the
   documented "correct, just not free" fallback; the alternative (fetch in discovery to keep the
   extraction reuse) makes a no-op delta cost 2x a full run, which is the defect. Zero-transfer
   skipping still requires a reproducible `sha256Hash`.
3. **Honest `documents_extracted`.** `DeltaRunner` derived it as
   `len(current_doc_ids) - len(reusable_ids)`, which silently under-reports when a Source's
   enumeration is partial. Derive it from what the run actually produced:
   `len(current_manifest) - len(reusable_ids)`.
4. **Docs tell the truth.** `README.md` (headline + live-smoke step) and the provider module
   docstring qualify the zero-transfer skip to backends that report `sha256Hash`, and state the
   SharePoint Online cost explicitly.
5. **A non-vacuous regression test** on `hash_algorithm="quickXorHash"`, counting transfers
   through the real `DeltaRunner`.

## Acceptance criteria

- [x] A first run over a `quickXorHash` library transfers each document exactly **once**
      (4 documents → 4 transfers, not 8).
- [x] A delta run over an **unchanged** `quickXorHash` library transfers **no more than a full
      run** (4, not 8) — an incremental run never costs more than a full one.
- [x] A delta run with 1 of 4 changed transfers at most a full pass (4, not 8).
- [x] The `sha256Hash` path is untouched: first run 4, unchanged delta 0, one-changed delta 1
      (the existing `test_sharepoint_delta_transfer.py` assertions stay exactly as they are).
- [x] `documents_extracted` matches the documents actually re-extracted, whatever the Source's
      enumeration reports.
- [x] The regression test
      (`test_a_quickxorhash_library_never_costs_more_than_a_full_run`) **fails on the pre-fix
      code** (`assert 8 == 4`) and passes on the fix.
- [x] `make test` stays green at ≥ 2985 passed / 28 skipped.
