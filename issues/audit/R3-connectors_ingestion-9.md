# R3-connectors_ingestion-9 — SharePoint `list_files` never sorts files, so the exported corpus is not byte-reproducible

**Severity:** major
**Area:** connectors / ingestion — `packages/latence-source-sharepoint`
**Invariant breached:** DETERMINISM (`ProviderProfile.deterministic=True` must be true)

## What is wrong

`graph.py`'s module docstring asserts "children of a folder are sorted by item id before being
walked, so two runs over an unchanged drive produce identical documents in an identical order";
`GraphClient.list_files`'s docstring repeats it; `SharePointSource.profile()` declares
`deterministic=True` and its docstring says "the listing is walked in sorted order".

No sort was applied to files anywhere. The sole `sorted()` in the module ordered the FOLDER
recursion stack (by drive path, not by item id). Files were yielded in whatever order
`_paginate` returned them from Graph. Microsoft Graph does not guarantee an ordering for a
`driveItem` children collection absent `$orderby`, and the connector never sends one.

The offline suite was structurally blind: `FakeGraphApi._children_of` sorted its own output
(`sorted(self.files, key=lambda f: f.path)`), so `test_traversal_is_deterministic_across_runs`
and `test_pagination_follows_odata_next_link_across_every_page` could not fail — a fake strictly
stricter than the real dependency, proving a property real Graph does not deliver.

## Failure scenario

* **Probe A (client level):** subclass `FakeGraphApi` so `_children_of` returns the same children
  reversed — a legal Graph response, since ordering is unspecified. `list_files()` then yields
  `['d.md','c.md','b.md','a.md']` instead of `['a.md','b.md','c.md','d.md']`; the claimed sort
  never runs.
* **Probe B (output level):** run the real `LocalRunner`
  (`source.sharepoint -> parser.plaintext -> chunk.markdown -> export.jsonl_parquet`) over the
  same 4-document drive under `PYTHONHASHSEED=0`, once with each ordering: `corpus.jsonl` comes
  out byte-different (SHA-256 `3fc7e272…` vs `e8cfbe2e…`) for an identical, unchanged library.

Any adopter diffing exports, or any downstream consumer relying on the declared
`deterministic=True`, sees spurious churn on a drive nobody touched.

## What to build

1. `GraphClient.list_files` must impose the ordering it documents: buffer each folder's accepted
   files and yield them sorted by Graph item id (opaque and stable per item), independent of the
   order Graph returned them in. Memory stays bounded by one folder's file metadata — the folder
   recursion stack already carries a comparable per-folder buffer, and no document bytes are held
   (ADR-0033 unaffected).
2. Make the folder recursion order match its own docstring, or make the docstring match the code.
3. De-blind the fake: `FakeGraphApi` must return each folder's children in a deliberately
   non-sorted order, so every offline test exercises the client's real sort rather than the
   fake's.

## Acceptance criteria

* A regression test drives `list_files` against a Graph double that returns children in an order
  other than sorted, and asserts the yielded item ids/names are in ascending item-id order. It
  fails on the pre-fix code.
* The end-to-end determinism test still passes, and now has teeth: the fake no longer pre-sorts.
* `SharePointSource.profile().deterministic is True` is now a true statement about the code.
* Docstrings in `graph.py` (module + `list_files`) and `provider.py` (`profile`) describe the
  ordering the code actually implements.
* `make test` stays green at or above 2985 passed / 28 skipped.
