# R2-connectors_ingestion-10 — SharePoint `document_id` is minted in two incompatible namespaces both labelled `sha256:`

**Severity:** major
**Area:** connectors / ingestion (#182 `latence-source-sharepoint`)
**Files:** `packages/latence-source-sharepoint/src/latence_source_sharepoint/provider.py`
(`produce`, `document_ids`, `_listed_document_id`),
`packages/latence-source-sharepoint/src/latence_source_sharepoint/testing.py`
**Invariants at stake:** ADR-0018 (delta guarantee), the module's own
"same id ⇒ byte-identical" contract, ADR-0034 (no silent errors).

## What is wrong

`SharePointSource` mints `Provenance.document_id` two different ways, and labels both `sha256:`:

* when Graph reports a `file.hashes` facet → `sha256("<algo>:<hash>|<size>")`;
* when it does not → `sha256(<raw bytes>)`.

These are different functions of the same document. Identical, unmodified bytes therefore get a
different id depending on which hash Graph happened to report at listing time — breaking the
module's stated invariant that "a document whose bytes did not change keeps the same id across
runs", and with it the ADR-0018 delta guarantee.

A second, quieter problem rides along: the hash-present branch inherits the strength of whatever
algorithm the backend reports. On SharePoint Online that is `quickXorHash`, a non-cryptographic
linear checksum — so "same id" was not in fact "byte-identical", as the docstring asserted.

## Failure scenario

Graph does not report `file.hashes` on every `driveItem`: the facet is computed asynchronously
after upload and is absent for some item/library states (the connector itself ships
`FakeFile(hash_algorithm=None)` as a supported drive state), and the algorithm differs by backend
(`quickXorHash` on SharePoint Online, `sha256Hash` on OneDrive Personal).

1. Run 1 lists an item whose hash facet has not materialised yet → `id = sha256(bytes)`.
2. Run 2 lists the same, untouched item, now carrying a `quickXorHash` →
   `id = sha256("quickxor:…|size")`.
3. The `DeltaRunner` intersects the enumerated ids with the parent Corpus Version, finds no
   match, and classifies an unchanged document as `DELETED` + `ADDED`. It is re-downloaded and
   re-extracted, the churn/drift metric spikes (potentially firing reconciliation), and the same
   flip across a library makes a no-op delta run cost **more** than a full run.

Proven: the same byte string `b"# unchanged bytes\n"` produced three distinct `document_id`s for
`hash_algorithm` `quickXorHash` / `None` / `sha256Hash`.

## What to build

One namespace, and only one: **`document_id` is always `"sha256:" + sha256(<raw bytes>)`** — the
same content address the reference `source.local_folder` stamps.

The delta win (skip an unchanged document without transferring a byte) is preserved wherever the
listing lets it be preserved *soundly*: Graph's `sha256Hash` **is** the SHA-256 of the file
content, so when the listing reports a well-formed `sha256Hash` the connector reads the id
straight off the listing — identical, by construction, to what hashing the bytes would produce.
Any other algorithm (`quickXorHash`, `sha1Hash`) is **not** a reproducible content address, so it
no longer mints an id at all: those items fall back to download-and-hash. Correct, just not free.
A follow-up may restore the fast path for `quickXorHash` by implementing Microsoft's published
QuickXorHash locally and verifying it against Microsoft's own test vectors — until that
verification exists it stays out (per the research-diligence rule: unverifiable ⇒ not shipped as
truth).

While the bytes are in hand after a listing-fast-path download, the connector verifies the
listing hash against them and raises a typed `ProviderError` on mismatch (ADR-0034), so
"same id ⇒ byte-identical" is now enforced rather than asserted.

## Acceptance criteria

1. For a fixed document body, `Provenance.document_id` is byte-for-byte identical whether Graph
   reports no hash, a `quickXorHash`, or a `sha256Hash`.
2. `document_ids()` (the `DocumentEnumerator` seam) yields exactly the ids `produce()` stamps,
   under all three of those drive states.
3. A document whose bytes change still gets a different id.
4. A `sha256Hash` drive still completes an unchanged delta run with **zero** bytes transferred
   (`tests/test_sharepoint_delta_transfer.py`, through the real `DeltaRunner`).
5. A listing hash that does not match the downloaded bytes surfaces as a typed error, never a
   silent wrong id.
6. `make test` stays green at or above the 2985 passed / 28 skipped baseline; no test weakened
   or deleted.

## Regression test

`packages/latence-source-sharepoint/tests/test_sharepoint_source.py`
::`test_document_id_is_one_namespace_across_every_hash_facet_state` (plus
`test_a_listing_hash_that_contradicts_the_bytes_is_a_typed_error`).
