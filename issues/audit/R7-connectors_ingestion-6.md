# R7-connectors_ingestion-6 — `GraphClient.list_files` recurses into folders with no cycle guard

**Severity:** major
**Area:** connectors / ingestion (`latence-source-sharepoint`)
**Location:** `packages/latence-source-sharepoint/src/latence_source_sharepoint/graph.py`
(`GraphClient.list_files`), contrast `GraphClient._paginate` twenty lines below.

## What to build

`list_files` walks the drive with an explicit `pending` stack, pushing a `/items/{id}/children`
URL for every entry that carries a `folder` facet — and it never remembers which folder ids it
has already walked. `_paginate`, immediately below it, *does* carry a `seen` guard for the
identical threat ("A server that echoed its own page as the nextLink would loop forever"), so
this is an omitted guard rather than an accepted design boundary. The module explicitly adopts
the hostile-response-body threat model (`require_same_origin` exists because "a hostile or
misdirected response body ... would replay the tenant's access token").

Give the folder recursion the same guard: remember every folder item id the walk has scheduled,
and refuse — with a typed `ProviderError`, never a hang and never a raw traceback (ADR-0034) —
when a listing names a folder that has already been visited. A real drive cannot legitimately
present the same item id twice: a Graph item id is opaque, unique and has exactly one parent.

## Failure scenario

A Graph-compatible endpoint (hostile, compromised, misdirected, or simply a misconfigured
`graph_base_url` pointed at a non-Graph proxy) returns, for the folder whose id is `FOLDER-1`,
a children page containing `{"id": "FOLDER-1", "name": "loop", "folder": {...}}`. `list_files`
pushes that same children URL back onto `pending` on every iteration and never terminates: the
client issues requests forever, and `pending` grows without bound alongside it, so the pipeline
both hangs and leaks memory (violating the O(batch) memory promise of ADR-0033 that the
docstring claims for this method). Reproduced against the REAL `GraphClient`: 20,000 requests
with no termination and no typed error, aborted only by a hard cap in the probe transport.

The same shape is reached by an entry carrying BOTH a `file` and a `folder` facet, because
`_is_folder` is tested before `_is_file`.

## Acceptance criteria

1. A children listing that names an already-visited folder id raises a typed `ProviderError`
   naming the repeating folder, instead of looping.
2. The walk stops at the cycle: the number of requests issued is bounded and small, not
   capped by an external limit.
3. The same holds for an entry that carries both a `file` and a `folder` facet.
4. A legitimate nested drive still lists every file, in the same order as before — the guard is
   inert on non-cyclic input (no behavioural change to any existing listing).
5. A non-vacuous regression test drives the real `GraphClient` and fails on the pre-fix code.
