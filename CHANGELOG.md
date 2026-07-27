# Changelog

All notable changes to the Latence Framework are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project (all publishable `latence-*` distributions) adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). All packages in the
workspace are versioned and released together under one version number.

## [Unreleased]

### Added

- **T7 — efficient per-chunk extraction: label-set bucketing + gliner2's native batched forward
  (`fused_entity_relation.gliner2`, `redaction.gliner2`, `latence_core.providers.perf`).** Per-chunk
  induction (T1/T2) gives every chunk its own small label set, but gliner2's batched entry points
  take ONE schema per call — so the naive readings were "one native call per chunk" (correct labels,
  no batching) or "union the batch's labels" (maximal width, but back to the ~40-label degradation
  and the ~17 GiB span-scoring tensor the redesign exists to remove). Both Providers now BUCKET each
  streaming window by label set through one shared, torch-free seam
  (`perf.iter_label_buckets`, first-seen bucket order, ascending indices) and run **one native
  batched forward per bucket** at that bucket's own small label count — maximal batch width, never
  the corpus union. The fused extractor was the un-batched half: it now drives
  `batch_extract_entities` / `batch_extract_relations` (bucketed independently per slot, since a
  chunk can share the entity schema and not the relation one), scatters per-text results back **by
  index**, and inherits the token-budget micro-batch bound plus the CUDA-OOM halving backoff its
  twin already had. Output is bucket- and batch-INVARIANT (asserted record-for-record against a
  strictly per-chunk reference). **This is the root fix for the CUDA-OOM**: the tensor was
  `batch × ~768 tokens × ~40 corpus-wide labels`; per-chunk labels collapse the label dimension, so
  it never forms — the reactive backoff stays only as a last-resort net. gliner2's native
  **`compile` is unlocked** as a consequence: it was forced OFF because a fresh (label-count, batch)
  shape family per forward recompile-thrashed `torch.compile(dynamic=True)`, and bucketing +
  micro-batching pin both dimensions, so it rides the shared `auto` default (True on cuda) again,
  behind the load-time guard AND the inference-time reload-uncompiled latch. `latence-extract-gliner2`
  raises its `gliner2` pin to `>=1.3,<2` (the batched slice does not exist across the whole 1.x
  line) and checks both batched methods at LOAD, so an out-of-range install is a typed
  `ProviderError`, never a native `AttributeError`. **Honest framing:** gliner2 ships no
  self-hostable inference server — its only server surface is `from_api`, a client for fastino's
  *cloud* API — so the throughput here is the native batched forward + compile + quantize,
  in-process. The eager-vs-compiled number and the peak-VRAM / backoff-count evidence are a GPU
  measurement: the harness is `scripts/validate-gliner2-unified-on-gpu.md` §7, unmeasured (never
  fabricated) until the pod run.

- **Blackwell / CUDA-13 compatibility matrix, machine-checked (#199, ADR-0016/0036/0030/0012).**
  `deploy/env/compatibility-matrix.yaml` maps every learned Provider and serving dependency of the
  enterprise-sota stack against every pinned environment (#195/#196/#198), and
  `docs/compatibility-matrix.md` is **generated** from it and byte-compared, so the published page
  cannot drift from the checked data. The rule the new `latence env matrix` command enforces is the
  reason the file exists: **every cell is a MEASUREMENT or explicitly `UNVERIFIED`**, with no third,
  softer status. A `verified` cell without date/hardware/command/observed output is
  `E-MATRIX-EVIDENCE-MISSING`; an `UNVERIFIED` cell without a *runnable* reproduce command is
  `E-MATRIX-REPRODUCE-MISSING` / `-UNRUNNABLE`; coverage is **derived from the committed
  `environment.yaml` files**, so a new enforced boundary or learned Provider is
  `E-MATRIX-CELL-UNCOVERED` until it is mapped; a recorded version that disagrees with that
  environment's committed lock is `E-MATRIX-VERSION-DRIFT`; a named served model without its
  weights AND code licences cited separately is `E-MATRIX-LICENCE-UNCITED` (ADR-0012); a committed
  lock that pins `nvidia-*` CUDA wheels `THIRD-PARTY-LICENSES.md` does not name, or whose path that
  register never cites, is `E-LICENCE-UNDISCLOSED` — the gate that relates an *environment* to the
  published licence page, so a second pinned GPU environment cannot ship disclosed only by the
  first one's carve-out. Each cell
  carries two independent axes — **resolution** (does it install: offline, checked on every commit)
  and **runtime** (does it run on `sm_120`: needs the hardware) — because a green resolution is not
  a runtime claim. The five failure modes of the live 519-page run are matrix entries with their
  resolutions and the file that enforces each (`FM-FLASHINFER-SM120` — the `sm75` check that
  misparses compute capability 12.0; `FM-NVRTC-JIT` — the TorchScript/nvrtc crash in the mdeberta
  encoder, #190; `FM-WHEEL-ABI` — the ABI-broken `flash-attn`/`torchcodec` wheels;
  `FM-MULTIMODAL-OOM` — the multimodal OOM at the generic serve settings; `FM-TRANSFORMERS-SPLIT`);
  dropping any one of them fails the check. `scripts/verify-local.sh` runs the verifier on every
  commit; `--require-verified` (the maintainer's switch ON the Blackwell host) additionally fails
  while anything is still unmeasured.

- **The one full-stack end-to-end verification (#199).** `deploy/env/full-stack/` composes the
  served-OCR environment and the GPU learned-Provider environment into a single pipeline — the only
  artifact where both legs run in one run — as one command:
  `bash deploy/env/full-stack/verify-full-stack.sh --corpus <scans> --record`. Four gates: both
  environments' offline spec gates, the gpu-learned smoke (GPU-placement probe included), the
  composed run, then `assert_full_stack.py`, which refuses every near-miss shape the project's own
  history has produced — a **skipped** Stage (correct on a CPU host, a false green for a GPU
  verification), a Stage that silently resolved to `cpu` (the #191 signature), an empty export,
  `document_count: 0` or `contracts_complete: false`. `--record` prints the evidence block to paste
  back into the matrix, so the measurement is transcribed by the machine that made it. **Status:
  UNVERIFIED** — it has not been run: no Blackwell/CUDA-13 host was available, and the matrix says
  so in the record rather than implying otherwise.

- **Go-live runbook + a machine-checked front door (#189, #72).** `docs/GO-LIVE-RUNBOOK.md`
  is the maintainer-manual launch procedure — visibility flip, docs deploy, signed release,
  PyPI publish, announcement — each step with the check that proves it landed, a rollback
  table, and the two decisions that must be made *before* step 1 (PyPI name availability,
  and the fact that flipping visibility publishes git **history**, not just the tree).
  Backing it, `test_front_door_truthfulness.py` now fails the build if the README claims a
  Runner adapter with no package behind it, quotes an ADR count that disagrees with
  `docs/adr/`, links at a file that does not exist, or tells a reader to install a
  distribution that is not in `packages/`. The PyPI publish was rehearsed end to end: all
  32 distributions build, `twine check` passes on all 64 artifacts, and `pip install
  latence-demo` into a clean venv runs the full demo to a green Quality Report.

- **Benchmark suites for Entity/Relation Extraction, Disambiguation and Redaction (#187).** The
  public benchmark harness (#181) now measures four more Stages through the same
  `BenchmarkSuite` seam, the same one command (`latence-benchmark run --suite all`) and the same
  published results page. Entity Extraction, Relation Extraction and Disambiguation run on
  **RED^FM** (`Babelscape/REDFM`, CC BY-SA 4.0 corpus + Apache-2.0 loader, verified separately
  per ADR-0012), pinned to a commit SHA, across all seven languages — so the P1 German/EU slice
  is one row of the table those suites emit. Their numbers are **not published yet** (RED^FM has
  to be downloaded first); `docs/running-the-benchmarks.md` declares them under *Not yet
  measured* with the command that produces them, and the results page carries no section for a
  Stage that has not been measured. Redaction runs on a **synthetic PII corpus that ships in
  the package**: no real personal data enters the repository, every value is a reserved or
  published example, the corpus is generated by a committed generator whose output is
  byte-checked, and it is pinned by **content digest** (`ShippedCorpusSource` refuses a corpus
  whose bytes no longer match its pin). Results gained a second reporting currency,
  `MetricScore` (TP/FP/FN → precision/recall/F1, with an undefined ratio reported as `n/a`
  rather than 0%), rendered on the same page next to the Parse pass-rate table. WikiANN/PAN-X
  was **excluded** because its dataset card carries `license: unknown` — recorded in
  `THIRD-PARTY-LICENSES.md` so the decision is not silently re-litigated.

- **GPU learned-Provider environment (#198, ADR-0016/0007/0030/0036/0045/0046).**
  `deploy/env/gpu-learned/` stands up the learned Provider set — GLiNER2 fused
  extraction, GLiNER2 financial-PII and learned GLiNER PII redaction, GLinker
  disambiguation, Granite r2 GPU-tier embeddings — pinned and reproducible on
  CUDA 13 / Blackwell (`torch==2.13.0+cu130`, the wheel line carrying `sm_120`
  kernels), from one documented `install.sh` with no trial-and-error and no
  `--break-system-packages` (PEP-668 safe). The `transformers` major split has
  exactly **one** supported resolution — the ADR-0030 endpoint boundary: the
  4.x GLiNER/mdeberta family in-process, the 5.x OCR VLMs served and reached
  over HTTP via `parser.endpoint`. A two-phase single-interpreter install was
  rejected as merely deferring the failure. The resolution is **enforced**:
  `latence env verify` now reports `E-CONSTRAINT-CONFLICT` when a package's
  declared version window is disjoint from an enforced boundary, and
  `E-PACKAGE-EXCLUDED` when an in-process OCR package enters the set directly or
  as another package's workspace dependency — both before `install.sh` runs, so
  the unsupported combination fails at dependency resolution rather than at model
  load.

- **Real GPU-placement proof (#198, closing the #191 class).** New
  `latence env placement` drives a real inference through every Provider an
  environment declares in `learned_providers` and reads torch's CUDA allocator
  around the call, reporting the allocation **delta above the Provider's resident
  footprint** (a warm-up drive plus a baseline subtraction, because real
  `reset_peak_memory_stats()` resets the peak to the *current* allocation — so
  weights parked on the GPU can never be counted as work). A forward pass that
  silently ran on the CPU allocates **zero CUDA
  bytes** above that baseline however correct its output is — which is exactly what #191 looked like
  on the pod (`device: cuda` configured, ~11.5 s/chunk, ~94 minutes for one
  Stage, the GPU idle, no warning). The assertion is behavioural and cannot be
  satisfied by a flag or a config echo; a missing CUDA device is a **failure**,
  not a skip. `latence env verify` additionally refuses a GPU environment whose
  smoke stack never instantiates a declared learned Provider
  (`E-LEARNED-UNDRIVEN`), since a Provider the gate never drives is a hole in the
  gate.

- **Optional Iceberg-table Export output (#184, ADR-0017).** The corpus Export can now
  *additionally* register the records it wrote as an Apache Iceberg table in an Iceberg **REST
  catalog** (Unity / Polaris / Glue-via-REST / Lakekeeper), so the same data becomes directly
  governable with no glue code. Enabled with one `iceberg:` config block on
  `export.jsonl_parquet`; **off by default**, and with it off `pyiceberg` is never imported and
  `records.jsonl` / `records.parquet` are byte-identical. The Iceberg schema *is* the Parquet
  schema (including opt-in signal columns) and carries the same contract version stamp; the
  registered data file is an immutable, content-addressed copy in the target table's own location
  (`<table location>/data/records-<sha256>.parquet`), so a snapshot never points at a path a later
  run rewrites — and never at one the framework's own Purge / run-retention cleanup deletes. The
  export directory itself is left byte-for-byte unchanged. Delta runs pick `mode: append` (a delta
  run's records, added as a snapshot) or `mode: overwrite` (a full re-export, replacing the
  contents in one atomic commit) — semantics documented in `docs/iceberg-export.md`. The client
  ships as the `latence-core[iceberg]` extra (pyiceberg, Apache-2.0), never as a core dependency.

- **OpenLineage run events (#185).** The Runner can emit standard OpenLineage
  `RunEvent`s — START/COMPLETE/FAIL per Stage plus a parent event for the run —
  with each Stage's input/output datasets and a `schema` facet from its carrier
  contract, so any OpenLineage consumer (Marquez, Atlan, Collibra, Dataplex) sees
  a latence run in its lineage graph with no custom integration. Off by default
  (`LocalRunner(lineage=...)` / `LATENCE_OPENLINEAGE_URL`), stdlib-only on the wire
  (no `openlineage-python` dependency), counts-only (no content, no PII, no secrets,
  and deliberately no `errorMessage` facet — only the typed error category), and an
  emission failure never fails a run. See `docs/observability.md`.

- **`latence-sink-qdrant`** — the first first-party **sink**: one command between "pipeline
  finished" and "my retrieval engine serves the corpus" (#183, the first slice of #156).
  `latence-sink-qdrant load ./out/export --collection corpus --url …` reads the Export's
  `records.parquet` (streamed in bounded batches), creates the collection if absent at the corpus's
  own vector width, and upserts with the `embedding` column as the vector and every remaining column
  as payload.
  - **Idempotent** — point ids are a deterministic UUIDv5 of the content-addressed `record_id` (a
    raw `record_id` string is not a legal Qdrant point id), and a stored content digest means an
    unchanged record is **not** re-upserted, so a re-run or an incremental/Delta run
    ([ADR-0018](docs/adr/0018-incremental-corpus-deltas.md)) writes only what changed.
  - **Retraction/Purge honoured** — records the Export no longer contains are deleted by a
    reconciling sweep, and the load **counts the stale set back to zero before reporting success**;
    `latence-sink-qdrant verify-absent <record_id>…` is the operator-side audit step (exit 1 while
    any named record is still there; `SinkVerificationError`/exit 2 rather than a false green when
    the named collection does not exist, so a typo can never pass for erasure evidence). `--no-reconcile` covers the incremental-Export case.
  - A store fault mid-load aborts **before** the sweep, so a half-uploaded corpus can never delete
    the records it did not reach.
  - The Export stays the source of truth ([ADR-0017](docs/adr/0017-ai-ready-deliverable-is-rag-corpus-plus-kg-as-files.md));
    the sink holds no index and runs no search ([ADR-0048](docs/adr/0048-retrieval-tooling-is-strictly-stateless.md)).
  - `latence-core` takes **no** new dependency: the mapping/idempotency/reconcile core is pure-Python
    over `latence-core`, and `qdrant-client` (Apache-2.0) rides the optional `[client]` extra,
    imported lazily. Ships a faithful `FakeQdrantClient` that rejects what a real Qdrant rejects
    (404/409/403/400 incl. non-UUID point ids and wrong-width vectors), plus an opt-in live smoke
    test (`LATENCE_QDRANT_URL`).

- **Reproducible environments (`deploy/env/`)** — a pinned lockfile, a one-command install recipe
  and a one-command smoke test per Provider-set slice, starting with `cpu-text` (GLiNER2 fused
  extraction, learned GLiNER PII redaction, GLinker disambiguation, Granite r2 dense embeddings,
  SPLADE sparse). The pins are **enforced constraints**: `latence env verify` cross-checks the
  generated `requirements.in`/`constraints.txt`, the solved `requirements.lock`, the Provider
  packages' real dependency declarations and the smoke stack, offline and import-free, and
  `scripts/verify-local.sh` runs it for every environment — so dependency drift that would break
  the stand-up turns the build red instead of a pod. New `latence env verify|render` commands and
  the optional `latence-core[envspec]` extra.

- **`latence-benchmark`** — the reproducible **public** benchmark harness (#181). One command
  (`latence-benchmark run --suite parse`) runs a framework Stage over a public dataset pinned to
  an exact commit SHA, under a pinned stack config, and publishes `docs/BENCHMARKS.md` plus a
  machine-readable `docs/benchmarks/results.json`. The Parse tracer slice scores
  `parser.pdfplumber` against [olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench)
  (ODC-BY-1.0 corpus, Apache-2.0 toolkit — verified separately per ADR-0012). Not a pipeline
  Provider: it registers nothing under `latence.providers`, and it is distinct from the
  adopter-side gold-set harness of ADR-0028.

  Structural guarantees, not conventions: a `DatasetSpec` rejects any revision that is not a full
  40-hex SHA; a partial download is refused rather than partially scored; the license gate runs
  *before* acquisition and treats an UNVERIFIED license as a non-overridable stop; a parse failure
  counts its document's tests as failures; unimplemented test classes are reported *unsupported*
  and excluded from the pass-rate denominator; vendor numbers are labelled claims and kept in
  their own section. `latence_benchmark.suite.BenchmarkSuite` is the seam further Stage suites
  (#187) plug into.

- **`latence-console`** — the read-only run console (#186). One command,
  `latence-console <storage-location>`, serves a localhost page over the files a run already
  left on Storage: the run list (status + timestamps) and each run's Quality Report rendered
  as **headline quality verdicts a non-engineer can read** — traceability, unsafe input,
  personal data, knowledge-graph statistics, gold-set scores, and the what-changed summary of
  an incremental run — with the per-Stage drill-down beneath. Read-only and stateless by
  construction (ADR-0010/0017): it reaches only `exists`/`ls`/`read_text`/`modified` on the
  Storage seam, refuses non-read HTTP methods with `405`, caches nothing between requests,
  binds the loopback interface only, and emits self-contained pages with no CDN, script, or
  beacon. Zero new dependencies — `http.server` from the standard library plus `latence-core`.

- **The run console's debugging loop (#188).** Three deep views on top of the console, reached
  from any run page. **Held back & flagged** lists every record Screening quarantined (checkpoint,
  detector, category, recorded reason, bytes withheld) and every record it kept but marked, with
  the risk marker that rode into the corpus. **The chunk inspector** answers "why is this chunk
  wrong": each chunk resolves back through its Provenance to its source document, source *page*
  and character offsets — computed with the framework's own `OffsetIndex`/`PageOffsetIndex` over
  the record's own maps, so the console cannot disagree with the pipeline — alongside its
  Classification and its Redaction disposition, including the case where the redaction policy ran
  *no* detectors (an empty PII list then says "the control was OFF", never "clean"). **The
  knowledge-graph browser** makes entities and edges navigable and shows **every edge with its
  Evidence** — the source mentions resolved to text, file, page, offsets and confidence; a cited
  mention the run no longer holds is shown as unresolved rather than dropped. All of it reads
  JSONL the Runner already wrote (`checkpoints/`, `quarantine/`, `screening/`) through the real
  contracts, so a malformed line is rejected exactly as the pipeline would reject it. Still
  strictly read-only, still no new persistent state, still zero new dependencies, still offline —
  and only the PII-handled corpus text is ever rendered.

- **`Storage.modified(uri)`** — a read-only last-modified probe on the Storage seam,
  normalising the per-backend spellings (`mtime` / `LastModified` / `updated` / `created`) to
  an aware UTC datetime and returning `None` rather than fabricating a time when a backend
  exposes none. This is how the run console reads real run timestamps without the framework
  persisting any new state.

### Changed

- **A chunk describes its OWN pages — `ChunkRecord.page_map` becomes `ChunkRecord.page_slice`
  and `PageIndexResolver` becomes STATELESS (`SCHEMA_VERSION` 18 → 19, BREAKING).** v18 kept the
  payload win (below) but bought it with a SHAPE defect: the whole document's `PageMap` rode ONE
  record of the chunk stream — the document's first chunk — and the resolver adopted it from
  whichever record it saw first. That made page resolution depend on two things that are not
  properties of the chunk being resolved: that the map-carrying record was **observed first**
  (every batching extractor resolves out of stream order) and that it **survived every filtering
  Stage** (any chunk→chunk Capability may drop chunk 0, and Screening quarantining a title page
  routinely did). Both failures degraded SILENTLY to the chunk's inherited page range — a
  plausible-looking WRONG page, indistinguishable from a correct one in the output. Four audit
  rounds patched the consequences (`rehome_page_maps()`, then an LRU retention window, then that
  window's own justification) and never the shape.
  `ChunkRecord.page_map: PageMap` is therefore **replaced** by `ChunkRecord.page_slice: PageSlice`
  — exactly the document page spans overlapping that chunk's own `[char_start, char_end)`, in the
  document's original coordinates. One span for a chunk inside a page, two for a chunk straddling
  a page break. A chunk never needed the whole document's map: the only question asked of it is
  which page a SUB-CHUNK offset falls on. Order- and drop-independence are now properties of the
  record rather than of the stream, so the failure mode is not mitigated but **inexpressible** —
  and ~99.8% of the v18 payload win is kept (~2 spans, ~60 bytes, against a ~30 KB whole-document
  map). Provenance is byte-identical (ADR-0021/0031).
  **Deleted, not deprecated:** `PageIndexResolver.observe()`, its per-document memoisation, the
  `max_live_pages` retention window and its `evicted_reentries`/`documents`/`unresolved`
  counters, `PageIndexResolver.from_documents()`, `rehome_page_maps()` and
  `DEFAULT_MAX_LIVE_DOCUMENTS` — every one of them was compensating mechanism for the old shape.
  `page_for(chunk, offset) -> int` and `pages_for_span(chunk, start, end) -> tuple[int, int]` no
  longer return `None`: a chunk carrying no spans raises `PageSliceMissingError` (ADR-0034 — a
  failure is observable or it is not handled), and `AdapterBase.resolve_pages` no longer degrades
  to the chunk's inherited range. A relation's **covering span** — whose endpoints may sit in
  different chunks, so no single chunk can answer for it — is now computed by the new
  `pages_for_covering_span(head, tail)`, arithmetic on the two endpoints' already-exact pages,
  replacing the "anchor chunk" lookup that only ever worked because the anchor carried the whole
  map. **Migration:** chunk checkpoints are run-scoped artifacts and a run is re-chunked from its
  Parse checkpoint, so there is no data migration to perform — the schema bump is the notice. A
  Provider that read `chunk.page_map` must go through `PageIndexResolver`; see ADR-0022's C1
  amendment and `docs/writing-an-adapter.md`. `DocumentRecord.page_map` is **unchanged** — the
  document genuinely has a whole-document map, and it is what the slice is cut from.
- **The document page map stops riding every chunk — one `PageIndexResolver` seam replaces five
  hand-rolled per-chunk indices (`SCHEMA_VERSION` 17 → 18).** `PageMap` is DOCUMENT-scoped state,
  invariant across every chunk of its document, yet `ChunkRecord` carried it by value on all N
  chunks: measured on a 519-page document chunked at 640 tokens that is ~30 KB of JSON on each of
  ~399 records — **~92% of every chunk checkpoint line**, ~12 MB redundant per document, paid again
  on every checkpoint write, every streaming read-back and every decode. The Chunk Stage now stamps
  `page_map` on a document's **first** chunk only, and offset→page resolution moves behind
  `latence_core.page_index.PageIndexResolver` — a narrow, document-keyed seam that memoises ONE
  `PageOffsetIndex` per `document_record_id`, owns the half-open-span end clamp, and aggregates
  drift corpus-wide (previously each throwaway per-chunk index carried its own diagnostics and
  nobody summed them). Five core call sites (`stages/entity`, `stages/header_refs`,
  `stages/inline_refs`, `stages/relation`, `redaction_policy`) re-derived that knowledge; exactly
  one of them cached, and only per chunk. Provenance is unchanged — offsets and pages are exact
  (ADR-0021/0031) — and the field stays optional, so a pre-v18 stream (map on every chunk) resolves
  identically: the resolver adopts whichever record carries the map first, or is seeded straight
  from the parsed `DocumentRecord`s (`PageIndexResolver.from_documents`, ADR-0017). Degradation is
  now COUNTED (`resolver.unresolved`) instead of a silent per-caller fallback.
  **`AdapterBase.resolve_pages` — the PUBLIC helper third-party adapters are pointed at — goes
  through that seam too (BREAKING, audit R2).** It signature changed from
  `resolve_pages(doc_char_start, doc_char_end, page_map=…, fallback_page_start=…,
  fallback_page_end=…)` to `resolve_pages(chunk, doc_char_start, doc_char_end, resolver=…)`: the
  old form built a throwaway `PageOffsetIndex` from `chunk.page_map`, which under v18 is `None`
  on every chunk after a document's first, so an adapter following `docs/writing-an-adapter.md`
  verbatim silently degraded those mentions to the parent chunk's inherited page range — the
  exact regression ADR-0021/0031 forbids, on the seam that is supposed to make the invariant the
  default. The interface got *smaller* (the two fallback arguments are now derived from the
  source chunk, which is what every caller passed), the guide documents the one-resolver-per-call
  + `observe()`-every-chunk discipline, and the reference adapter demonstrates it.
- **The benchmark guide no longer describes numbers the results page does not carry (audit
  R1-docs_truth-9, #187).** `docs/running-the-benchmarks.md` said in the present tense that the
  Entity/Relation Extraction, Disambiguation and Redaction slices were published on
  `docs/BENCHMARKS.md`, which carried the Parse suite and nothing else. The **redaction** number —
  the one that makes the framework's central privacy claim externally checkable, and the one suite
  that needs no download because its synthetic corpus ships in the package — is now measured and
  published (100.0% precision / 77.1% coverage recall over 35 gold items in 20 documents, with the
  per-language, per-PII-type and per-match-criterion tables). The three RED^FM suites are declared
  under an explicit **Not yet measured** heading with the command that produces them. The
  structural cause is fixed too: a run covering a subset of the suites now **republishes** the
  other Stages' published measurements instead of overwriting the page with its own suite
  (`--no-carry-forward` opts out), refuses to republish a number whose dataset has since been
  re-pinned, and every measurement records and renders the date it was produced.
- **Package license metadata is PEP 639 SPDX (#72).** Every publishable package moved from
  the legacy `license = { text = "Apache-2.0" }` + `License ::` classifier combination —
  which strict twine/PyPI validators reject — to a bare SPDX expression plus a
  `license-files` glob, and each distribution now carries its own `LICENSE`. Wheels emit
  `License-Expression: Apache-2.0` and `License-File: LICENSE`.
- **Front-door claims corrected before publishing (#189).** The README promised "thin
  adapters run the same Stages on Databricks or Airflow" when only `latence-runner-airflow`
  exists; it quoted 32 ADRs against 52 on disk; it advertised a `latence` meta-package that
  is not publishable. All three are fixed in the README and in the matching prose in
  `CONTEXT.md`, `docs/index.md` and `docs/concepts/`. The dangling `docs/INTEROP.md`
  references (a document that was never written) now point at
  `docs/RETRIEVAL-ENGINE-INPUT-SPEC.md`. `THIRD-PARTY-LICENSES.md`'s version snapshot was
  re-read from the installed environment.
- **Docs navigation is complete (#189).** ADR-0032 and ADR-0047 were missing from both the
  site nav and `docs/adr/index.md`, and `deployment.md` / `observability.md` /
  `secrets.md` were published but unreachable from the nav. Heading slugs now use the
  GitHub-compatible slugifier, so a hand-written `#anchor` resolves on github.com *and* on
  the docs site (this fixes four broken in-page links in `deployment.md`).

## [0.1.0] - 2026-07-14

First public, coherent release of the workspace. All publishable distributions
(`latence-core` plus the provider packages) are pinned to `0.1.0` and licensed
Apache-2.0.

### Added

- **The pipeline spine** — a model-agnostic, cloud-agnostic set of Stage contracts
  and a local Runner that turn messy, siloed source data into AI-ready outputs:
  Source → Parse/OCR → Chunk → Screening → Entity/Relation Extraction →
  Schema Induction → Redaction (PII) → Profiling → Disambiguation (entity linking /
  GLinker-style resolution) → Graph Assembly → Graph Completion → Context Enrichment →
  Export (Parquet · JSONL · TTL · GraphML). Every Stage declares typed input/output
  contracts and is fulfilled by a swappable Provider discovered via entry points.

- **`latence-core`** — data contracts, Capability protocols, the Pipeline + local
  Runner, the `latence` CLI (`stack validate`, `bake-off`), the Storage seam
  (`file://` and `s3://` via fsspec), and the near-zero-dependency built-in reference
  Providers (local-folder source, markdown/sentence chunkers, gazetteer entities,
  pattern relations, hybrid-rule redaction, statistical/lightweight profiling, cascade
  and exact-surface disambiguation, canonical/weighted graph assembly, reference graph
  completion, KG-header context enrichment, hashing embedder, JSONL/Parquet + KG export).

- **The provider ecosystem** — each heavy/endpoint model isolated behind a Stage seam in
  its own package so `latence-core` stays thin (ADR-0016):
  - _Parse / OCR_ — `latence-parser-plaintext`, `latence-parser-document`,
    `latence-parser-pdfplumber`, `latence-parser-render` (universal any-file, license-clean),
    `latence-parser-lighton`, `latence-parser-lighton-vllm`, `latence-parser-glm`,
    `latence-parser-endpoint`.
  - _Entities / Relations_ — `latence-ner-gliner`, `latence-ner-endpoint`,
    `latence-relation-gliner`, `latence-relation-llm`, `latence-extract-gliner2`,
    `latence-schema-inducer`.
  - _PII / Redaction_ — `latence-pii-gliner`, `latence-pii-gliner2`, `latence-pii-presidio`.
  - _Embedders_ — `latence-embedder-st`, `latence-embedder-endpoint`.
  - _Disambiguation_ — `latence-disambig-embedding`, `latence-disambig-glinker`.
  - _Graph completion_ — `latence-linkpred-ultra`.
  - _Runners_ — `latence-runner-airflow`.
  - _Demo_ — `latence-demo` (end-to-end showcase).

- **Enterprise-readiness gates** — a Provider Conformance suite parametrized over every
  registered Provider (contract, graceful failure, license declaration, determinism),
  the Goal-A end-to-end stack validation on a bundled corpus, and the Goal-B per-Stage
  bake-off matrices (quality/latency/memory/cost/license/determinism/error-rate).

- **Packaging & metadata** — every publishable package carries full `[project]` metadata
  (description, readme, `Apache-2.0` license, authors, keywords, PyPI classifiers marked
  `Development Status :: 4 - Beta` and `Typing :: Typed`, and Homepage/Repository/Issues/
  Changelog URLs).

### Engineering

- **CI** split into a fast required lane (`ci.yml`: secret-scan, lint+type fast gate,
  test matrix on py3.11/3.12 with surfaced coverage, Goal-A stack validation, all-package
  build) and a heavy lane (`ci-heavy.yml`: MinIO `s3://` e2e, real-Airflow DAG tests, and
  the Goal-B bake-offs) that also runs nightly. Actions are pinned by commit SHA with
  least-privilege `permissions`, per-ref `concurrency`, and docs-only `paths-ignore` on the
  heavy lane. `scripts/verify-local.sh` mirrors the gates for offline, deterministic local
  verification.

- **Release engineering** — `release.yml` builds all distributions and cuts a GitHub
  Release with generated notes on a `v*` tag; a PyPI Trusted-Publishing (OIDC) job is
  scaffolded, environment-gated, and disabled pending the maintainer's go-ahead
  (see `docs/RELEASING.md`).

[Unreleased]: https://github.com/ddickmann/latence-framework/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ddickmann/latence-framework/releases/tag/v0.1.0
