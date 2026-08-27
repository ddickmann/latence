# Agent briefing

**You are a coding agent. This page is written for you.** It is the structured entry point to the
Latence Framework: what it does, how it is organised, which models are available at each Stage, and
the exact commands that take a user from a folder of documents to a retrievable corpus and knowledge
graph.

Read this top to bottom once. It is ordered so that each section only depends on the ones above it.

Two ground rules, both load-bearing:

1. **The code is the authority.** Every fact below was read out of a file at the path named, and the
   paths are cited so you can re-verify rather than trust. Where this page and the code disagree,
   the code is right and this page is a bug — fix it in the same change.
2. **Nothing degrades silently.** A missing Provider, an unserveable retrieval leg, a Stage that
   cannot reach its device — each is a loud typed error naming what to do, never a quiet fallback.
   If you are handling an exception to "make it work", you are almost certainly defeating a
   guarantee. Read the error; it names the fix.

> **Changing this repository** rather than using the framework? Stop here and read
> [`AGENTS.md`](https://github.com/ddickmann/latence/blob/main/AGENTS.md) instead — the gates, the conformance rules, and the specific things
> that are expensive to get wrong when editing this codebase.

---

## 1 · What the framework does

Point it at messy documents. Get back two portable deliverables, as plain files:

- a **RAG corpus** — one row per chunk, PII-masked, embedded, carrying provenance to the exact
  character offset and source page;
- a **knowledge graph** — resolved canonical entities and the relations between them, every edge
  carrying the sentence it came from.

```
corpus folder ──► latence setup ──► latence.stack.yaml ──► latence process
                  (scans the folder,   (a Pipeline: DAG of      (the local Runner)
                   emits a validated    Stages, Pydantic-             │
                   config)              validated)                    ▼
                                          <storage_uri>/_latence/runs/<run_id>/
                                            export/corpus.{jsonl,parquet}       ← the RAG corpus
                                            export/graph-{nodes,edges}.parquet  ← the KG
                                            export/graph.{ttl,graphml}
                                            quality-report.{json,md}
                                                                      │
                                                                      ▼
                                      latence-sink-qdrant load  ·  latence-retrieval (query time)
```

---

## 2 · The five-minute path

From a clean clone. CPU-only, offline, no credentials.

```bash
uv sync                    # the CPU-first dev environment
uv run latence-demo        # end-to-end on a bundled messy corpus; exits non-zero on drift
```

Then on the user's own documents:

```bash
# 1. Generate a validated config. --non-interactive uses the same defaults the prompts offer,
#    so identical inputs produce a byte-identical config.
uv run latence setup --non-interactive --source ./their-documents

# 2. Lint it offline — sub-second, no corpus read, no model load. Run this every time.
uv run latence stack check latence.stack.yaml

# 3. Execute.
uv run latence process --config latence.stack.yaml
```

`latence setup` writes `latence.stack.yaml` in the working directory (`-o` to change it). It
dry-validates before saving: Pydantic, then DAG, then capability wiring, then the phase boundary.
Any failure exits 1 and writes nothing.

**Zero-network variant.** The default config wires an LLM for schema induction. To avoid that
entirely, either mine a type space from the corpus first —

```bash
uv run latence-label-profile ./their-documents --candidates-out candidates.json
uv run latence setup --non-interactive --source ./their-documents --label-candidates candidates.json
```

— which wires the deterministic `label_inducer.lexicon`, or pass `--predefined-labels schema.json`
to emit no induction Stage at all. The two flags are mutually exclusive and rejected together.

---

## 3 · The mental model

Five nouns. Use them exactly; [`CONTEXT.md`](https://github.com/ddickmann/latence/blob/main/CONTEXT.md) is the canonical glossary and lists the
synonyms this project bans.

| Noun | What it is |
|---|---|
| **Capability** | A narrow `typing.Protocol` — `Parser`, `EntityExtractor`, `PIIDetector`. The core names Capabilities and *never* names a model. |
| **Provider** | A concrete thing that fulfils a Capability: an in-process model, a served endpoint, a cloud service. Registered as a plugin via an entry point. |
| **Stage** | A typed transform over records, depending only on a Capability. |
| **Pipeline** | A DAG of Stages over a set of documents, declared once in YAML. |
| **Runner** | Executes a Pipeline. The built-in local Runner checkpoints and resumes; `latence-runner-airflow` runs the identical Stages as a DAG. |

Two consequences worth internalising, because they determine what a change costs:

- **Swapping a model is a config key, not a refactor.** Providers satisfy Protocols structurally, so
  a new Provider needs no core edit and no import in core.
- **Heavy dependencies live in the Provider package.** Installing the framework does not pull torch.
  `latence-core` stays importable on a laptop with no GPU and no network.

The entry-point name encodes the wiring: the prefix before the first `.` is the Capability.
`entity.gliner` fulfils `EntityExtractor`; `redaction.presidio` fulfils `PIIDetector`.

---

## 4 · Which models are available, per Stage

★ marks the CPU-first reference default — deterministic, no weights, no network. Every Stage has
one, so the whole spine runs offline before any model is chosen.

**Licences are declared per Provider, weights and code separately.** Check `ProviderProfile` before
recommending anything. One Provider in the catalogue is not licence-cleared: see the warning at the
end of this section.

### Ingest

| Capability | Providers |
|---|---|
| `source` | ★ `source.local_folder` · `source.sharepoint` (MS Graph; env-var credentials only, inline secrets rejected) |
| `screening.intake` | ★ `screening.intake_signature` — magic-byte, size and zip-ratio checks **before** parsing |
| `parse` | ★ `parser.pdfplumber` (MIT) · `parser.plaintext` · `parser.document` (pypdf) · `parser.render` (universal: office + images, delegates page OCR) · `parser.lighton` (GPU, `LightOnOCR-1B`) · `parser.glm` (GPU, `GLM-OCR`, MIT) · `parser.lighton_vllm` (served vLLM) · `parser.endpoint` |
| `chunk` | ★ `chunk.markdown` (offset- and page-preserving) · `chunk.sentence_window` · `chunk.page` (one chunk per measured page, 1024-token fallback) |
| `screening.content` | ★ `screening.content_keyword` · `screening.content_fuzzy` — prompt-injection and harmful-content guardrails |

### Extraction and privacy

| Capability | Providers |
|---|---|
| `label_inducer` | ★ `label_inducer.lexicon` (deterministic, offline, no cost) · `label_inducer.llm` (any OpenAI-compatible endpoint) |
| `entity` | ★ `entity.gazetteer` · `entity.gliner` (`urchade/gliner_multi-v2.1`) · `entity.endpoint` |
| `relation` | ★ `relation.pattern` · `relation.gliner_relex` · `relation.llm` |
| `fused_entity_relation` | ★ `fused_entity_relation.header_refs` · `…inline_refs` (deterministic cross-document references) · `…gliner2` (`fastino/gliner2-multi-v1`) · `…gliner25` (`fastino/gliner2.5-multi-v1`, 4096-token window) |
| `redaction` | ★ `redaction.hybrid_rule` · `redaction.gliner2` (`GLiNER2-Guardrails-PII-Multi`) · `redaction.gliner25` · `redaction.gliner_pii` · `redaction.presidio` (MIT) |

> **The GLiNER major boundary.** `gliner2>=1.3,<2` and `gliner2>=2.0,<3` are disjoint pins — a mixed
> v1/v2.5 extract-and-redact pair **cannot be installed**. This is why `latence setup` has a single
> `--engine gliner2|gliner25` answer that selects the extractor, redactor, chunker, chunk budget and
> window ceiling as one row. Do not hand-assemble a mixed pair; it will fail at resolve time.
>
> `--engine gliner25` also turns guardrails **off** and its PII is zero-shot and **unmeasured** —
> no GLiNER 2.5 PII checkpoint exists. The CLI prints this; do not suppress it.

### Resolution, graph, export

| Capability | Providers |
|---|---|
| `profiling` | ★ `profiling.statistical` · `profiling.lightweight` (skips the O(n²) co-occurrence pass) |
| `type_consolidation` | ★ `type_consolidation.cascade` · `type_consolidation.exact_surface` |
| `disambiguation` | ★ `disambiguation.cascade` (5-strategy resolver, audited merge log) · `…exact_surface` · `…embedding` · `…glinker` (GPU, `gliner-linker-large-v1.0`) |
| `graph` | ★ `graph.canonical` (content-addressed, per-edge Evidence) · `graph.weighted` |
| `graph_completion` | ★ `graph_completion.reference` (bounded 2-hop closure) · `graph_completion.ultra` (`ultra_3g`, MIT) |
| `context` | ★ `context.kg_header` — projects the KG back onto each chunk, embedding input only |
| `embedding` | ★ `embedding.hashing` · `embedding.sentence_transformers` (`granite-embedding-311m-multilingual-r2`; 97m on CPU) · `embedding.endpoint` |
| `export` | ★ `export.jsonl_parquet` · `export.knowledge_graph` · `export.demo_site` |

### Signal seams

These are **not** Stages. They are sub-Provider slots nested inside an Export's or Graph Assembly's
`config`, and they are additive — each one adds columns without changing the default schema.

| Seam | Providers | Nested in |
|---|---|---|
| `sparse` | ★ `sparse.hashing` · `sparse.splade` | Export `config.sparse_embedder` |
| `multivector` | ★ `multivector.hashing` | Export `config.multivector_embedder` |
| `fde` | ★ `fde.reference` · `fde.muvera` | Export `config.fde_converter` |
| `tokenizer` | ★ `tokenizer.regex` · `tokenizer.whitespace` | Export `config.bm25` |
| `graph_features` | ★ `graph_features.pagerank` · `graph_features.degree` | Graph Assembly `config.graph_features` |
| `delta` | ★ `delta.affected_set` | the delta Runner |

> **`sparse.splade` is the one Provider that is not licence-cleared.** It declares
> `license_verified=False`, `license="UNVERIFIED"`, and pins no default checkpoint. `latence
> retrieval splade-index` refuses to run without an explicit `--accept-noncommercial-license` and an
> operator-chosen model. **Never wire it silently on a user's behalf** — surface the licence
> question to them.

The generated, always-current version of this catalogue is
[`docs/reference/stage-contracts.md`](reference/stage-contracts.md), regenerated by
`latence contracts --write`. When in doubt, run `latence contracts --capability parse` and read the
JSON rather than trusting this table.

---

## 5 · The Pipeline YAML contract

The schema authority is the Pydantic model in
`packages/latence-core/src/latence_core/pipeline.py`. Both `Pipeline` and `Stage` are
`extra="forbid"` — **any unknown key is a hard error**, which means a typo fails loudly at
`stack check` rather than being ignored.

| Top-level key | Required | Default | Notes |
|---|---|---|---|
| `name` | **yes** | — | |
| `storage_uri` | **yes** | — | Base URI for state and exports. `file://`, `s3://`, `gs://`, `abfs://` |
| `stages` | **yes** | — | ≥ 1; order in the list is irrelevant, `depends_on` is the DAG |
| `schema_version` | no | `2` | |
| `gold_set_uri` | no | `null` | Annotations → a goldset section in the Quality Report |
| `extract_on` | no | `unmasked` | `masked` makes extraction read redacted text; validated to require a reachable redaction Stage |
| `auto_generate_labels` | no | `false` | **Documentation marker only** — flips no behaviour |

Each Stage takes `name` (unique), `capability`, `provider`, and optionally `config` and
`depends_on`.

```yaml
name: my-corpus
storage_uri: file:///data/out

stages:
  - name: source
    capability: source
    provider: source.local_folder
    config: {path: file:///data/corpus, sensitivity: internal, extensions: [txt, md, pdf]}

  - name: parse
    capability: parse
    provider: parser.pdfplumber
    depends_on: [source]

  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
    config: {max_tokens: 256, overlap_tokens: 32, min_tokens: 8}

  - name: redact
    capability: redaction
    provider: redaction.hybrid_rule
    depends_on: [chunk]

  - name: export_corpus
    capability: export
    provider: export.jsonl_parquet
    depends_on: [redact]          # depending on redact is what makes the corpus PII-safe
    config:
      basename: corpus
      # every block below is an OPTIONAL, ADDITIVE {provider, config} sub-Provider seam
      embedder: {provider: embedding.hashing, config: {dimension: 32}}
      bm25:     {provider: tokenizer.regex}
      context_columns: true
```

**Wiring rules are enforced, not documented.** Some Capabilities hard-require particular parents:
`relation` needs both an `EntityMention` producer and a `ChunkRecord` producer; `graph` accepts
`DisambiguationRecord` **exclusively**. `latence stack check` catches all of it offline.

---

## 6 · The CLI

| Command | Use it when |
|---|---|
| `latence setup` | First contact. Generates a validated config from a corpus folder. |
| `latence stack check <yaml>` | Before every run. Offline lint: DAG, wiring, dead config keys. Sub-second. |
| `latence process --config <yaml>` | Run the wizard's config. Runs `stack check` first and refuses on errors. |
| `latence run <yaml>` | Run a hand-written pipeline. `--tuned` applies a device-fingerprinted config. |
| `latence contracts` | **Your reference.** Prints generated Stage contracts — signatures, carriers, wiring, registered Providers. `--capability parse` to narrow. |
| `latence stack validate <yaml>` | End-to-end run on the bundled corpus with assertions. This is the CI gate. |
| `latence delta <yaml>` | Files changed. Recomputes only the affected set. `--dry-run` classifies without running Stages. |
| `latence retract <yaml> --doc <id>` | Soft-delete: leaves the live corpus, retained in the prior Version for audit. |
| `latence purge <yaml> --doc <id>` | **Destructive, non-recoverable** GDPR erasure across every Version. Prompts unless `--yes`. |
| `latence bake-off <matrix.yaml>` | Compare Providers for one Stage under controlled conditions. |
| `latence tune <yaml>` | On a real GPU, sweep dtype/batch/compile knobs, correctness-gated. |

Other console scripts: `latence-demo` (the one-command showable artifact), `latence-console` (a
read-only localhost browser over runs and Quality Reports), `latence-sink-qdrant` (idempotent load
into Qdrant, plus `verify-absent` as erasure evidence), `latence-label-profile` (mine a type space
with no model and no network).

---

## 7 · What a run produces

```
<storage_uri>/_latence/runs/<run_id>/
├── manifest.json              # the exact Pipeline that produced this run, credentials scrubbed
├── quality-report.{json,md}   # the machine + human artifact substantiating "AI-ready"
├── checkpoints/<stage>.jsonl  # presence ⇒ resume
├── quarantine/<stage>.jsonl   # screened-out records; never exported
└── export/
    ├── corpus.jsonl, corpus.parquet
    ├── graph-nodes.parquet, graph-edges.parquet
    ├── graph.ttl, graph.graphml
    ├── bm25-stats.json, bm25-postings.parquet    # only with a bm25 block
    └── multivectors.parquet                      # only with keep_multivectors
```

The corpus Parquet always carries `record_id`, `schema_version`, `provenance`, `classification`,
`content`, `media_type`. Columns appear **only when the producing Stage is wired**, so the default
schema stays byte-stable: `embedding`, `sparse_indices`/`sparse_values`, `fde_embedding`,
`context_header`, `kg_node_ids`, `neighbor_node_ids`, `kg_node_community`, `kg_node_centrality`.

**The PII guarantee.** Where Redaction produced `masked_content`, that is what fills the `content`
cell, the embedder input, the sparse and multivector inputs, and the BM25 postings. Beneath any
learned redactor sits a deterministic floor for `ssn` / `credit_card` / `iban` that no Provider
choice can bypass. The single opt-out is `unsafe_unmasked_corpus: true`, and it is named that way
deliberately.

---

## 8 · Retrieval

Retrieval is a **separate, strictly stateless library** (`latence-retrieval`). It holds no index and
runs no first-stage search. The user keeps whatever engine they already run; the library fuses,
expands over the emitted graph, reranks and packs the candidates *that engine* returned.

It uses its own entry-point group, `latence.retrieval`, deliberately disjoint from
`latence.providers` — which is why `sparse.splade` (index time) and `query_sparse.splade` (query
time) can coexist without colliding.

```yaml
fuser:       {provider: fuser.rrf}        # rank-only, scale-free; the default
processors:                                # ordered; defaults to EMPTY on purpose
  - {provider: processor.expand}
  - {provider: processor.bm25}
  - {provider: processor.rerank}
  - {provider: processor.pack, config: {token_budget: 4000}}
fetch_top_k: 50
fuse_weights: {dense: 2.0, bm25: 1.0}
final_top_k: null
```

`processors` is empty by default because `rerank` needs a licence-cleared model and `pack` needs a
token budget — neither has a safe hidden default.

**The one architectural fact that decides whether the graph helps:** `graph` is a *first-stage
retrieval mode*, not a rescoring pass. `processor.expand` and `processor.multihop_expand` run after
fusion and can only reorder a pool the graph did not choose. Only a real graph leg — seeded from
`seed_node_ids`, entering fusion as a peer — can surface a document that dense and sparse never
retrieved.

What each signal needs on disk:

| Signal | Requires | Produced by |
|---|---|---|
| dense | `embedding` column | Export `config.embedder` |
| sparse | `sparse_indices` + `sparse_values` | Export `config.sparse_embedder` |
| bm25 | `bm25-stats.json` + `bm25-postings.parquet` | Export `config.bm25` |
| multivector | `multivectors.parquet` | `multivector_embedder` + `keep_multivectors: true` |
| graph 1-hop | `kg_node_ids` and friends | Export `context_columns: true` (needs a `context` Stage upstream) |
| graph multi-hop | `graph-edges.parquet` | `export.knowledge_graph` |

The DuckDB multi-hop leg needs nothing but that Parquet file — no server, no graph database, and it
never mutates the file. `duckdb` is imported lazily on first use.

---

## 9 · Adding a Provider — do this instead of editing core

If the user needs a model the framework does not ship, **you write a package, not a patch**. No core
edit is required or accepted.

1. Implement the Capability's Protocol. Structural satisfaction — do not import or subclass anything
   from core to "register".
2. Declare a `ProviderProfile`, with the weights licence and the code licence verified
   **separately**. An unverified licence must say so; do not guess one.
3. Register the entry point in `[project.entry-points."latence.providers"]`, named
   `<capability-prefix>.<name>`.
4. Keep heavy dependencies in *your* package and import them lazily.
5. Raise typed `ProviderError` — never a raw native error from the underlying library.
6. Pass conformance.

The full walkthrough is [Authoring a Provider](providers/README.md); the hands-on version with a
deliberate-sabotage exercise is [Bringing your own Provider](tutorials/04-bringing-your-own-provider.md).

---

## 10 · Verifying your work

```bash
uv run ruff check packages benchmark/s5 benchmark/serving benchmark/sota-campaign scripts
MYPY_CMD="uv run mypy" bash scripts/typecheck.sh
PYTHONHASHSEED=0 uv run pytest packages
uv sync --extra docs && uv run mkdocs build --strict     # if you touched docs/
```

Cheapest first, so a lint break costs seconds rather than ten minutes. The full merge gate is
`bash scripts/verify-local.sh`.

---

## 11 · Six things that will burn you

1. **`PYTHONHASHSEED=0` is mandatory and cannot be set from a fixture.** It is read once at
   interpreter startup, so it must be in the environment *before* the process launches. The root
   `conftest.py` fails the suite if it is not pinned.
2. **`-q` on the command line becomes `-qq`.** `addopts = "-q"` is already configured, so adding
   your own `-q` suppresses the count line. Pass `-o addopts=""` when you need to see counts.
3. **mypy cannot run whole-tree.** The package directories are hyphenated and every
   `tests/conftest.py` maps to the same bare module, so a single invocation aborts with
   `Duplicate module named "conftest"` before checking anything. Use `scripts/typecheck.sh`.
4. **`make type` is not the real gate.** It checks `src` only, over a hardcoded package list. CI runs
   `scripts/typecheck.sh`, which checks src **and** tests for every package.
5. **Capture exit codes by redirect, never by pipe.** A shell pipeline reports the status of its
   *last* command, so `verify-local.sh | tail -20` exits `0` on a thoroughly red tree. That mistake
   has been made here and produced a false green that reached a written summary.
6. **`LATENCE_CUDA=0` makes a GPU Provider skip-with-flag, which is a PASS.** That is deliberate —
   a skipped Stage says so in the Quality Report rather than fabricating a number. Do not "fix" a
   skip into a fake result.

---

## 12 · Where to read more

| You want | Read |
|---|---|
| The vocabulary, exactly | [`CONTEXT.md`](https://github.com/ddickmann/latence/blob/main/CONTEXT.md) |
| Why any of it is the way it is | [the decision log](adr/index.md) — every architectural decision, with its rejected alternatives |
| The generated contracts | [`reference/stage-contracts.md`](reference/stage-contracts.md), or `latence contracts` |
| A worked path end to end | [Quickstart](quickstart.md), then the [tutorials](tutorials/index.md) |
| The mechanics, Stage by Stage | [Pipeline anatomy](anatomy/index.md) |
| What is measured, and what is not | [Benchmarks](BENCHMARKS.md) and [`PERF-RESULTS.md`](perf-results.md) |
| To change this repository | [`AGENTS.md`](https://github.com/ddickmann/latence/blob/main/AGENTS.md) |
