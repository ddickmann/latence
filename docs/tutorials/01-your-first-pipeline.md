# Tutorial 1 — Your first pipeline

**What you will build.** A folder of four ordinary documents turned into an AI-ready corpus: chunked
and PII-masked text with vectors, a knowledge graph whose entities are resolved across documents,
BM25 term statistics, and a Quality Report that substantiates all of it. Then you will take the
pipeline apart and see what each of the fourteen stages actually produced.

**Prerequisites.** A clone of the framework with `uv sync` run, and nothing else. This tutorial uses
only the in-core reference Providers, so it needs **no network, no model downloads and no GPU**, and
the whole run finishes in well under a second. That is deliberate: you should be able to see the
shape of the system before you spend anything on it.

**Time.** About 20 minutes, most of it reading output.

---

## 1. Make a corpus

Anywhere outside the repo:

```bash
mkdir -p first-pipeline/documents && cd first-pipeline
```

Create four files under `documents/`. They are small on purpose — small enough that you can hold the
whole corpus in your head and check the pipeline's every claim by eye.

`documents/handbook.md`

```markdown
# Acme Corporation Employee Handbook

Acme Corporation was founded in 1998. Alice Turner works at Acme Corporation
as head of engineering. Bob Reyes joined Acme in 2004.

Payroll questions go to payroll@acme.example or +1 555 0100.
```

`documents/contract.md`

```markdown
# Master Services Agreement

This agreement is between Acme Corporation and Globex. Carol Nguyen signed
on behalf of Globex. Invoices are settled to IBAN DE89370400440532013000.
```

`documents/policy.md`

```markdown
# Travel and Expense Policy

Globex reimburses travel booked through the corporate portal. Employees of
Globex must submit receipts within 30 days. Alice Turner approves exceptions.
```

`documents/incident.md`

```markdown
# Incident Report 2024-07

An outage affected the Globex billing service. Bob Reyes led the response.
Bob Reyes works at Acme Corporation and was on loan to Globex that week.
```

Note what is in there beyond the prose: an email address, a phone number and an IBAN (all PII), and
— the part that matters — **facts that only make sense across documents**. Bob Reyes appears in two
files. Alice Turner appears in two. Nothing in `policy.md` mentions Acme, yet Acme is one step away
through Alice.

✓ **look for** — `ls documents/` shows exactly four `.md` files.

---

## 2. Write the stack

The [quickstart](../quickstart.md) generates a stack with `latence setup`, which is the front door
and the right default. Here you will write one by hand instead, because seeing the whole DAG in one
readable file is the fastest way to understand what a pipeline *is*. A stack config is a plain,
diffable YAML file either way — the wizard's output invites hand-editing in its own header.

Save this as `latence.stack.yaml` next to `documents/`:

```yaml
# A CPU-only, offline, deterministic full-spine stack.
name: first-pipeline
storage_uri: "file://./latence-out"

stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: ./documents
      sensitivity: internal
      extensions: [txt, md, markdown]

  - name: intake_screen
    capability: intake_screening
    provider: screening.intake_signature
    depends_on: [source]
    config: {max_bytes: 2097152, max_zip_ratio: 50.0}

  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [intake_screen]

  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
    config: {max_tokens: 256, overlap_tokens: 32, min_tokens: 8}

  - name: content_screen
    capability: content_screening
    provider: screening.content_keyword
    depends_on: [chunk]

  - name: entities
    capability: entity_extraction
    provider: entity.gazetteer
    depends_on: [content_screen]
    config:
      labels:
        ORG: {terms: [Acme, Acme Corporation, Globex]}
        PERSON: {terms: [Alice Turner, Bob Reyes, Carol Nguyen]}

  - name: relations
    capability: relation_extraction
    provider: relation.pattern
    depends_on: [content_screen, entities]
    config:
      window_chars: 200
      relations:
        works_for: {head: PERSON, tail: ORG, triggers: [works at, employed by, joined]}
        partner_of: {head: ORG, tail: ORG, triggers: [partner of, partnered with, between]}

  - name: redact
    capability: redaction
    provider: redaction.hybrid_rule
    depends_on: [content_screen]

  - name: profiling
    capability: profiling
    provider: profiling.statistical
    depends_on: [parse, entities]

  - name: disambiguate
    capability: disambiguation
    provider: disambiguation.cascade
    depends_on: [entities, relations]

  - name: graph
    capability: graph_assembly
    provider: graph.canonical
    depends_on: [disambiguate]
    config:
      corpus_id: first-pipeline
      graph_features: {provider: graph_features.pagerank, config: {}}

  - name: enrich
    capability: context_enrichment
    provider: context.kg_header
    depends_on: [redact, graph]
    config: {max_neighbors_per_entity: 4, max_triples: 6}

  - name: export_kg
    capability: export
    provider: export.knowledge_graph
    depends_on: [graph]
    config: {basename: graph}

  - name: export_corpus
    capability: export
    provider: export.jsonl_parquet
    depends_on: [enrich]
    config:
      basename: records
      context_columns: true
      embedder: {provider: embedding.hashing, config: {dimension: 256}}
      bm25: {provider: tokenizer.regex, config: {}}
```

Read it once from the top. Every stage is the same four things: a `name`, the **Capability** it
fulfils, the **Provider** that fulfils it, and what it `depends_on`. That is the entire structure.
Swapping any Provider is a one-line edit and nothing else moves — which is the framework's central
claim, and the thing Tutorial 4 makes you do yourself.

Three details worth pausing on:

* **`depends_on` is a DAG, not a list.** `redact` depends on `content_screen`, not on `entities` —
  redaction and extraction are independent branches over the same chunk stream. `enrich` waits for
  both `redact` and `graph`, which is what lets the exported corpus be masked *and* graph-enriched
  at once.
* **`graph_features` and `bm25` are nested `{provider, config}` blocks.** A Provider can resolve
  another Provider through the same registry, so optional signals bolt on without a new stage.
* **`storage_uri` is an fsspec URI.** `file://./latence-out` here; `s3://bucket/prefix` works
  unchanged, and nothing else in the file changes.

### Check it before you run it

```bash
uv run latence stack check latence.stack.yaml
```

This reads the file and inspects Provider *classes* — no corpus, no models, no run. It catches DAG
errors, the phase-boundary invariant, capability mis-wiring, every nested `{provider, config}`
block, and config keys that go nowhere.

✓ **look for**

```
stack check OK — 0 error(s), 0 warning(s)
```

It should finish in well under a second. If it reports an error, fix the YAML — the whole point of
this command is that a typo costs you a second instead of a pipeline. A Provider that is simply not
installed on this host is reported as a **warning**, not an error, so the same file passes on a CPU
box and a GPU pod.

---

## 3. Run it

```bash
uv run latence run latence.stack.yaml --run-id run-0001
```

✓ **look for** — two log lines and then the Quality Report as JSON on stdout:

```
INFO latence loaded pipeline 'first-pipeline' with 14 stages
INFO latence run complete: 4 documents, contracts_complete=True
```

`contracts_complete=True` is the assertion that matters. It means every record the run produced
carries its full Provenance and Classification and has offsets that are in range and correctly
ordered — the invariant every Provider is held to. A run that produces beautiful-looking output with
`contracts_complete=False` has produced nothing you can trust.

The rendered report is also on disk:

```bash
cat latence-out/_latence/runs/run-0001/quality-report.md
```

✓ **look for** — this, modulo timings:

```
# Quality Report — first-pipeline (run-0001)

- documents: 4
- duration: 0.133s
- contracts complete: PASS
  - provenance 23/23, classification 23/23, offsets aligned 23/23
- provenance drift: clean (8 lookups, max 0 chars)

## Stages
- source [source/source.local_folder]: 0 in -> 4 out, 0.008s
- intake_screen [intake_screening/screening.intake_signature]: 4 in -> 4 out, 0.006s
- parse [parse/parser.plaintext]: 4 in -> 4 out, 0.006s
- chunk [chunk/chunk.markdown]: 4 in -> 4 out, 0.006s
- content_screen [content_screening/screening.content_keyword]: 4 in -> 4 out, 0.007s
- entities [entity_extraction/entity.gazetteer]: 4 in -> 19 out, 0.006s
- redact [redaction/redaction.hybrid_rule]: 4 in -> 4 out, 0.010s
- relations [relation_extraction/relation.pattern]: 23 in -> 9 out, 0.007s
- profiling [profiling/profiling.statistical]: 23 in -> 5 out, 0.007s
- disambiguate [disambiguation/disambiguation.cascade]: 28 in -> 14 out, 0.008s
- graph [graph_assembly/graph.canonical]: 14 in -> 14 out, 0.010s
- enrich [context_enrichment/context.kg_header]: 18 in -> 4 out, 0.010s
- export_kg [export/export.knowledge_graph]: 14 in -> 14 out, 0.026s
- export_corpus [export/export.jsonl_parquet]: 4 in -> 4 out, 0.016s

## Stage metrics present: Parse, Chunk, Screening, Entities, Relations, Redaction, Profiling, ...

## Context enrichment
- chunks enriched: 4, avg neighbors/chunk: 1.250, hub entities capped: 2
```

Those record counts are exact, not approximate. Every Provider in this stack declares itself
deterministic, and the conformance suite fails any Provider that declares determinism it does not
have — so a seeded run of this stack is byte-reproducible, and a different count means the input
differs, not the machine.

The interesting line is `disambiguate: 28 in -> 14 out`. Twenty-eight mention records went in and
fourteen canonical records came out: the resolver collapsed "Acme", "Acme Corporation" and "Acme" in
three different documents into one node. That collapse is the entire reason the graph is worth
building, and it is why the pipeline does resolution *after* extraction rather than forcing one
schema up front.

---

## 4. Look at what each stage produced

Every stage checkpoints its output as newline-delimited JSON. This is not a debug mode — it is the
architecture. Run state is files on Storage, no database, and each checkpoint is the durable,
inspectable dataset the next stage reads.

```bash
ls latence-out/_latence/runs/run-0001/checkpoints/
```

✓ **look for** — one `.jsonl` per stage: `source`, `intake_screen`, `parse`, `chunk`,
`content_screen`, `entities`, `relations`, `redact`, `profiling`, `disambiguate`, `graph`, `enrich`,
`export_kg`, `export_corpus`.

Walk down them. (`python -m json.tool` is enough; `jq` is nicer.)

### `parse.jsonl` — one record per document

```bash
head -c 400 latence-out/_latence/runs/run-0001/checkpoints/parse.jsonl
```

✓ **look for** — a record carrying `content` (the document as markdown), a `page_map`, a
`disposition`, and a `provenance` block naming `source_uri`, `file_name`, `file_type`, `file_size`
and a content-addressed `document_id` of the form `sha256:<64 hex>`. That id is how the document is
identified for the rest of its life — including by `latence retract` and `latence purge`.

### `chunk.jsonl` — retrieval-sized pieces that remember where they came from

✓ **look for** — each chunk carries `token_count`, an `offset_map`, and a `page_slice` listing
exactly the document pages its `[char_start, char_end)` span overlaps. A chunk is *self-describing*:
you can resolve any span inside it back to a source page with the chunk alone, in any order, with no
per-document state. That property is what keeps provenance exact all the way to the export.

### `entities.jsonl` — mentions, with true document offsets

✓ **look for** — one record per mention with `text`, `label`, `confidence`, `chunk_record_id`, and a
`provenance` whose `char_start`/`char_end` point into the **original document**, not into the chunk.
Nineteen of them for this corpus.

### `disambiguate.jsonl` — the corpus-level collapse

✓ **look for** — records carrying a `scope` of `entity` or `relation` rather than raw mentions —
five and nine of them respectively, for this corpus:

```bash
uv run python -c "
import collections, json
print(collections.Counter(
    json.loads(l)['scope']
    for l in open('latence-out/_latence/runs/run-0001/checkpoints/disambiguate.jsonl')))
"
# Counter({'relation': 9, 'entity': 5})
```

This is the phase boundary. Everything above it worked on one chunk or document at a time, with no
corpus-wide state. Everything from here down sees the whole corpus. That split is what keeps the
expensive half parallel and independent of corpus size.

### `screening/` and `quarantine/`

```bash
ls latence-out/_latence/runs/run-0001/screening/ latence-out/_latence/runs/run-0001/quarantine/
```

✓ **look for** — a `.jsonl` for each of the two screening checkpoints, and empty quarantine files.
Nothing in this corpus is oversized, malformed or injection-bearing, so nothing was held back. Run
`latence-demo` if you want to see them populated: its bundled corpus plants a zip bomb, an oversized
file, a type-spoofed PDF and a prompt-injection payload, and the demo exits non-zero if any of them
gets through.

---

## 5. Check the two deliverables

```bash
ls latence-out/_latence/runs/run-0001/export/
```

✓ **look for** — eight files:

```
bm25-postings.parquet  graph-edges.parquet  graph.graphml  records.jsonl
bm25-stats.json        graph-nodes.parquet  graph.ttl      records.parquet
```

Two quick sanity checks before the next tutorial reads them properly.

**The corpus is masked.** The exported chunk text is the redacted text — the pipeline does not
export the clean copy:

```bash
uv run python -c "
import json
for line in open('latence-out/_latence/runs/run-0001/export/records.jsonl'):
    r = json.loads(line)
    for span in r['pii_spans']:
        print(r['provenance']['file_name'], span['pii_type'], span['placeholder'], span['detector'])
"
```

✓ **look for** — three spans across two documents, each naming the detector that found it:

```
contract.md iban [IBAN] regex:iban
handbook.md email [EMAIL] regex:email
handbook.md phone [PHONE] regex:phone
```

and those placeholders present in the exported `content`, which matches the `redaction` block of
`quality-report.json` (`pii_total: 3`, `documents_with_pii: 2`).

!!! note "What this Provider will and will not catch"
    `redaction.hybrid_rule` is the in-core deterministic redactor: a curated regex library plus a
    gazetteer. It caught the IBAN above because the IBAN was written without spaces, which is what
    its pattern expects. Write the same IBAN as `DE89 3704 0044 0532 0130 00` and this Provider
    will not recognise it as an IBAN — a digit-run inside it gets masked as a phone number instead:
    masked, but mislabelled.

    That is the honest limit of a rule-based redactor, and it is exactly why the learned
    `redaction.gliner2` Provider exists, and why the framework enforces a
    **universal financial-PII floor** at the shared seam every redaction Provider funnels through
    ([ADR-0044](../adr/0044-universal-redaction-floor-and-optional-mask-before-extract.md)). Safety
    categories are never narrowed per chunk and never depend on which model you plugged in.

**The graph resolved entities across documents.**

```bash
uv run python -c "
import pyarrow.parquet as pq
t = pq.read_table('latence-out/_latence/runs/run-0001/export/graph-nodes.parquet')
for r in t.to_pylist():
    print(f\"{r['type']:7} {r['canonical_name']:14} centrality={r['centrality']:.4f}\")
"
```

✓ **look for** — exactly five nodes, and `Acme` the most central of them:

```
PERSON  Bob Reyes      centrality=0.1493
ORG     Acme           centrality=0.3396
ORG     Globex         centrality=0.2127
PERSON  Carol Nguyen   centrality=0.1493
PERSON  Alice Turner   centrality=0.1493
```

Five nodes from nineteen mentions across four documents. `Acme`, `Acme Corporation` and the bare
`Acme` in `incident.md` are one node. The centrality column is PageRank over the assembled graph —
it is not decoration, it is a signal the retrieval layer reads later to avoid being swamped by hub
entities.

!!! note "Five nodes is the ceiling here, and that is the point"
    `entity.gazetteer` finds exactly the terms you listed and nothing else. It found no dates, no
    monetary amounts, no product names, because you did not name any. That is its virtue —
    deterministic, offline, zero dependencies, and it cannot hallucinate an entity — and it is also
    its ceiling: on a real corpus you do not know the term list in advance.

    That is the concrete reason production stacks run a learned zero-shot extractor
    (`fused_entity_relation.gliner2`) against a schema *induced per chunk*, rather than a term list
    fixed up front. Tutorial 4 shows how a Provider like that plugs in behind this same seam.

---

## 6. Change one line and see the seam

The claim the whole architecture rests on is that a Provider is swappable without touching anything
else. Test it. Change the chunker:

```yaml
  - name: chunk
    capability: chunk
    provider: chunk.sentence_window        # was: chunk.markdown
    depends_on: [parse]
    config: {max_tokens: 256, overlap_tokens: 32, min_tokens: 8}
```

```bash
uv run latence stack check latence.stack.yaml
uv run latence run latence.stack.yaml --run-id run-0002
```

✓ **look for** — `stack check OK`, then `contracts_complete=True` again, and a report line naming
the new Provider:

```
- chunk [chunk/chunk.sentence_window]: 4 in -> 4 out, 0.017s
```

Nothing else in the file changed, and no stage below `chunk` knows or cares. On this corpus the
downstream counts are unchanged too — each of these four documents is short enough to be a single
chunk under either strategy, so the two chunkers happen to agree. That is a property of a
four-paragraph corpus, not of the seam; on documents longer than the 256-token budget the chunk
boundaries, and everything derived from them, diverge.

`chunk.sentence_window` packs whole sentences with sentence-aligned boundaries and no overlap, where
`chunk.markdown` splits on markdown structure with a configurable overlap — genuinely different
strategies behind one interface, not a rename. When you want to know which is better *for your
corpus* rather than in the abstract, that is what `latence bake-off` is for: it holds every other
stage constant and runs the same corpus through each candidate for one stage. The repo ships a
ready-made matrix for exactly this comparison (run it from inside the repo, where `matrix/` and
`stacks/` live):

```bash
uv run latence bake-off matrix/chunk.yaml
```

✓ **look for** — a `MATRIX-RESULTS-chunk.md` table plus a JSON companion, one row per candidate:

```
# Bake-off — `chunk` (chunk) — provider matrix

- base stack (other stages held constant): `default.yaml`
- corpus documents: 7
- candidates: 2

| candidate | quality | latency (docs/s) | memory measured/declared (MiB) | ... |
| `chunk.markdown`        | chunks=4 (offset_preserving_fraction=1, tokens=185) | 339.3 | 19.06 / 16 | ... |
| `chunk.sentence_window` | chunks=4 (offset_preserving_fraction=1, tokens=185) | 345.1 | 19.03 / 16 | ... |
```

Three things to notice. The quality, license and determinism columns are read from each Provider's
own **declared** profile plus the run's Quality Report — nothing is invented for the table. A
candidate that crashes or is device-skipped becomes a **row saying so**, never an aborted matrix.
And the corpus is the *bundled* one (7 documents), not your folder: a bake-off is a controlled
comparison of Providers, so it holds the corpus constant along with every other stage.

Latency here is wall-clock and machine-relative — reported, not gated. Two CPU chunkers within 2%
of each other on a seven-document corpus is a measurement, not a result.

Restore `chunk.markdown` before moving on, so the next tutorial's numbers match.

---

## What you have

A run directory that contains, in files you can read with ordinary tools: every stage's output, both
screening ledgers, a knowledge graph in three formats, a RAG corpus with vectors and BM25
statistics, and a report that says what happened. No database, no service, nothing hidden.

**Next:** [Understanding the output](02-understanding-the-output.md) — every artifact, field by
field, and how to trace a single graph edge back to the exact sentence that produced it.
