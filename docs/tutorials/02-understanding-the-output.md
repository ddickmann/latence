# Tutorial 2 — Understanding the output

**What you will build.** Nothing new — this tutorial builds *fluency*. By the end you will know
what every emitted artifact contains, which field to reach for, and how to take a single edge in
the knowledge graph and produce the exact source sentence that justifies it. That last skill is the
one that turns "the pipeline says so" into an audit.

**Prerequisites.** [Tutorial 1](01-your-first-pipeline.md)'s output at
`latence-out/_latence/runs/run-0001/`. Everything here is read-only; you will not re-run anything.

**Time.** About 25 minutes.

---

## 1. The shape of a run directory

```bash
find latence-out -type f | sort
```

✓ **look for** — five groups of files. They are worth knowing apart, because they answer different
questions:

| Path | Question it answers |
| --- | --- |
| `export/` | *What did I get?* The two deliverables — the RAG corpus and the knowledge graph — plus their optional signal sidecars. This is what you ship. |
| `checkpoints/<stage>.jsonl` | *What did each stage produce?* One newline-delimited JSON file per stage, and the durable input the next stage reads. |
| `screening/` + `quarantine/` | *What was held back, and why?* The two safety checkpoints keep separate ledgers: what they inspected, and what they stopped. |
| `quality-report.json` / `.md` | *Can I trust it?* The machine- and human-readable verdict on the run. |
| `manifest.json` + `wal.log` | *What exactly ran?* The resolved pipeline as executed, and a batch-granular write-ahead log for crash forensics. |

Run state is **files on Storage — no database**. Every one of these is readable with ordinary tools,
copyable to another machine, and diffable. That is the same property that makes the pipeline
storage-agnostic: change `storage_uri` to `s3://…` and this whole tree lands there instead, with no
other change.

---

## 2. `records.parquet` — the RAG corpus

This is the file a retrieval system consumes. Start with its schema:

```bash
uv run python -c "
import pyarrow.parquet as pq
print(pq.read_schema('latence-out/_latence/runs/run-0001/export/records.parquet'))
"
```

✓ **look for** — twelve columns:

```
record_id: string
schema_version: int64
provenance: string
classification: string
content: string
media_type: string
embedding: list<element: double>
context_header: string
kg_node_ids: list<element: string>
neighbor_node_ids: list<element: string>
kg_node_community: list<element: string>
kg_node_centrality: list<element: double>
```

Four of those columns are **not** in a default export. `embedding` appears because you wired an
`embedder` block; the last four appear because you set `context_columns: true` on a stack that has
an `enrich` stage. Every signal column is additive: with none of them wired, the Parquet schema is
byte-identical to the base one. Nothing is written speculatively.

Read one row:

```bash
uv run python -c "
import json
r = json.loads(open('latence-out/_latence/runs/run-0001/export/records.jsonl').readline())
print(json.dumps({k: v for k, v in r.items() if k != 'embedding'}, indent=1)[:1200])
"
```

### The columns, and what each is for

**`record_id`** — `sha256:<document hash>#chunk-<n>`. Stable, content-addressed, and the same key
the Qdrant sink writes and the query side reads back. This is why a collection loaded by this
framework needs no mapping configuration.

**`content`** — the chunk text **as exported**, which is the *redacted* text. This is the single
most important thing to understand about the corpus: redaction is a chunk→chunk transform that sets
each chunk's `masked_content`, and the Export materialises that. There is no clean copy in the
deliverable. Look at `contract.md`'s row and you will see `IBAN [IBAN].`, not the digits.

**`provenance`** — a JSON object carrying `source_uri`, `file_name`, `file_type`, `file_size`, the
content-addressed `document_id`, `page_start` / `page_end`, and `char_start` / `char_end` **in the
original document's coordinates**. Not the chunk's coordinates. That distinction is the difference
between a citation you can check and a citation you can only hope about.

**`classification`** — `language`, `category`, and `sensitivity` (`internal` here, from your
`source` config). Sensitivity travels with the record and redaction policy keys off it.

**`embedding`** — the dense vector, at the width the Embedder declares. In this stack it is 256
dimensions from `embedding.hashing`.

!!! warning "`embedding.hashing` is a reference, not a semantic model"
    The in-core Embedder is a deterministic hashing embedder: it maps terms to buckets. It exists so
    the seam, the emission and the whole offline test suite can run with zero downloads and
    byte-reproducible output — and it does that job perfectly.

    What it is **not** is semantic. Two paraphrases with no shared words get near-zero similarity
    from it, because it has no notion of meaning. If you evaluate "does dense retrieval beat BM25?"
    against this Embedder, you are evaluating a lexical signal against a lexical signal and the
    answer is meaningless. Wire a real one — `embedding.sentence_transformers` with a Granite r2
    checkpoint — before drawing any conclusion about dense retrieval.
    [Tutorial 3](03-retrieval.md) returns to this.

**`context_header`** — a JSON string holding the knowledge graph's projection onto this chunk. Open
one:

```bash
uv run python -c "
import json
for line in open('latence-out/_latence/runs/run-0001/export/records.jsonl'):
    r = json.loads(line)
    if r['provenance']['file_name'] == 'incident.md':
        print(json.dumps(r['context_header'], indent=1))
"
```

✓ **look for** — the chunk's own entities with their types, confidences, node ids, centralities and
communities, plus `neighbor_triples`:

```json
{
 "entities": ["Acme", "Bob Reyes", "Globex"],
 "entity_types": ["ORG", "PERSON", "ORG"],
 "kg_node_ids": ["96987fab1d4dd7e2", "28fcaa908348a40a", "9c7c04ce2d8d545a"],
 "kg_node_centrality": [0.33955224, 0.14925373, 0.21268657],
 "neighbor_node_ids": ["bb0d32639119f88d"],
 "neighbor_triples": ["Bob Reyes —works_for→ Acme", "Bob Reyes —works_for→ Globex"]
}
```

That `neighbor_node_ids` entry is Alice Turner, who is **not mentioned in `incident.md` at all**.
She is there because she is one edge away in the graph from an entity that is. The header is
prepended to the **embedding input only** — the stored `content` is untouched — so the dense vector
for this chunk is graph-aware while the text you show a user is not polluted.

**`kg_node_ids` / `neighbor_node_ids` / `kg_node_community` / `kg_node_centrality`** — the same
projection denormalised into native Parquet list columns, named for exactly the metadata keys the
query-time expander, multi-hop and packer components read by default. A store that copies Parquet
columns into candidate metadata makes graph-augmented retrieval work end to end with no glue code.
This is the hinge between the pipeline and retrieval, and Tutorial 3 uses it directly.

---

## 3. The knowledge graph

Three renderings of one graph. `graph-nodes.parquet` and `graph-edges.parquet` are the source of
truth; `graph.ttl` (RDF/Turtle) and `graph.graphml` are the same content for tools that speak them.

### Nodes

```bash
uv run python -c "
import json, pyarrow.parquet as pq
t = pq.read_table('latence-out/_latence/runs/run-0001/export/graph-nodes.parquet')
n = [r for r in t.to_pylist() if r['canonical_name'] == 'Acme'][0]
print(json.dumps({k: v for k, v in n.items() if k != 'evidence'}, indent=1))
"
```

✓ **look for** — the Acme node, and specifically these fields:

* `node_id` — a 16-hex content hash. **Not** a readable label. You cannot seed a graph traversal
  with the string `"Acme"`; you resolve the label to its `node_id`(s) via `canonical_name` first.
  This is the single most common first-run mistake with the retrieval side.
* `entity_id` — `entity:acme`, the normalised surface the resolver clustered on.
* `member_mention_ids` — a JSON array of **every mention that collapsed into this node**, each one a
  `…#chunk-N#entity-M` id you can look up in `checkpoints/entities.jsonl`.
* `source_document_ids` — the documents this node was seen in. Three, for Acme, and its
  `properties` record `mention_count: 7` across them.
* `properties` — `centrality`, `community`, `document_count`, `mention_count`, `linked`.

`member_mention_ids` is the auditable core of entity resolution. The claim "these three surface
forms are one organisation" is not asserted; it is *listed*, mention by mention, and each mention
carries offsets into its source document.

### Edges

```bash
uv run python -c "
import json, pyarrow.parquet as pq
t = pq.read_table('latence-out/_latence/runs/run-0001/export/graph-edges.parquet')
e = t.to_pylist()[0]
print(e['label'], e['source_node_id'], '->', e['target_node_id'])
print(json.dumps(json.loads(e['evidence']), indent=1))
"
```

✓ **look for** — an edge plus an `evidence` block:

```json
{
 "confidence": 0.75,
 "document_ids": ["sha256:93df2e95…"],
 "mention_ids": [
  "sha256:93df2e95…#chunk-0#entity-2",
  "sha256:93df2e95…#chunk-0#entity-6"
 ],
 "snippet": "edge 'works_for' bb0d32639119f88d -> 96987fab1d4dd7e2"
}
```

Be precise about what that is. The `evidence` block names the **source documents and the exact
mentions** the edge was built from; the `snippet` here is a structural label, not the source
sentence. The sentence itself is one lookup away — which is the next section, and the most useful
five minutes in this tutorial.

### RDF

```bash
head -12 latence-out/_latence/runs/run-0001/export/graph.ttl
```

✓ **look for** — Turtle under a `urn:latence:graph:` namespace, entities typed `lg:Entity` with
`rdfs:label`, `lg:entityType` and a typed `lg:confidence`. Loadable into any triple store as-is.

---

## 4. Trace an edge back to its sentence

This is the exercise that makes provenance concrete. The relation records carry `char_start` and
`char_end` **in original-document coordinates**, and the parse checkpoint carries the document
content those coordinates index into. So the evidence sentence is a slice.

```bash
uv run python -c "
import json
R = 'latence-out/_latence/runs/run-0001'
docs = {json.loads(l)['record_id']: json.loads(l) for l in open(R + '/checkpoints/parse.jsonl')}
for line in list(open(R + '/checkpoints/relations.jsonl'))[:3]:
    r = json.loads(line)
    p = r['provenance']
    text = docs[r['document_record_id']]['content'][p['char_start']:p['char_end']]
    print(f\"{r['label']}: {r['head_text']} -> {r['tail_text']}\")
    print(f\"   page {p['page_start']} of {p['file_name']}: {text!r}\")
"
```

✓ **look for** — each relation with the literal span it was extracted from:

```
works_for: Alice Turner -> Acme
   page 1 of handbook.md: 'Alice Turner works at Acme'
works_for: Alice Turner -> Acme
   page 1 of handbook.md: 'Alice Turner works at Acme Corporation\nas head of engineering. Bob Reyes joined Acme'
works_for: Alice Turner -> acme
   page 1 of handbook.md: 'Alice Turner works at Acme Corporation\nas head of engineering. ...'
```

Two things this makes visible.

First: **the offsets are real.** The slice is the actual source text, at the actual page, from the
actual file. Nothing was reconstructed or approximated. A relation's covering span runs from one
endpoint mention to the other and those may live in different chunks, which is why the span is
recorded against the document rather than a chunk.

Second: **look at the third row.** `Alice Turner -> acme`, lower case, spanning a much longer range
that runs past the end of the sentence into the payroll line. `relation.pattern` pairs mentions
inside a `window_chars: 200` window and fires on trigger phrases, so a wide window over a short
document produces plausible-looking but sloppy pairings — including one whose tail is the `acme` in
`payroll@acme.example`. Nine edges from a corpus that honestly contains about four.

That is not a bug being papered over; it is what a deterministic windowed extractor does, and it is
visible precisely because every edge carries its span. Narrow `window_chars`, or move to a learned
extractor. Either way, **the reason you can tell is that the evidence is there to check.** An
extraction system that gives you edges without spans gives you no way to notice this at all.

---

## 5. `quality-report.json` — the machine-readable verdict

The rendered `.md` you read in Tutorial 1 is a projection of this. The JSON is what you assert on in
CI.

```bash
uv run python -c "
import json
r = json.load(open('latence-out/_latence/runs/run-0001/quality-report.json'))
print(sorted(r.keys()))
"
```

✓ **look for** — the top-level blocks:

```
['chunk', 'completeness', 'conformance', 'context_enrichment', 'contracts_complete',
 'delta', 'disambiguation', 'document_count', 'drift', 'entities', 'goldset', 'graph',
 'parse', 'pipeline_name', 'profiling', 'redaction', 'relations', 'run_id',
 'schema_version', 'screening', 'stages', 'total_duration_seconds', 'type_consolidation']
```

`delta` and `goldset` are `null` for a plain run — the first is filled by
[`latence delta`](05-production.md), the second by a gold-set evaluation.

The blocks worth knowing:

**`contracts_complete`** — the run-level pass/fail on Provenance, Classification and offset
alignment. This is the one to gate CI on.

**`graph`** — `nodes_total`, `edges_total`, `nodes_by_type`, `edges_by_label`,
`edges_with_evidence`, and `evidence_coverage`. For Tutorial 1's run: 5 nodes, 9 edges,
`evidence_coverage: 1.0`. An evidence coverage below 1.0 means some edge cannot be traced — treat
that as a defect, not a statistic.

**`redaction`** — `documents_redacted`, `documents_with_pii`, `pii_total`, `per_type`, `per_action`.
Note what is *not* here: no PII values, no record content. The report is designed to be safe to
persist and share, and the conformance suite's C6 check asserts that Providers do not leak raw PII
into non-content fields, logs or spans.

**`drift`** — the provenance-resolution roll-up: how many page lookups happened, how many needed
fuzzy recovery, the maximum character drift. `clean (8 lookups, max 0 chars)` means every span
resolved exactly.

**`stages[]`** — per stage: `records_in`, `records_out`, `duration_seconds`, `device_selected`,
`license`, `model_id`, `skipped` + `skip_reason`, `error_count` + `error_category`, and
`from_checkpoint`. Two of those are the honesty machinery:

* `device_selected` and `skipped` are how the framework refuses to fabricate a GPU number. A
  `compute="gpu"` Provider on a CPU-only host is recorded as skipped-with-reason, never quietly run
  on CPU and reported as if it had run on GPU.
* `license` is read from the Provider's declared profile, which cannot be constructed with a
  dishonest claim — a `license_verified=True` without a citation and a date raises at construction.

---

## 6. The signal sidecars

**`bm25-stats.json`** — the corpus term statistics a store loads to drive **its own** first-stage
BM25 search. The framework emits the artifact; it does not run the search
([ADR-0048](../adr/0048-retrieval-tooling-is-strictly-stateless.md)).

```bash
uv run python -c "
import json
d = json.load(open('latence-out/_latence/runs/run-0001/export/bm25-stats.json'))
print({k: v for k, v in d.items() if k != 'terms'})
print('terms:', len(d['terms']), d['terms'][:3])
"
```

✓ **look for**

```
{'artifact': 'latence.bm25.termstats', 'avgdl': 28.0, 'doc_count': 4,
 'tokenizer': 'tokenizer.regex', 'total_tokens': 112, 'version': 1}
terms: 76 [{'df': 1, 'idf': 1.2039728, 'term': '07'}, ...]
```

The `tokenizer` field is the point: BM25's statistics are a pure function of how text is split into
terms, so the tokenisation is a **Capability seam** and the artifact records which Provider produced
it. The statistics are computed over the *stored redacted* text, so the lexical index can never
re-leak the PII the corpus removed.

**`bm25-postings.parquet`** — per-record forward postings: `record_id`, `doc_len`, and parallel
`terms` / `term_frequencies` arrays, with terms strictly ascending so two runs agree byte for byte.

---

## 7. `manifest.json` and `wal.log`

```bash
uv run python -c "
import json
m = json.load(open('latence-out/_latence/runs/run-0001/manifest.json'))
print(sorted(m.keys()))
print([s['name'] + ':' + s['provider'] for s in m['pipeline']['stages']][:4])
"
```

✓ **look for** — `['pipeline', 'run_id']`, then the resolved stage list:

```
['source:source.local_folder', 'intake_screen:screening.intake_signature',
 'parse:parser.plaintext', 'chunk:chunk.markdown']
```

`manifest.json` holds the pipeline **as executed** — every stage with its capability, provider,
`depends_on` and the complete config it ran with, defaults filled in. If you ever need to answer
"what exactly produced this export", this file is the answer, and it sits next to the export rather
than in a wiki.

```bash
head -3 latence-out/_latence/runs/run-0001/wal.log
```

✓ **look for** — batch-granular events:

```
{"event": "stage_begin", "stage": "source", "ts": …}
{"batch": 0, "event": "stage_batch", "records": 4, "stage": "source", "ts": …}
{"event": "stage_commit", "stage": "source", "ts": …}
```

The WAL is for crash forensics — it tells you how far a killed run got inside a stage. It is
**not** what resume reads: a finalized checkpoint's presence is what makes resume skip a stage, and
checkpoints are written atomically, so a reader ever sees the previous checkpoint or the complete
new one, never a torn write. [Tutorial 5](05-production.md) uses this.

---

## What you can now do

Point at any number the pipeline produced and name the file it came from. Take an edge and produce
its sentence. Tell a signal column from a base one. Tell an honest gap (`skipped`, `UNVERIFIED`)
from a claim.

**Next:** [Retrieval](03-retrieval.md) — turn `records.parquet`, `bm25-stats.json` and
`graph-edges.parquet` into a working four-signal query path, and see what each signal actually
contributes.
