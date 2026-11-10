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
| `export/` | *What did I get?* The two deliverables — the corpus and the knowledge graph. This is what you ship. |
| `checkpoints/<stage>.jsonl` | *What did each stage produce?* One newline-delimited JSON file per stage, and the durable input the next stage reads. |
| `screening/` + `quarantine/` | *What was held back, and why?* The two safety checkpoints keep separate ledgers: what they inspected, and what they stopped. |
| `quality-report.json` / `.md` | *Can I trust it?* The machine- and human-readable verdict on the run. |
| `manifest.json` + `wal.log` | *What exactly ran?* The resolved pipeline as executed, and a batch-granular write-ahead log for crash forensics. |

Run state is **files on Storage — no database**. Every one of these is readable with ordinary tools,
copyable to another machine, and diffable. That is the same property that makes the pipeline
storage-agnostic: change `storage_uri` to `s3://…` and this whole tree lands there instead, with no
other change.

---

## 2. `records.parquet` — the corpus

This is the corpus as a table: one row per chunk, the form a warehouse or a vector store loads.
Start with its schema:

```bash
uv run python -c "
import pyarrow.parquet as pq
print(pq.read_schema('latence-out/_latence/runs/run-0001/export/records.parquet'))
"
```

✓ **look for** — seven columns:

```
record_id: string
schema_version: int64
provenance: string
classification: string
content: string
media_type: string
embedding: list<element: double>
  child 0, element: double
```

One of those columns is **not** in a default export: `embedding` appears because you wired an
`embedder` block. Without it the Parquet file has the first six columns and nothing else. Nothing
is written speculatively.

`records.jsonl` holds the same rows in the same order, and more of each: a JSONL row is the whole
chunk record, so it also carries `pii_spans`, `masked_content`, `offset_map`, `page_slice`,
`token_count` and `risk_markers`. The Parquet file is the flat table; the JSONL file is the
record. Read one row:

```bash
uv run python -c "
import json
r = json.loads(open('latence-out/_latence/runs/run-0001/export/records.jsonl').readline())
print(json.dumps({k: v for k, v in r.items() if k != 'embedding'}, indent=1)[:1200])
"
```

### The columns, and what each is for

**`record_id`** — `sha256:<document hash>#chunk-<n>`. Stable and content-addressed: the same
document bytes always give the same ids, so a row can be joined to its checkpoints, to the
mention ids in the graph, and to the same row in a later run.

**`content`** — the chunk text **as exported**, which is the *redacted* text. This is the single
most important thing to understand about the corpus: redaction is a chunk→chunk transform that sets
each chunk's `masked_content`, and the Export materialises that. There is no clean copy in
`export/`. Look at `contract.md`'s row and you will see `IBAN [IBAN].`, not the digits. (The
checkpoints of the stages before the export do hold the unmasked text; they are run state, not
the deliverable.)

**`provenance`** — a JSON object carrying `source_uri`, `file_name`, `file_type`, `file_size`, the
content-addressed `document_id`, `page_start` / `page_end`, and `char_start` / `char_end` **in the
original document's coordinates**. Not the chunk's coordinates. That distinction is the difference
between a citation you can check and a citation you can only hope about.

**`classification`** — `language`, `category`, and `sensitivity` (`internal` here, from your
`source` config). Sensitivity travels with the record and redaction policy keys off it.

**`embedding`** — the dense vector, at the width the Embedder declares. In this stack it is 256
dimensions from `embedding.hashing`.

!!! warning "`embedding.hashing` is a reference, not a semantic model"
    The in-core Embedder is a deterministic hashing embedder: a token-hash bag-of-words. It exists
    so the seam, the emission and the whole offline test suite can run with zero downloads and
    byte-reproducible output — and it does that job perfectly.

    What it is **not** is semantic. Two paraphrases with no shared words get near-zero similarity
    from it, because it has no notion of meaning. Do not judge anything that depends on meaning by
    these vectors. Wire a real Embedder — `embedding.sentence_transformers` with a Granite r2
    checkpoint — before drawing a conclusion. [Embedders](../embedders.md) lists the Providers and
    the Stages that use one.

---

## 3. The knowledge graph

Three renderings of one graph. `graph-nodes.parquet` and `graph-edges.parquet` are the source of
truth; `graph.ttl` (RDF/Turtle) and `graph.graphml` are the same nodes and edges for tools that
speak those formats, with the Evidence reduced to a count. The column-by-column reference is
[the knowledge-graph export](../anatomy/04-knowledge-graph-export.md).

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

* `node_id` — a 16-hex content hash. **Not** a readable label. Edges refer to nodes by this id, so
  to follow the edges of "Acme" you first find its `node_id` by filtering on `canonical_name`.
* `entity_id` — `entity:acme`, the normalised surface the resolver clustered on.
* `member_mention_ids` — a JSON array of **every mention that collapsed into this node**, each one a
  `…#chunk-N#entity-M` id you can look up in `checkpoints/entities.jsonl`.
* `source_document_ids` — the documents this node was seen in. Three, for Acme, and its
  `properties` record `mention_count: 7` across them.
* `properties` — a JSON object: `confidence`, `document_count`, `mention_count`, `linked`.

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
 "schema_version": 22,
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
['chunk', 'completeness', 'conformance', 'contracts_complete', 'delta', 'disambiguation',
 'document_count', 'drift', 'entities', 'goldset', 'graph', 'parse', 'pipeline_name',
 'profiling', 'redaction', 'relations', 'run_id', 'schema_version', 'screening', 'stages',
 'total_duration_seconds', 'type_consolidation']
```

Four of them are `null` for this run. `delta` is filled by [`latence delta`](05-production.md),
`goldset` by a gold-set evaluation, `conformance` when a conformance roll-up is folded in, and
`type_consolidation` when the stack has a type-consolidation stage, which this one does not.

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

## 6. `manifest.json` and `wal.log`

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
`depends_on` and its config as you wrote it, plus the pipeline-level settings with their defaults
filled in (`extract_on`, `schema_version` and the rest). If you ever need to answer "what exactly
produced this export", this file is the answer, and it sits next to the export rather than in a
wiki.

One thing to know when you read it: values under keys that look like credentials are written as
`***REDACTED***`, so a key or a token never lands in the manifest. The match is on the key's
name, and `token` is one of the patterns, so this stack's chunk budget is redacted too:

```
{'max_tokens': '***REDACTED***', 'min_tokens': '***REDACTED***', 'overlap_tokens': '***REDACTED***'}
```

The stack file itself is the record of those three values.

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
its sentence. Tell the opt-in column from the base ones. Tell an honest gap (`skipped`,
`UNVERIFIED`) from a claim.

**Next:** [Bringing your own Provider](04-bringing-your-own-provider.md).
