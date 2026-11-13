# 2 · Extraction and privacy

*Chunks arrive: sized to fit the extraction model's window, screened, each one carrying the exact
character span and page it came from. This chapter is what happens to them before the pipeline is allowed to look at
the corpus as a whole — what types to look for, what to pull out, and what must never leave.*

Four stages live here, and they are deliberately **siblings, not a chain**. `extract` and
`redact` both hang off the same upstream stage and read the same clean chunk text; neither
waits for the other. `profiling` folds documents and mentions into statistics. Only one
configuration line — `extract_on` — would put redaction *in front of* extraction, and §6
explains exactly what that costs and buys.

---

## 1 · What arrives and what leaves

Everything in this chapter is expressed in two carriers and three record types, all in
`packages/latence-core/src/latence_core/contracts.py`.

| Carrier | Produced by | Key fields this chapter fills |
|---|---|---|
| `ChunkRecord` (in **and** out) | chunk → screening → induce → redact | `induced_labels`, `masked_content`, `pii_spans`, `redaction_disabled_for_sensitive`, `risk_markers` |
| `EntityMention` | `extract` | `label`, `text`, `confidence`, `provenance.char_start/char_end/page_start/page_end` |
| `RelationMention` | `extract` | `label`, `head_mention_id`, `tail_mention_id`, `head_text`, `tail_text`, `confidence`, covering span |
| `FeatureRecord` | `profiling` | `DOCUMENT`- and `CORPUS`-scope statistics |

Two of these stages are **chunk → chunk transforms**: induction and redaction hand back the
*same* chunks with extra fields set, in input order, one for one. That shape is what lets them
be siblings — a transform that adds fields cannot invalidate another transform that adds
different ones.

The extraction stage is the exception: it consumes chunks and returns a `FusedExtraction`
dataclass holding both lists together
(`packages/latence-core/src/latence_core/capability.py:253`), which the Runner routes to *both*
downstream consumers without running a second entity stage.

Three invariants hold across all of it, and each is enforced by a validator rather than by
convention:

- **A mention's offsets are document offsets, not chunk offsets.** `EntityMention` refuses to
  validate without `char_start`, `char_end`, `page_start` and `page_end` on its Provenance
  (`contracts.py:1202`). A Provider that forgets to resolve them cannot emit a record at all.
- **A PII span never stores the value it found.** `PIISpan` carries the type, the offsets, the
  action and the derived placeholder — never the surface string (`contracts.py:1543`). The
  audit trail is not itself a leak.
- **The clean `content` is never overwritten.** Redaction writes a *second* field,
  `masked_content`, and leaves `content` byte-identical (`contracts.py:1112`). Which of the two
  a downstream stage reads is a decision, made once, in one place — see §6.

---

## 2 · Where the labels come from

Both extraction and redaction are **zero-shot**: the model is handed a list of natural-language
type names at inference time and scores candidate spans against them. Nothing about *which*
types is compiled in. So the first question in this chapter is where that list comes from, and
the five campaign pipelines answer it two different ways.

### 2.1 The split, and why it falls where it does

| Pipeline | Corpus | `induce` stage | Entity labels handed to the extractor |
|---|---|---|---|
| `multihop_rag` | text-native news | **absent** | 7 hand-authored, on every chunk |
| `wiki2multihop` | text-native Wikipedia | **absent** | 7 hand-authored, on every chunk |
| `ohr_bench` | OCR'd documents | `label_inducer.llm` | 2-label floor + whatever *that chunk* induced |
| `vidoseek` | OCR'd documents | `label_inducer.llm` | same |
| `uda` | OCR'd financial/academic PDFs | `label_inducer.llm` | same |

The split is not per-pipeline taste. It is one line in the generator: a dataset induces its
schema if and only if it went through OCR
(`packages/latence-benchmark/src/latence_benchmark/campaign/ocr_stage.py:64` lists
`ohr_bench`, `vidoseek`, `uda`), and the generator branches on exactly that
(`packages/latence-benchmark/src/latence_benchmark/campaign/s3_lib.py:312`).

The reasoning is legible once stated. For a Wikipedia or news corpus you already know what is
in it — people, organisations, places, works, dates — and a hand-authored list is not a
guess, it is knowledge. For a pile of scanned enterprise PDFs you do not, and a hand-authored
list is a guess dressed as a schema: it will name types the corpus does not contain (paying
zero-shot precision for nothing) and miss the ones it does.

The two hand-authored sets are the whole schema for their pipelines:

| Slot | Labels |
|---|---|
| `labels` (entities) | `PERSON`, `ORGANIZATION`, `LOCATION`, `WORK`, `EVENT`, `DATE`, `PRODUCT` |
| `relation_labels` | `located_in`, `part_of`, `member_of`, `created_by`, `founded_by`, `spouse_of`, `directed_by`, `works_for`, `released_in` |

The three induced pipelines carry a deliberately *tiny* config list — `person`,
`organization`, and the single relation `works_for`
(`s3_lib.py:62`). That is not an oversight and it is not a floor in the usual sense. Which
brings us to the one idea this chapter turns on.

### 2.2 The asymmetry: relevance labels are scoped, safety labels are not

`latence_core.induced` is a pure function over two lists — a Provider's config labels and a
record's induced labels — and one policy enum that decides what the *config* list means
(`packages/latence-core/src/latence_core/induced.py:61`).

| `LabelFloor` | Effective label set for a chunk that induced types | Used by |
|---|---|---|
| `PER_CHUNK` | the chunk's **own induced set alone**; the config list is a recall net used *only* when induction produced nothing | `fused_entity_relation.gliner2` (default, `provider.py:160`) |
| `UNIVERSAL` | `config ∪ induced`, deduped case-insensitively, config order first | `redaction.gliner2` (pinned, `provider.py:144`); the function's own default |

Read that as one sentence: **entity and relation types are relevance judgements, so they are
scoped to the chunk that evidenced them; PII and guardrail types are security invariants, so
they are stamped on every chunk regardless of what that chunk "is about".** A leaked card
number in a chunk about shipping logistics must still be masked. A person type in a chunk
about shipping logistics is genuinely just noise.

The redactor does not merely default to `UNIVERSAL` — it *refuses* to be configured otherwise.
Setting `label_floor: per_chunk` on `redaction.gliner2` is a loud `ConfigError`, not an ignored
key (`packages/latence-pii-gliner2/src/latence_pii_gliner2/provider.py:382`), because a
silently-ignored knob leaves an operator believing they narrowed a control they did not.

Two consequences worth stating plainly:

- **`PER_CHUNK` forfeits recall insurance, on purpose.** A chunk whose induction under-called
  and missed `person` will not be scored for `person`, because the config floor no longer
  re-inflates it. The code says so in as many words (`extract-gliner2/provider.py:150`). The
  trade is bought back downstream: type-vocabulary drift (`org` here, `organization` there) is
  reconciled by corpus-wide canonicalisation in chapter 3, not prevented upstream by paying
  recall for it.
- **A pipeline with no `induce` stage is byte-identical under either policy.** A record with no
  induced labels falls through to the config list either way (`induced.py:175`). That is what
  makes the policy safe to change.

### 2.3 The LLM inducer: the one stage that can call out

`label_inducer.llm` is the only Provider in this chapter that opens a network connection.
Everything else runs in-process on the operator's own hardware. Corporate readers should read
this subsection closely and §7 for the summary.

**What it sends.** One chat-completions request per unit of induction, containing exactly two
messages: a fixed system prompt and a user message holding **the chunk's text verbatim**
(`packages/latence-schema-inducer/src/latence_schema_inducer/provider.py:899`). There is no
sampling, no summarisation, no redaction in between — the raw chunk goes to the endpoint. That
is unavoidable for the task and it is the reason the substrate is a config line: the same
Provider talks to OpenAI, to OpenRouter, or to a vLLM box inside your own network, because all
three speak the same API and only `base_url` changes (`provider.py:184`).

**What it asks for.** A single JSON object of exactly three arrays:

```json
{"entity_types": ["..."], "relation_types": ["..."], "pii_types": ["..."]}
```

The system prompt is engineered against that shape and asks for *salient, proper-noun-bearing
categories actually evidenced in the text*, explicitly forbidding invented types for generic
nouns, and preferring a small high-value set over an exhaustive one
(`packages/latence-schema-inducer/src/latence_schema_inducer/prompts.py:18`).

**What happens when it lies.** The reply is treated as untrusted output, and there are seven
layers between it and a label reaching a model. None of them can crash the run.

| # | Net | Failure it catches | On failure |
|---|---|---|---|
| 1 | `response_format={"type":"json_object"}` | free prose | — |
| 2 | truncation detect (`finish_reason == "length"`) | reply cut mid-JSON | one retry at **double** `max_tokens`, then fall back |
| 3 | parse-and-repair | malformed JSON | peel a Markdown fence or surrounding prose, re-run the **strict** parser; then one corrective re-prompt; then fall back (`provider.py:1066`) |
| 4 | schema validation | wrong shape (missing slot, list of ints, string not list) | reject → fall back (`provider.py:1090`) |
| 5 | label sanitisation | control chars, over-long labels, duplicates, count overflow, **value-shaped strings** | drop + tally, never silently (`provider.py:992`) |
| 6 | empty-induction recovery | parsed and valid but sanitised to nothing | one sharper re-prompt at double budget, then fall back **loudly** (`provider.py:737`) |
| 7 | fail-open | anything at all — network, auth, terminal 4xx | `induced_labels` stays `None`, the extractors use their config labels, the run continues (`provider.py:761`) |

Net 3 is worth a note for its restraint: it looks in more *places* for a JSON object (inside a
fence, between the first `{` and the last `}`) but it never loosens the parser. Every candidate
still goes through strict `json.loads`, so genuine noise still fails. And nothing anywhere
`eval`s model text.

Net 5 is worth a note for what it learned from a live run. The sanitiser's value-shape guard
(`sanitize.py:177`) rejects candidates that are instances masquerading as types — `1q22`,
`10.3`, `1024 fulton street`, `october 27, 1986` — using a digit-ratio threshold of `0.4` plus
fiscal-quarter, ISO-date, numeric-range, month-token, address-suffix and measurement-head
patterns. Two borderline cases are pinned in the source as tests of the threshold:
`2008 farm bill` is **kept** (ratio 0.33), `1933 act` is **rejected** (0.57). The guard is
opt-in and only the induced path opts in — a hand-authored config label is never shape-checked,
because an operator's deliberate type name is not a hallucination.

**Cost, and how it is bounded.** Per-chunk induction on a 220,129-chunk corpus (`uda`, in the
August 2026 run) is a lot of calls, so three layers sit in front of the fan-out
(`provider.py:638`):

1. **Text dedup** (`dedup: true` by default) — jobs collapse to distinct texts in first-seen
   order; one induction per distinct text; duplicates receive the identical `InducedLabels`.
   Lossless at temperature 0, and the single largest lever on a real corpus full of
   boilerplate.
2. **A call budget** (`max_induction_calls`) — the campaign sets 15,299 / 2,677 / 242,141 for
   the three datasets: each dataset's measured chunk count plus 10% headroom. Jobs past the budget make no call and fail open, setting
   `induction_budget_exhausted` and emitting **one** aggregate warning. The budget cuts a
   deterministic *input-order prefix*, never "whichever finished first", so a budgeted run is
   still byte-reproducible.
3. **Bounded concurrency** — `asyncio.Semaphore(max_concurrency)`, 32 in the campaign, with
   `gather(return_exceptions=True)` so one item's failure fails open for that item only.
   Results are reassembled by index, so completion order cannot perturb the output.

Cost is *reported*, not estimated: `induction_calls`, `induction_cache_hits`,
`induction_prompt_tokens` and `induction_completion_tokens` are read defensively off each
response's `usage` block, and an endpoint that reports no usage leaves them at zero — an honest
"unknown" rather than a fabricated figure (`provider.py:949`).

Deliberately absent: a *token* budget. Which jobs a mid-flight token budget would cut depends
on completion order, and that would make the run's output non-deterministic. Tokens are
reported; only calls are enforced (`provider.py:222`).

**Granularity.** Three modes, and the campaign uses the first:

| `granularity` | Unit | Effect |
|---|---|---|
| `chunk` | one call per distinct chunk text | each chunk keeps its own small set; no salience cap; a type appearing in one chunk survives on that chunk |
| `doc` | one call per document | the document's schema is broadcast to all its chunks |
| `corpus` | per-document, then merged | one schema for everything, ranked by **document frequency** and capped at `max_labels`, ties broken by label ascending |

The corpus mode's ranking detail is a bug fix worth knowing: capping an *alphabetically* sorted
union silently discarded a corpus's most explicit high-frequency relations (`obsoletes`,
`updates`, `references` — all sorting late) in favour of alphabetically-early rare noise.
Ranking by document frequency keeps what the corpus most agrees on (`provider.py:600`).

**The domain hint.** An operator can supply `domain: "..."` free text, which is folded into
every call as a fenced `<domain_context>` block explicitly framed as reference material, not
instructions. It is untrusted text and is treated as such: `sanitize_domain` strips control
and format characters, **removes `<` and `>`** so the hint cannot forge or close its own fence,
collapses it to one line and caps it at 512 characters (`sanitize.py:272`). Every downstream
net still applies, so a hint that tries to redirect the model still cannot escape the JSON
contract.

### 2.4 The zero-network alternative

`label_inducer.lexicon` fulfils the same Capability with **no model, no calls and no network**
(`packages/latence-schema-inducer/src/latence_schema_inducer/lexicon.py:135`). It takes an
operator-declared candidate type space — each type with a list of surface cues — and induces a
type for a chunk when at least `min_hits` **distinct** cues match (repeats of one cue count
once, `lexicon.py:125`). Cues are `re.escape`d, so a config file is never a regex surface;
matching is whole-word and case-insensitive by default.

It is `deterministic=True` with `cost_per_1k=0.0`, and it deliberately offers no `corpus` mode.
It emits the same `InducedLabels` carrier with `model_id="lexicon"` — naming the *method*, not
a checkpoint. Its observability attributes mirror the LLM inducer's exactly, so a Quality Report
compares like with like. An empty or unusable candidate space is a `ValueError` at
**construction** — config errors are loud, run-time is fail-open.

The candidate space it consumes is what `latence-label-profiler` emits: that tool mines a
corpus sample and selects a type pool by greedy max-coverage. There is no code dependency
either way; the relationship is a file. Profiler = corpus-level pool, lexicon inducer =
per-chunk narrowing of that pool.

For an air-gapped deployment that still wants induced rather than hand-authored labels, this is
the whole answer.

---

## 3 · Fused entity and relation extraction

### 3.1 Why one model instead of two

`fused_entity_relation.gliner2` runs `fastino/gliner2-multi-v1` — one mdeberta-v3 backbone doing
zero-shot NER *and* zero-shot relation extraction — and so a single Provider fulfils two
Capabilities ([ADR-0013](../adr/0013-relation-extraction-provider-chosen-empirically-fused-capability-allowed.md)).
The Runner routes its output to both downstream consumers; there is no separate entity stage.

The argument in the code is not "fewer models is tidier". It is a **coordinate-system**
argument. The native API returns entity spans as chunk-local `start`/`end`, and returns each
relation's `head` and `tail` as sub-dicts carrying *the same* chunk-local `start`/`end`
(`packages/latence-extract-gliner2/src/latence_extract_gliner2/provider.py:44`). So a relation
is tied to the two mentions it links by exact `(start, end)` identity — a dictionary lookup,
not a fuzzy alignment:

```python
# packages/latence-extract-gliner2/src/latence_extract_gliner2/provider.py:1010
def _resolve_endpoint(rel, role, by_span):
    endpoint = rel.get(role)
    ...
    key = (int(endpoint["start"]), int(endpoint["end"]))
    return by_span.get(key)
```

Two separate models cannot give you that. A standalone relation extractor emits head/tail as
*text*, and matching text back to mentions is where duplicate surfaces and boundary
disagreements turn into wrong edges. Here, a relation whose endpoint is not among the emitted
mentions is simply **dropped** (`provider.py:922`) — no un-groundable record leaks into the
graph.

What it costs: you buy both label sets from one checkpoint's zero-shot quality, and you lose the
ability to pick the best entity model and the best relation model independently. The framework
keeps that door open — see §3.6 — but the blessed stack takes the fused deal.

One requirement is non-negotiable: `include_spans=True` is always passed
(`provider.py:792`). Without it gliner2 returns entity *strings* and relation head/tail *text*
only, which the framework contract cannot ground. It is not a knob.

### 3.2 The forward pass: bucket, micro-batch, scatter

Per-chunk induction gives every chunk its own label set, but one batched forward takes **one**
schema. The Provider resolves the shape of that problem in three steps
(`provider.py:692`):

1. **Buffer** a streaming window of `batch_size` chunks (default 16, hard-clamped to 64 so a
   large corpus always logs progress often enough to never *appear* hung).
2. **Bucket** that window by label set — separately for the entity slot and the relation slot,
   since a chunk can share one and not the other. The key is a `tuple[str, ...]`, because label
   **order** is part of the question asked of the model; two chunks share a bucket only if they
   would have been asked the identical question (`provider.py:163`).
3. **Micro-batch** each bucket by a token budget (`microbatch_max_texts` default 4,
   `microbatch_token_budget` default 3072 = 4 × 768) and run one native forward per
   micro-batch.

Results are scattered back **by index**, never by output order, so bucketing can never permute
one chunk's result onto another. The emitted records are bucket- and batch-invariant — a test
drives a per-chunk reference and asserts record-for-record equality.

This matters more than it looks. The span-scoring activation is `batch × positions × LABELS`,
and a measured ~17 GiB spike was `batch × ~768 tokens × ~40` corpus-union labels. Per-chunk
label sets collapse the label dimension so the tensor never forms; the reactive OOM backoff
below is now the last-resort net rather than the cure. It also unlocked `torch.compile`: gliner2
compiles with `dynamic=True`, and a fresh `(label-count, batch)` shape family per chunk used to
thrash dynamo past its recompile limit — worker pinned at ~290% CPU with the GPU at 0%.
Bucketing pins the label dimension, micro-batching pins the batch dimension, so dynamo sees one
shape family per bucket.

### 3.3 The threshold

`threshold: 0.5` is passed straight to the native call and gates **both** slots — the same value
for entities and relations (`provider.py:791`). It is also the code default, so the campaign
turned no knob here.

What it does in practice:

- **Below it, spans are dropped by the model**, not by the framework. There is no post-hoc
  filter to relax.
- **Precision/recall.** Raising it yields fewer, cleaner mentions; lowering it yields more
  mentions of lower confidence. The confidence survives onto every record
  (`EntityMention.confidence`), so a downstream consumer can filter harder — but it can never
  recover a span the model already dropped.
- **Graph density is the real lever it moves.** Every accepted mention is a candidate node, and
  the relation slot compounds it: a relation survives only if **both** its endpoints cleared the
  threshold. Lowering it therefore grows the edge set super-linearly, and a denser graph is not
  automatically a better one — chapter 3 resolves and assembles whatever this stage admits.

A relation's own score is not reported by the model at the relation level. The Provider derives
it as the **weakest link** — the minimum of its head and tail confidences — falling back to 1.0
when neither endpoint carries one, while still honouring a top-level `confidence` if a future
version adds one (`provider.py:983`).

### 3.4 How offsets survive

This is the recurring coordinate trap
([ADR-0031](../adr/0031-entity-extraction-emits-true-original-offsets-via-carried-strip-offset-map.md)),
and the chapter-1 chunker is what makes the fix possible.

The model returns offsets into the chunk's `content`, which is **markup-stripped**. So
`chunk.char_start + local` is only a *lower bound* on the true original offset. Each chunk
carries an `offset_map` (stripped → original) and the Provider resolves through it:

```python
# packages/latence-extract-gliner2/src/latence_extract_gliner2/provider.py:870
if offset_index is not None:
    doc_start, doc_end = offset_index.resolve_span(local_start, local_end)
else:
    doc_start, doc_end = chunk_char_start + local_start, chunk_char_start + local_end
```

`resolve_span` resolves the exclusive end from the **last included character** plus one
(`packages/latence-core/src/latence_core/offsetmap.py:68`), so a span whose interior contained
stripped markup still covers the full original extent. The `else` branch is the honest
degradation for a pre-v11 chunk carrying no map.

Pages are resolved separately, through a `PageIndexResolver` that reads each chunk's **own**
`page_slice` — so a mention on the third page of a multi-page chunk cites page three, not the
chunk's first page. One resolver serves the whole `extract_fused` call, so the drift roll-up is
per call. A chunk with no page spans **raises** rather than returning a plausible-looking wrong
page (`packages/latence-core/src/latence_core/page_index.py:139`).

A relation's covering span crosses two chunks, so no single chunk can answer for it. It does not
need to: both endpoints already carry exactly-resolved pages, and the covering span's pages are
their `min`/`max` — arithmetic on two records, no lookup (`page_index.py:72`).

### 3.5 What goes wrong, and what the Provider does about it

| Fault | Where it surfaces | Recovery |
|---|---|---|
| `gliner2` not installed | first use | `ImportError` naming `pip install latence-extract-gliner2` (`provider.py:518`) |
| unsupported/forked `gliner2`, or a cloud `from_api` client | **load**, before any forward | typed `ProviderError` naming the supported range `>=1.3,<2` (`provider.py:229`) |
| unexpected output shape (flat list instead of the nested dict) | per micro-batch | typed `ProviderError` — never a silent zero-span result (`provider.py:209`) |
| `torch.compile` backend failure (compiles lazily, so it fails *inside* the first forward) | inference | reload once with `compile=False`, quantize kept, retry; throughput only (`provider.py:568`) |
| CUDA OOM | inference | split the micro-batch in half and retry each after a cache release; a batch of **one** that still OOMs surfaces as a typed error — no dropped chunk, no fake green (`provider.py:761`) |
| degenerate zero-length span | record building | skipped (an `EntityMention` with `text=""` violates its ≥1-char contract), **counted and logged** (`provider.py:865`) |
| `device: cuda` on a host with no GPU | construction | loud `ConfigError`, not a silent CPU fallback (`provider.py:528`) |

That last row has history. This Provider once probed CUDA availability directly and never read
`config["device"]` at all — so `device: cuda` in the blessed stack was a no-op and, worse,
`device: cpu` on a GPU box was silently overridden. Resolving the device once, at construction,
through the shared seam is what makes the YAML's `device: cuda` mean something.

The whole error posture is one rule
([ADR-0034](../adr/0034-observability-typed-errors-and-thin-tracing-seam.md)): a native fault
becomes a typed `ProviderError` naming the operation and the native exception type **only** —
never the chunk text, so an error log can never become a data leak.

### 3.6 Swapping the extractor

The Capability is the seam. Anything satisfying `FusedEntityRelationExtractor` (or the entity
and relation halves separately) drops in by changing the `provider:` line.

| Provider | Package | Model | Licence (code / weights) | Notes |
|---|---|---|---|---|
| `fused_entity_relation.gliner2` | `latence-extract-gliner2` | `fastino/gliner2-multi-v1` | Apache-2.0 / Apache-2.0, verified 2026-07-10 | the configured one; also exposes `extract()` for the NER half alone |
| `relation.gliner_relex` | `latence-relation-gliner` | `knowledgator/gliner-relex-multi-v1.0` | Apache-2.0 / Apache-2.0, verified 2026-07-06 | the fused Provider gliner2 superseded; identical config shape and identical `(start,end)` endpoint resolution, so it is a one-line swap |
| `entity.gliner` | `latence-ner-gliner` | `urchade/gliner_multi-v2.1` | Apache-2.0 / Apache-2.0, verified 2026-07-06 | entities only; no bucketing or micro-batching — `batch_size` **is** the forward; always unions config ∪ induced |
| `relation.llm` | `latence-relation-llm` | any OpenAI-compatible endpoint (default id `gpt-4o-mini`) | Apache-2.0 client, **no weights** | relations from mentions + text; **makes network calls** |
| `entity.gazetteer` | `latence-core` | none | Apache-2.0 | `deterministic=True`; literal terms + operator regexes, with a static ReDoS guard that rejects catastrophic-backtracking patterns at construction |
| `relation.pattern` | `latence-core` | none | Apache-2.0 | `deterministic=True`; trigger/pattern match in a `window_chars: 200` gap between two mentions, with fan-out caps and audit tallies |

The zero-model pair at the bottom is not a toy. `entity.gazetteer` + `relation.pattern` give a
fully deterministic, CPU-only, dependency-free extraction path — which is what makes a run
byte-reproducible when that is the requirement.

`relation.llm` deserves one line of caution for the same reason as the inducer: it sends
document text to an endpoint. Its bad-output posture is the same family — a `JSONDecodeError`
returns an empty relation list rather than crashing, and `_valid_relations` drops out-of-range
indices, self-relations, duplicates and unknown labels — but the egress is real.

---

## 4 · Redaction: a control that runs before anything leaves

### 4.1 It runs per chunk, and that is a fix

Redaction was once document-level: the whole assembled markdown was fed into a fixed 768-token
model window, and **any PII past that window was silently truncated and leaked into the corpus
unmasked**. Making redaction a chunk → chunk transform is what closed it
([ADR-0042](../adr/0042-chunk-level-redaction-fixes-the-document-truncation-leak.md)). Each
chunk is ≤ the window, each chunk is scanned in full, nothing is truncated.

The document-level path still exists as `redact_documents` and is marked deprecated
(`packages/latence-core/src/latence_core/stages/redaction.py:539`).

### 4.2 The configured control

```yaml
- name: redact
  capability: redaction
  provider: redaction.gliner2
  config:
    pii_labels: [person, email, phone number, address,
                 credit card number, social security number]
    guardrails: true
    device: cuda
```

Six PII labels, natural-language and lower-case because GLiNER2-family zero-shot models score
natural-language type names best. The checkpoint is
`fastino/GLiNER2-Guardrails-PII-Multi` — the same engine family as the extractor, so PII
detection adds no second model library to the environment.

The batching machinery is the *same shared primitive* the extractor uses — `iter_label_buckets`
and `iter_token_budget_microbatches` from `latence_core.providers.perf`, imported by both — so
the two Providers cannot drift on bucket order or index order
(`packages/latence-pii-gliner2/src/latence_pii_gliner2/provider.py:770`).

### 4.3 What `guardrails: true` turns on

The checkpoint is a **guardrails+PII** model, so with the flag on, guard categories ride the
*same* forward pass as the PII types — one call covers both
(`pii-gliner2/provider.py:726`). The universal guard floor is two labels
(`packages/latence-core/src/latence_core/redaction_policy.py:411`):

| Label sent to the model | `RiskMarker.category` recorded |
|---|---|
| `prompt injection` | `prompt_injection` |
| `harmful content` | `harmful` |

Four things about this are worth reading carefully:

- **A guard hit is flagged, never masked.** It marks the chunk's *content* as risky and emits a
  `RiskMarker` that survives into the exported corpus so a RAG consumer can exclude the chunk.
  Masking it would silently corrupt the text (`pii-gliner2/provider.py:815`).
- **The category vocabulary is deliberately the same one `screening.content_keyword` emits**, so
  a consumer filters one set regardless of which control caught it.
- **A marker never echoes the matched text.** Its `reason` carries the category and the hit
  count and says so explicitly — a marker that quoted the injected string would re-publish
  exactly what was flagged (`pii-gliner2/provider.py:905`).
- **A label claimed by both sets is read back as a guard hit.** Dropping a security screen is
  worse than losing one mask. A config-level collision between `guard_labels` and `pii_labels`
  is already a `ConfigError`, so this only arbitrates *induced* overlap.

`guard_labels` supplied without `guardrails: true` is a `ConfigError`, never a silently ignored
list (`pii-gliner2/provider.py:432`).

One honest gap, verified in the source: **no shipped inducer produces `guard_types`.** Both
`label_inducer.llm` and `label_inducer.lexicon` emit only the three slots
`entity_types` / `relation_types` / `pii_types` (`schema-inducer/provider.py:156`). The
`InducedLabels.guard_types` slot and the redactor's read of it are wired and tested, but nothing
in the current roster fills them, so in practice the guard set is exactly the two-label floor
plus any operator `guard_labels`.

### 4.4 The floor the model cannot lose

A learned PII model can miss. So a small set of high-harm financial and national-identity types
is recognised **deterministically**, with zero dependencies, at the shared choke point every
Redaction Provider funnels through
([ADR-0044](../adr/0044-universal-redaction-floor-and-optional-mask-before-extract.md)):

| Detector | Reported `pii_type` |
|---|---|
| `ssn` (`NNN-NN-NNNN`) | `ssn` |
| `ssn_nodash` (`NNNNNNNNN` / `NNN NN NNNN`, with a plausible-SSN guard and a letter-glue guard) | `ssn` |
| `credit_card` (13–16 digits, optional separators) | `credit_card` |
| `iban` | `iban` |

Because the floor runs in `finalize_redacted_chunk`
(`redaction_policy.py:569`) rather than inside any one Provider, it is **non-bypassable by
model or backend choice** and impossible to forget when writing a new Redactor. Before this,
only the in-core rule Provider recognised these types and the three learned ones inherited only
their model's coverage.

The guarantee is deliberately narrow and stated as such: the floor guarantees *detection*
regardless of backend. It does **not** override an operator's explicit `action` or `skip`. A
floor span is handled with the chunk's resolved policy action, and the floor runs only when
redaction actually runs on the chunk. The patterns are byte-identical to the `hybrid_rule`
built-ins (a drift-guard test pins them equal), so a `hybrid_rule` stack is byte-identical:
every floor span coincides with a Provider span and the Provider wins the tie, keeping its
richer provenance.

The `ssn_nodash` letter-glue guard is a good example of what a live corpus teaches — and of what
that lesson is worth once the thing it taught turns out to have been a bug. The markdown chunker
used to weld table cells together, so `| Industrial drives | 410 | 455 | 470 |` reached the
detector as `Industrial drives410455470` — an innocent numeric row masked as an SSN. A fixed-width
negative lookbehind rejects a digit run fused to a Unicode letter, with an explicit exception for
the one letter-glued form that *is* a real presentation: `SSN123456789`
(`redaction_policy.py:347`).

The weld is gone (chapter 1, pattern 13). The same fix showed that the guard had been doing more
than sparing innocent rows: on a welded `Ada Lovelace123 45 6789` it refused a *genuine* SSN, so
tabular PII went unmasked while the identical value in a sentence was masked. The guard was
re-measured against the fix rather than deleted along with its original motivation, and it stays,
on two grounds that were checked: it costs no recall now that cells arrive separated (a table
cell's SSN is masked exactly as the same value in prose), and it still buys precision on
letter-glued digit runs that have nothing to do with tables — part and lot codes like
`Lot 410455470A`, a label ending in a non-ASCII letter like `Menü410455470`.

### 4.5 How masking is represented

Nothing is destroyed. The redactor writes three fields onto the chunk and leaves `content`
untouched:

| Field | Contents |
|---|---|
| `masked_content` | the chunk text with every non-`TAG` span replaced by its placeholder |
| `pii_spans` | `list[PIISpan]` — type, detector, **chunk-local** offsets, action, placeholder, resolved **original** source page, confidence |
| `redaction_disabled_for_sensitive` | the loud "the control was OFF" bit — see below |

Four actions, resolved per document sensitivity (`redaction_policy.py:112`):

| Action | Placeholder | Property |
|---|---|---|
| `mask` (default) | `[EMAIL]` | lossy, irreversible |
| `replace` | `EMAIL_a1b2c3d4` | stable pseudonym; text stays readable |
| `hash` | `<EMAIL:a1b2c3d4e5f6>` | equal values stay joinable across the corpus without exposing them |
| `tag` | `""` | detect-only; the surface text is left intact |

All three non-`TAG` surrogates are derived from `sha256(salt:type:value)`. The value is mixed
into the digest and **never returned in the clear or persisted**, so a `replace`/`hash` corpus
is deterministically joinable without storing what was joined on. `PIISpan` refuses to validate
a non-`TAG` span with an empty placeholder (`contracts.py:1593`) — a masking action that wrote
nothing cannot be recorded as if it had.

Masking is applied **right-to-left** over the ordered non-overlapping span list, so replacing one
span never shifts the offsets of an earlier one (`redaction_policy.py:133`).

**Policy is metadata-aware.** The `policies` map keys off each document's
`Classification.sensitivity` and can set a per-sensitivity `action`, a `types` allow-list, or
`skip: true`. And because "an operator turned it off" and "there was nothing to find" both
produce an empty span list, the framework distinguishes them: a chunk whose classification is in
the sensitive set (`confidential`, `restricted` by default) *and* whose applied policy would run
no detector at all carries `redaction_disabled_for_sensitive = True`
(`redaction_policy.py:249`). The floor does not force redaction on — an operator may
intentionally skip — it only makes the disposition first-class and auditable.

**Overlaps clip, they never drop.** Two span sets get unioned at the choke point: the floor with
the Provider's spans, and this stage's spans with any an upstream Redaction stage already
applied. The naive greedy rule — accept if `start >= last_end`, else drop — *un-masks* text: a
span overlapping an accepted one by a single character is discarded wholesale, so the part
nothing else covered is exported verbatim (an audit found 12 digits of an already-masked card
re-exposed this way). Clipping to the covered prefix instead makes the resolution
**coverage-preserving**: every character covered by any input interval stays covered
(`redaction_policy.py:433`). A companion rule falls out of the same fix — masking spans outrank
`TAG` spans, because `TAG` is the one action `apply_masking` skips, and an earlier-starting
`TAG` span must never displace a real mask.

### 4.6 Swapping the redactor

| Provider | Package | Engine | Licence (code / weights) | Deterministic |
|---|---|---|---|---|
| `redaction.gliner2` | `latence-pii-gliner2` | `fastino/GLiNER2-Guardrails-PII-Multi` | Apache-2.0 / Apache-2.0, verified 2026-07-09 | no |
| `redaction.gliner_pii` | `latence-pii-gliner` | `urchade/gliner_multi_pii-v1` | Apache-2.0 / Apache-2.0, verified 2026-07-06 | no |
| `redaction.presidio` | `latence-pii-presidio` | Microsoft Presidio `AnalyzerEngine` over spaCy `en_core_web_sm` | MIT / MIT, verified 2026-07-08 | no |
| `redaction.hybrid_rule` | `latence-core` | regex + gazetteer, zero-dep | Apache-2.0, no weights | **yes** |

All four share policy resolution, span building, placeholder derivation, masking, page
resolution and the financial floor — a Provider owns **only detection**. That is why the swap is
one line and why a new Redactor cannot forget the floor.

`redaction.hybrid_rule` is the deterministic reference and its coverage is stated honestly
rather than sold as "comprehensive": email, SSN (both forms), credit card, phone, IP, IBAN, AWS
access-key **id**, and a set of prefix-anchored modern secret formats (GitHub, Stripe, OpenAI,
Google, Slack, GitLab tokens, PEM private-key blocks, password-bearing connection strings, JWTs)
plus labelled-assignment secrets. Its documented boundary is equally explicit: the **bare**
40-character AWS *secret* access key is deliberately not recognised, because a bare high-entropy
blob is high-false-positive; only the labelled `aws_secret_access_key=…` context is
(`stages/redaction.py:33`). Every pattern is linear-time by construction — the email recogniser
carries a paragraph explaining why a naive `\b[local]+@[domain]+` is O(n²) on a dotted run and
how a negative lookbehind plus a possessive quantifier fixes it (`stages/redaction.py:121`).

---

## 5 · Profiling

`profiling.statistical` is the fourth stage in this segment and the least dramatic. It reads
parsed documents **and** the extractor's mentions and streams out `FeatureRecord`s: one
`DOCUMENT`-scope record per parsed document, then exactly one `CORPUS`-scope record, emitted
last, over the folded aggregate (`packages/latence-core/src/latence_core/stages/profiling.py:130`).

Per document: character/word/sentence counts, unique words, type-token ratio, density,
readability, Zipf alpha, compression ratio, average word and sentence length. Corpus-wide:
folded word frequencies, language/category/sensitivity distributions, entity frequency and
entity co-occurrence.

It is pure Python, `compute="cpu"`, `deterministic=True`, no weights. Three caps keep it bounded
on a large corpus rather than letting a large NER run explode it:
`max_co_occurrences: 50`, `max_entities_per_document: 200` (so one document cannot build an
O(distinct-entities²) pair set), and `max_tracked_pairs: 100_000`. A `PARSE_ERROR` document
contributes to neither per-document nor corpus statistics.

The one non-obvious detail: because it consumes `mentions`, it depends on `extract` — which is
why the DAG shows `parse` and `extract` both feeding `profiling`.

---

## 6 · `extract_on: unmasked` — the flag that decides what the graph knows

Every campaign pipeline carries this at the top level:

```yaml
extract_on: unmasked
```

It is `ExtractOn.UNMASKED` by default (`packages/latence-core/src/latence_core/pipeline.py:197`),
so writing it changes nothing — it is written to make the posture explicit rather than implied.

### What it actually does, in code

Exactly one thing, in one place:

```python
# packages/latence-core/src/latence_core/runner.py:1853
masked = pipeline.extract_on is ExtractOn.MASKED
for chunk in chunks:
    if masked and chunk.masked_content is not None:
        yield chunk.model_copy(update={"content": chunk.masked_content})
    else:
        yield chunk
```

In `masked` mode, each chunk handed to an extraction stage gets its `content` **replaced** by
its `masked_content`. The extractors read `content` and know nothing about the substitution.
That is the entire mechanism. A chunk without `masked_content` falls back to its own `content`,
so the mode never fabricates or drops text.

It applies to precisely three Capability kinds — `entity_extraction`, `relation_extraction`,
`fused_entity_relation` (`pipeline.py:131`) — because they, and only they, read chunk text to
find mentions. Redaction, screening and export are untouched.

### Why it is a flag and not a default

The two postures answer different questions:

| | `unmasked` (default) | `masked` |
|---|---|---|
| What the extractor sees | the real chunk text | `[PERSON]`, `[EMAIL]`, … |
| What the KG contains | real entity names — `canonical_name` is the actual name | placeholders, or nothing where a name was |
| PII reaches the KG? | **yes** | no |
| Exported corpus text | `masked_content` (unless explicitly overridden) | `masked_content` |

`masked` is the stricter posture some regulated deployments require: PII never reaches
extraction, so it never reaches the knowledge graph, so it cannot be reconstructed from the
graph even if the corpus is perfectly masked. The cost is stated without hedging — a knowledge
graph of `[PERSON]` nodes is a knowledge graph about nobody. Disambiguation can only resolve
what extraction still sees, so it degrades in proportion to how much of the corpus was masked.

Defaulting to `unmasked` is therefore a real decision, not an oversight, and the framework
refuses to let `masked` fail quietly. Turning it on without wiring the DAG for it is a
**structural error at Pipeline validation**, not a silent no-op:

- no redaction stage at all → `ValueError` naming the problem;
- an extraction stage that does not transitively depend on a redaction stage → `ValueError`
  telling the operator to route `depends_on` through `redact`
  (`pipeline.py:220`).

Without that check the mode would silently read unmasked text — a privacy hole that looks
exactly like a working configuration. And in the guided setup, `extract_on: masked` with PII
redaction off is rejected up front (`setup/recipe.py:717`).

### One caveat specific to the five committed pipeline files

The generated header comment on every campaign YAML reads:

> `extract_on unmasked (extraction reads raw chunks; masking applies to the exported corpus)`

For a pipeline the generator writes today, both halves hold: the generator does not set
`unsafe_unmasked_corpus`, so the corpus export writes `masked_content`.

**The second half is not true of the five files committed under
`benchmark/sota-campaign/stages/s3-pipeline/pipelines/`.** They are the record of the August 2026
run, and each one sets `unsafe_unmasked_corpus: true` on `export_corpus`, which makes the
exported and embedded text the raw `content` as well. So in that run masking applied to
*neither* extraction *nor* the exported corpus; only the detection evidence was kept, in
`pii_spans` and `masked_content`. The reason is recorded in those files as a comment: with
masking on, 82% of 2Wiki/MultiHop-RAG rows carried placeholders, including the article titles
the questions target. That was measured on the earlier pipeline, for a retrieval benchmark that
was removed by
[ADR-0068](../adr/0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md).
With the flag on, the export logs a warning once per run naming the risk.

---

## 7 · What leaves the process

For a regulated deployment, the summary is short.

| Stage | Leaves the process? | What crosses the boundary |
|---|---|---|
| `induce` (`label_inducer.llm`) | **yes**, one HTTPS request per distinct chunk text | the chunk text verbatim, plus the fixed prompt, to whatever `base_url` names |
| `induce` (`label_inducer.lexicon`) | no | — |
| `extract` (`fused_entity_relation.gliner2`) | no | in-process forward on the operator's own hardware |
| `redact` (`redaction.gliner2`) | no | in-process forward |
| `profiling` | no | pure Python |
| every other Provider in this chapter | no | — |

Two notes that matter more than they look:

**The extraction and PII models ship no server, and the code says so rather than pretending
otherwise.** gliner2 has exactly one server surface — `GLiNER2.from_api`, a *client* for
fastino's cloud API — which is disqualifying for an on-prem framework. All the throughput here
is the native batched forward plus `compile` and `quantize`, running in-process
(`extract-gliner2/provider.py:38`). The `_require_batched_api` load check exists partly to catch
an api-client object being handed in where a local model belongs.

**Running with no egress at all is a config change, not a fork.** Three routes: drop the
`induce` stage (both hand-authored pipelines already do, and every extractor falls back to its
config labels byte-identically); or point `base_url` at a vLLM box inside your own network; or
swap `label_inducer.llm` for `label_inducer.lexicon`, which has no network path to configure.

And the no-content discipline runs through every log line in this chapter: a native fault names
the operation and the exception type, never the text; an induction fallback logs the safety-net
reason and the model id, never the document; a guardrail `RiskMarker` records the category and
the hit count, never the matched string.

---

## Where this leaves us

Out of this segment come three streams: `EntityMention`s and `RelationMention`s with real
document offsets and real page numbers, and `ChunkRecord`s carrying their masked variant and
their audited PII spans. Both mention streams are still **per-chunk observations** — the same
organisation named in forty chunks is forty mentions, and if those chunks induced their own
labels it may be `organization` in one and `org` in the next.

Reconciling that is the corpus phase's job, and it is where the pipeline stops looking at one
chunk at a time. Chapter 3, `03-corpus-to-knowledge-graph.md`, picks up exactly here: type
consolidation canonicalises the drifted vocabulary, disambiguation merges mentions into single
canonical entities, and graph assembly turns the surviving relations into evidence-linked edges.
