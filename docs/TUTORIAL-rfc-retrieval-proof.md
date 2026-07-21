# Tutorial — RFC corpus → AI-ready data → fused-retrieval proof (end-to-end)

**Who this is for.** A human operator *or* an AI agent, working with no prior context on this
project. Every step is a real command plus a **`✓ look for`** block stating exactly what a correct
result looks like. If a step's real output does not match its `✓ look for`, that is a **finding** —
stop, record it (step id + command + expected vs actual + suspected cause), and do not paper over it.

**What you will produce, in order.**
1. A multi-document **English RFC corpus** (various sources, genuinely multi-hop-hard).
2. An **AI-ready dataset** — the enterprise-SOTA pipeline output: OCR-clean chunks, induced schema,
   NER/RE, PII-safe redaction, disambiguated entities, a knowledge graph with centrality/community,
   dense embeddings, and a BM25 term-stats artifact — all as files (ADR-0017).
3. **Four retrieval indexes over that dataset**: dense vectors (Qdrant), **SPLADE-v3 sparse vectors
   (Qdrant-native)**, BM25, and **KG augmentation** (DuckDB multi-hop over the emitted graph parquet).
4. A set of **super-hard fused questions + gold chunk sets** that provably need multi-hop + lexical +
   semantic search *together*.
5. A **measured retrieval comparison** (dense-only vs BM25-only vs SPLADE-only vs full fused) with
   **charts**, packaged as a **decision-maker-facing report**.

**Ground truth beats any claim in this file.** Where a step's expectation here disagrees with what
the code actually does, the code is the truth and the disagreement is a finding.

**No fake green — non-negotiable.** Every external dependency in this run (Qdrant, the models, GPU
placement) must be exercised for real. A test passing against a fake/mock proves the fake, not the
product. The `FakeQdrantClient` is offline-CI-only and must never appear in this proof's green path.
If a real dependency cannot be stood up, that is a finding to fix — not a cue to fall back to a fake.

**The CLI wizard (`latence setup`) is the entry gate.** The pipeline is configured by running the
interactive wizard, not by hand-authoring YAML. If the wizard cannot produce a working, runnable stack
for this corpus, that is a top-priority finding — the wizard *is* the product's front door.

**This pod:** Blackwell RTX PRO 4500, 32 GB VRAM, no Docker. The RFC corpus is plain text, so the
LightOn-OCR vLLM endpoint is **not served** — `.txt` routes to the plaintext parser. That is correct
for this corpus and keeps the 32 GB free for the learned stack (Granite, gliner2, GLinker, SPLADE-v3);
schema induction uses an API key, not local VRAM.

**Two things this run deliberately carries, stated up front (see Appendix A/B):**
- **SPLADE checkpoint = `naver/splade-v3`, whose weights are CC-BY-NC-SA-4.0 (non-commercial).** This
  is a deliberate, operator-accepted choice for a *private internal proof*. It is **not** shippable as
  a default, and the exported sparse vectors inherit the NC + share-alike terms. Appendix A.
- **Qdrant-native sparse vectors are new wiring** (the sink historically carried sparse as payload
  only). Step 4.3 is the first exercise of that path; treat any gap there as an expected finding.
  Appendix B scopes the pre-work.

---

## 0. Prereqs

- A RunPod GPU pod, CUDA-13 / Blackwell-class, per `docs/TUTORIAL-fresh-pod-walkthrough.md` §1.
- The framework cloned at `/workspace/latence-framework`, on `main`.
- The enterprise-SOTA stack installed (`fresh-pod` §1.2) **plus** the retrieval extras:
  ```bash
  cd /workspace/latence-framework
  uv pip install -e "packages/latence-splade[local]"        # SPLADE signal generator (torch/transformers)
  uv pip install -e "packages/latence-sink-qdrant[client]"  # the Qdrant loader + client
  uv pip install -e "packages/latence-retrieval"            # fusion / rerank / bm25 / multi-hop / pack
  ```
  ✓ **look for** — three clean installs, no dependency-resolution error. A `transformers` version
  conflict here (gliner needs `<5`, some sparse stacks want `5.x`) is a **finding**: the SPLADE
  generator must resolve alongside the gliner extract/redact stack, or be isolated behind the
  ADR-0030 endpoint seam.
- Env from `fresh-pod` §1.3 (`PYTORCH_JIT=0`, the HF-Hub XET workaround, the induction key).

---

## 1. Assemble the RFC corpus

The corpus is built by a small fetch script that (a) downloads a pinned set of RFCs and (b) records
their `Obsoletes` / `Updates` / `References` edges as a sidecar the benchmark's gold-set builder reuses.

```bash
cd /workspace/latence-framework
python benchmark/rfc/build_corpus.py \
  --out /workspace/corpus-rfc \
  --seeds 9110,9111,9112,9113,9114 \
  --expand obsoletes,updates,references \
  --max-docs 80
```

✓ **look for**
- `/workspace/corpus-rfc/*.txt` — between 60 and 80 RFC text files, each with a real header.
- `/workspace/corpus-rfc/_refs.jsonl` — one row per RFC with `obsoletes`, `updates`, `references`
  arrays (the human-authored cross-reference graph, kept separate from any KG the pipeline induces —
  it is the **independent** gold signal, never fed into the pipeline).
- The manifest prints a connectivity stat: **median out-degree ≥ 3** and **at least 15 documents that
  are ≥ 2 hops from a seed**, with **no `WARNING: corpus is too shallow` line**. If connectivity is
  below that, the corpus is too shallow to prove multi-hop — widen `--max-docs`, add seeds, or raise
  `--ref-fanout`. (Why it matters: the German single-book run capped real multi-hop at ~2 chains
  because one document's induced KG is hub-heavy; a multi-document RFC graph is the fix, but only if it
  is actually deep. The HTTP-core seeds are reference *hubs* — RFC 9110 alone lists 67 references — so a
  naïve breadth-first fetch fills the whole `--max-docs` budget with depth-1 documents and never reaches
  depth 2; the builder caps how many `references` targets each document contributes to the fetch
  frontier — `--ref-fanout`, default 8 — so the frontier advances in depth. Obsoletes/updates chains are
  always followed in full, and `_refs.jsonl` still records every document's complete edge set.)

License note: RFC texts are published under the IETF Trust's free-use terms; the build script records
the exact terms in `/workspace/corpus-rfc/LICENSE-CORPUS.md`. Verify it exists before proceeding
(ADR-0012 / #181: no dataset enters a benchmark without its license recorded).

---

## 2. Configure + run the pipeline (enterprise-SOTA, GPU)

**The wizard is the entry gate — configure through it, do not hand-write YAML.** Follow
`docs/TUTORIAL-fresh-pod-walkthrough.md` **§3 (`latence setup` → `latence process`)** with
`{CORPUS_DIR}` = `/workspace/corpus-rfc`. **Skip fresh-pod §2 (serving LightOn-OCR vLLM):** the corpus
is `.txt`, so at the wizard's parser prompt choose the **plaintext** path, not the OCR VLM. That is
correct for text and leaves the 32 GB VRAM for the learned stack. If the wizard forces an OCR-VLM parse
on a text corpus (or otherwise cannot yield a runnable stack), that is a top-priority wizard finding.

The wizard emits the full SOTA spine (verified: #193/#194/#197):
`parse → chunk[640] → content_screen → induce → extract[gliner2] → redact[gliner2, financial-PII
floor] → profiling → disambiguate[glinker] → graph[+pagerank/community] → enrich(context) →
export_kg + export_corpus[Granite r2 dense + bm25 term-stats]`.

```bash
latence setup --profile enterprise-sota   # answer prompts; writes ./latence.stack.yaml
latence process                            # runs the whole pipeline
```

✓ **look for**
- `latence.stack.yaml` contains a `graph_features.pagerank` block **and** a `bm25` export block
  **and** `embedding.sentence_transformers` with the Granite r2 model. (If any is missing the wizard
  regressed — a finding against #193/#194.)
- `latence process` exits 0 and writes a run under `_latence/runs/<run_id>/`.
- `export/records.parquet` carries columns: `chunk_id`, `content`, `context_header`, `embedding`
  (768-d), `induced_labels`. `graph-nodes.parquet` / `graph-edges.parquet` and `graph.ttl` exist,
  and nodes carry `pagerank` + `community` columns.
- `export/bm25-stats.json` + `bm25-postings.parquet` exist and are non-empty.

---

## 3. Inspect every stage — visually and qualitatively

### 3.1 Stage-by-stage record inspection
Follow `fresh-pod` §5 to read each stage's `.jsonl` (`source → intake_screen → parse → chunk →
induce → extract → redact → profiling → disambiguate → graph → enrich → export`). For an RFC corpus
the `✓ look for` specifics:
- `induce.jsonl` — domain types like `protocol`, `header_field`, `status_code`, `requirement_level`,
  and relations like `obsoletes`, `updates`, `references`, `defines`.
- `extract.jsonl` — entities are RFC numbers, header-field names, method names; relations connect them.
- `graph-edges.parquet` — the **induced** `obsoletes`/`updates` edges should substantially overlap the
  corpus's **independent** `_refs.jsonl` graph. Overlap is a qualitative correctness signal; a wide
  divergence is worth a finding (the extractor is missing the corpus's most explicit relations).

### 3.2 The visual console
```bash
python -m latence_console.cli --run _latence/runs/<run_id>    # read-only, localhost, offline
```
✓ **look for** — in a browser: the run's **Quality Report** (verdict headlines, not raw JSON), the
**chunk inspector** (each chunk resolves to a source RFC + page/offset), and the **KG browser** where
**every edge shows its Evidence** (the sentence it was extracted from). Confirm on a known edge, e.g.
`RFC 9110 —obsoletes→ RFC 7230`: the Evidence must be the real "Obsoletes" line.

### 3.3 Correctness charts
```bash
python benchmark/rfc/stage_charts.py --run _latence/runs/<run_id> --out /workspace/report/stage-charts
```
✓ **look for** — self-contained SVGs (no external assets), one per claim:
- chunk-size distribution (mass at/under 640 tokens),
- entities/relations per stage (funnel: extract → disambiguate collapses duplicate entities),
- KG degree distribution + top-centrality nodes,
- **induced-vs-independent edge overlap** (the §3.1 correctness bar, as a Venn/bar).
These are dependency-free SVG (ADR-0016: no heavy chart lib in the pipeline). A PNG-only or
external-CDN chart is a finding.

---

## 4. Build the four retrieval indexes

### 4.1 Dense vectors → Qdrant (live, REAL server — the fake is forbidden here)
This pod has **no Docker**, so run the real Qdrant server from its pinned release binary. The
`FakeQdrantClient` exists only for offline CI and is **FORBIDDEN in this proof**: every green in §4 and
§6 must trace to a real query against this running server. If you cannot reach a real Qdrant, that is a
**finding** — never fall back to the fake, and never report a fake-backed pass as success.
```bash
QDRANT_VER=v1.12.4
mkdir -p /workspace/qdrant-bin && curl -sL \
  https://github.com/qdrant/qdrant/releases/download/${QDRANT_VER}/qdrant-x86_64-unknown-linux-gnu.tar.gz \
  | tar xz -C /workspace/qdrant-bin
cd /workspace/qdrant-bin && nohup ./qdrant > /workspace/qdrant.log 2>&1 &   # listens on :6333
sleep 6 && curl -sf localhost:6333/healthz && echo " <- real qdrant up"
export LATENCE_QDRANT_URL=http://localhost:6333
cd /workspace/latence-framework
latence-sink-qdrant load --export _latence/runs/<run_id>/export --collection rfc --vector dense
```
✓ **look for** — `healthz` returns OK **from the real server**; the loader reports N points upserted =
N corpus chunks; a re-run upserts the same ids idempotently (no duplication); `curl
localhost:6333/collections/rfc` shows the dense vector config, and a real scroll
(`curl -s localhost:6333/collections/rfc/points/scroll -H 'content-type: application/json' -d '{"limit":1}'`)
returns a genuinely stored point. **This is the first live-Qdrant exercise in the project's history** —
a failure here is a real finding, not a config typo to wave away. (If the pinned binary layout differs,
fix the command against the real release — that is an env finding, not a reason to use the fake.)

### 4.2 SPLADE-v3 sparse vectors (generate)
```bash
latence retrieval splade-index \
  --export _latence/runs/<run_id>/export \
  --model naver/splade-v3 \
  --accept-noncommercial-license \
  --out _latence/runs/<run_id>/export/splade
```
✓ **look for** — a `splade/` artifact with per-chunk `(indices, values)` sparse vectors; the run log
records the checkpoint **and** `weights_license=CC-BY-NC-SA-4.0, license_verified=false,
operator_accepted=true` (ADR-0012 honesty — see Appendix A). The `--accept-noncommercial-license`
flag is **mandatory**; without it the command must refuse (a silent default-to-permissive is a finding).

### 4.3 SPLADE sparse → Qdrant-native (the new-wiring step)
```bash
latence-sink-qdrant load --export _latence/runs/<run_id>/export \
  --collection rfc --vector sparse --sparse-source splade
```
✓ **look for** — the `rfc` collection now has a **named sparse vector** alongside dense, and a sparse
query returns candidates ranked by SPLADE dot-product. **Expected finding surface:** the sink
historically wrote sparse only as payload (Appendix B). If this command rejects `--vector sparse`, or
loads sparse as inert payload rather than a queryable named sparse vector, that is the pre-identified
finding — hand it to the fix-fleet, which wires Qdrant `SparseVector` named vectors + a regression
test, then re-run this step.

### 4.4 BM25 + KG augmentation (already emitted; verify usable)
```bash
python - <<'PY'
from latence_retrieval import Bm25Rescorer, DuckDBGraphSource, MultiHopExpander
# BM25 stats loaded from export/bm25-stats.json; graph from export/graph-edges.parquet
print("bm25 ok:", Bm25Rescorer.from_export("_latence/runs/<run_id>/export") is not None)
g = DuckDBGraphSource("_latence/runs/<run_id>/export/graph-edges.parquet")
print("2-hop neighbours of a seed:", len(g.neighbors(["RFC9110"], max_hops=2)))
PY
```
✓ **look for** — BM25 loads from the emitted stats (no re-tokenisation of the corpus), and the
DuckDB graph source returns a non-trivial 2-hop neighbourhood for a hub RFC. Zero neighbours = the KG
augmentation leg is dead → finding.

---

## 5. Derive super-hard fused questions + gold sets

```bash
python benchmark/rfc/build_questions.py \
  --run _latence/runs/<run_id> \
  --refs /workspace/corpus-rfc/_refs.jsonl \
  --target 40 --min-lift 0.24 \
  --out /workspace/report/rfc-benchmark.json
```
The builder mines the **independent** `_refs.jsonl` graph for gold pairs/chains, then keeps only
questions that pass a hard gate: **fused recall@10 > max(dense, bm25, splade) + `min-lift`** *and*
fused finds ≥1 gold chunk no single signal finds. It then requires literal-presence of the subject in
every gold chunk (kills index-page/incidental-mention false positives). Question types to produce:
- **multi-hop**: "RFC A obsoletes B which updates C — what does C say about X?" (needs the KG chain).
- **lexical-critical**: exact header-field / status-code / `MUST`-clause lookups (BM25/SPLADE win).
- **semantic-critical**: paraphrased-concept questions with no lexical overlap (dense wins).
- **compound**: two distant sections linked only by a co-mentioned entity (fusion wins).

✓ **look for** — a JSON with **≥ 30 questions that survive the gate** (do not pad to 40 with filler —
a smaller rock-solid set is the honest deliverable), each with: question, type, gold chunk ids, the
KG chain used, and per-signal recall. If far fewer than 30 survive, that is a real result about corpus
depth, not a failure to hide — report the true number.

---

## 6. Run the fused retrieval + measure

```bash
python benchmark/rfc/run_benchmark.py \
  --questions /workspace/report/rfc-benchmark.json \
  --qdrant $LATENCE_QDRANT_URL --collection rfc \
  --configs dense,bm25,splade,fused \
  --out /workspace/report/rfc-results.json
```
The `fused` config is the real stack: dense-top-100 (Qdrant) + BM25-top-100 + SPLADE-top-100 (Qdrant
native sparse) → `RrfFuser` → KG graph-injection + `MultiHopExpander` (2-hop, hubs filtered) →
cross-encoder `Reranker` → knapsack `Packer` @16k tokens.

✓ **look for** — real recall@10 per config, from real retrieval runs (never fabricated). The
proof-of-value bar: **fused mean recall@10 materially exceeds every single signal**, and on a
meaningful share of questions **no single signal finds any gold in top-10 while fused does**. Report
per-type breakdowns so the story is legible (where does SPLADE beat BM25? where does the KG chain
carry it?). If fused does *not* win, that is the most important finding of the whole run — surface it,
do not bury it.

---

## 7. The decision-maker report

```bash
python benchmark/rfc/make_report.py \
  --results /workspace/report/rfc-results.json \
  --charts /workspace/report/stage-charts \
  --out /workspace/report/PROOF.md
```
✓ **look for** — `PROOF.md` with: the headline recall table (dense/bm25/splade/fused), 3–5 worked
example questions each showing why single signals miss and fusion recovers, the correctness charts
from §3.3, and an **honest limitations section** (corpus ceiling, any refuted questions, the SPLADE-v3
non-commercial caveat). The claim it must support is the defensible one — *"the fused stack unlocks a
class of queries single retrievers structurally cannot answer"* — **not** an inflated single number.

---

## Appendix A — the SPLADE-v3 license posture (read before shipping anything)

- **Weights:** `naver/splade-v3` → **CC-BY-NC-SA-4.0** (non-commercial, share-alike). **Code:**
  loaded via `transformers` (Apache-2.0).
- Consequence: this is valid for a **private internal proof** only. The **exported sparse vectors are
  a derivative** and inherit NC + share-alike. It **cannot** be the framework's shipped default, and
  must never be committed as a default `model_id`.
- The framework enforces this: `latence-splade` pins **no** default checkpoint and declares
  `license=UNVERIFIED, license_verified=false` until an operator names a cleared model. The
  `--accept-noncommercial-license` flag in §4.2 is the operator's recorded, auditable override.
- To ship a permissive default later, swap to an Apache-2.0 sparse encoder (OpenSearch neural-sparse
  or `prithivida/Splade_PP_en_v1`) and drop the flag.

## Appendix B — pre-identified finding: Qdrant-native sparse vectors

The Qdrant sink (`packages/latence-sink-qdrant/src/latence_sink_qdrant/sink.py`) currently carries the
Export's `sparse_indices`/`sparse_values` **as payload**, and deliberately declares named sparse
vectors out of scope. Step 4.3 needs them as **queryable Qdrant named sparse vectors**. This is real
feature work, not a bug — scoped as its own ticket (`issues/dogfood/qdrant-native-sparse.md`): add a
`--vector sparse` load path that writes `models.SparseVector` under a named sparse vector, with a
regression test proving a sparse query returns SPLADE-ranked candidates (and dense stays byte-identical
when sparse is absent). Either pre-built before the loop or fixed on first encounter at 4.3.
