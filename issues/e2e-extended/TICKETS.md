# Extended E2E — ticket set

Rationale for every decision here lives in [DECISIONS.md](DECISIONS.md). This file is the work.

Rules that apply to every ticket, without exception:

- **A fix lands with a test that FAILED before it.** No test, no merge.
- **No fake green.** A stub, a mock, or a skipped GPU path is not a pass. If a leg cannot be proven
  offline, it is marked `GPU-ONLY` here and proven on the pod — not quietly asserted.
- Every knob added to the wizard must be reachable **end-to-end**: `SetupAnswers` field → CLI flag →
  emitted YAML → provider actually reads it. A field that stops at any of those four is not done.
- The wizard's byte-identical determinism contract holds: same answers ⇒ same YAML.

Legend: **S** ≤ half a day · **M** ~1 day · **L** > 1 day.

---

## Lane A — device honesty (blocks everything; do first)

### T-DEV-1 — Gate the TorchScript-JIT kill on compute capability · M

**Now:** `neutralize_torch_jit()` (`providers/perf.py:539`) sets `PYTORCH_JIT=0` unconditionally at
four call sites (`perf.py:306`, `:679`, `:1147`, `latence-disambig-glinker/backends.py:338`). No
device check, no capability check, no config knob.

**Why:** it exists solely for FM-NVRTC-JIT — `libnvrtc-builtins.so.13.0` missing on Blackwell/CUDA-13.
That failure does not exist on sm_86. It fires on every GLiNER/GLiNER2/GLinker load, i.e. on
`extract`, `redact`, `disambiguate` — the three stages that were **1,604 s of the prior run's
1,743 s**.

**The design constraint that shapes the fix:** the fault surfaces on the *first forward*, so no
load-time `try/except` can catch it. A wrong "skip" is a **crash**; a wrong "neutralize" is merely
**slow**. Therefore: **fail safe.** Neutralize when capability is unknown, unreadable, or ≥ 12.0;
skip only on a positively-identified known-good architecture. Keep an explicit operator override.

**Do not** let Providers set `PYTORCH_JIT` themselves — `test_perf_fp32_regression_guard.py:611`
forbids it and that is correct. `perf.py` stays the one site.

**Acceptance:**
1. A test that fakes `get_device_capability() == (8, 6)` and asserts the JIT is **left on** —
   red before the change.
2. A test that fakes `(12, 0)` and asserts it is **neutralized**.
3. A test that fakes a raising/absent `torch.cuda` and asserts it is **neutralized** (fail-safe).
4. The existing "no Provider sets PYTORCH_JIT" guard still passes.
5. `GPU-ONLY`: on the 3090, log line confirms the JIT was left enabled, and `extract` wall-clock is
   recorded for comparison.

### T-DEV-2 — Microbatch and thread defaults must read the machine, not a remembered pod · M

**Now:** `DEFAULT_MICROBATCH_MAX_TEXTS = 4` (`perf.py:915`), `DEFAULT_MICROBATCH_TOKEN_BUDGET = 3072`
(`perf.py:924`), `DEFAULT_INTRA_OP_THREADS = 8` (`perf.py:733`). Derived from a **32 GB** card and a
**128-core** host. Already operator-overridable; the defect is the defaults.

**Why:** batch-4 peaked at **21.6 GiB** on 32 GB. On our 24 GB card that is 2.4 GiB of headroom
before the OCR server and before fragmentation. The default is not conservative here — it is close to
the ceiling, and it is arrived at by coincidence rather than by measurement.

**Acceptance:**
1. Defaults derived from detected total VRAM and `os.cpu_count()`, with the 32 GB/128-core values
   reproduced exactly when those are what is detected (so nothing changes on the old pod).
2. Explicit config still wins over detection — test it.
3. A test asserting the resolved bound on a faked 24 GB device is **at or below** the faked 32 GB
   bound. Red before.
4. `GPU-ONLY`: peak VRAM recorded per stage on the 3090; no OOM across the full corpus.

---

## Lane B — the OCR path

### T-OCR-1 — A real vLLM serving surface, Ampere-correct · M

**Now:** `serve.sh` exports `VLLM_USE_FLASHINFER_SAMPLER=0` by default for **every** host
(`serve.sh:27`) — a workaround for FlashInfer misparsing capability `12.0` as `12 < 75`. `86` parses
fine. Missing entirely: `--async-scheduling`, an explicit `--compilation-config`, and an attention
backend selection.

**Do:**
- Make the FlashInfer override **conditional on detected capability ≥ 12.0**, defaulting off on
  Ampere. Keep the operator escape hatch.
- Add `--compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}'` explicitly. It is already the
  V1 default, but stating it makes the guarantee legible and version-proof. **Never `--enforce-eager`.**
- Add `--async-scheduling` (LightOn's own published recipe).
- Do **not** request `FULL` cudagraph mode: FlashAttention-2 declares `UNIFORM_BATCH` and vLLM
  auto-downgrades to `FULL_AND_PIECEWISE` anyway.
- Do **not** enable `cudagraph_mm_encoder`: `LightOnOCRForConditionalGeneration` is absent from
  vLLM's supported-models table for encoder CUDA graphs. It is a no-op at best.
- Attention backend: FA2 is already the sm_86 default. Prefer omitting the flag and **verifying from
  the startup log** over asserting it. Note `VLLM_ATTENTION_BACKEND` is deprecated for removal.
- Re-derive `--gpu-memory-utilization` for 24 GB. Start 0.80, fall back to 0.72 on any OOM. This is a
  **measurement**, not a constant to copy.

**Acceptance:**
1. A test asserting `VLLM_USE_FLASHINFER_SAMPLER` is **not** forced on a faked sm_86, and **is** on
   sm_120. Red before.
2. A test asserting `--enforce-eager` appears in no code path.
3. `GPU-ONLY`: startup log shows torch.compile active and CUDA graphs captured; pages/sec recorded.

### T-OCR-2 — Round both image sides to a multiple of 28 · S

**Now:** `rasterize.py` caps the longest side at `DEFAULT_MAX_LONGEST_DIM = 1540` (`rasterize.py:52`)
via LANCZOS; the shorter side lands wherever it lands.

**Why:** `Mistral3PatchMerger` merges 2×2 with `unfold(kernel=2, stride=2)`, which emits **only
complete windows**. An odd patch grid **silently drops** the trailing patch row/column — the bottom
or right ~14 px of the page. 1540 is safe (110 patches). A US-Letter page → width 1190 → **85
patches → odd**. Patch 14 × merge 2 = **28**.

Honest scope note: the mechanism is verified in vLLM/transformers source; the visible-loss
consequence is inferred. It is cheap to satisfy and cheap to test, so satisfy it.

**Acceptance:**
1. A property test over many page aspect ratios asserting both output dimensions ≡ 0 (mod 28) and
   the longest side ≤ 1540. Red before.
2. A test asserting aspect ratio is preserved within one 28-px step.
3. No upscaling: a page already under the cap is never enlarged.

### T-OCR-3 — Expose DPI and resolution policy through the wizard · S

**Now:** `DEFAULT_DPI = 200` (`rasterize.py:47`) and `DEFAULT_MAX_LONGEST_DIM = 1540` are unreachable
from setup on **every** parser path (`recipe.py:668-673`, `:684`, `:692`).

200/1540 are LightOn's published values and stay the defaults. The ticket is reachability, so the run
can be tuned without hand-editing YAML.

**Acceptance:** `SetupAnswers` field → CLI flag → emitted `parse.config` → provider reads it. One
test per hop. Byte-identical output when the flags are unset.

---

## Lane C — chunking and labels

### T-WIZ-1 — Chunk token budget must be an answer, not a literal · S

**Now:** `recipe.py:715` hard-codes `{"max_tokens": 640, "overlap_tokens": 80, "min_tokens": 8}`.
No override field on `ChunkContribution`, no `SetupAnswers` field, no CLI flag.

**Target:** 768, aligned to the GLiNER2 window.

**Carry the known trap into a test:** the chunker's tokens run ~1.2× *fewer* than mdeberta's, so a
768-token chunk budget previously produced median 666 / **max 921** mdeberta tokens — silent
truncation. `perf.py:700-707` binds `max_len=768` on the model side. The test must assert the
**model-side** token count stays within the window for the emitted budget, not merely that the YAML
says 768.

**Acceptance:** field → flag → YAML → provider; plus the mdeberta-window assertion. Red before.

### T-LABEL-1 — The corpus profiler (no LLM) · L

**Build:** a provider that samples the corpus and emits a candidate type space for
`label_inducer.lexicon` — `{slot: {type_name: [surface cues]}}`.

**Method (all of this is settled literature, not invention):**
- Candidate mining: noun-phrase / keyphrase extraction over a sample (YAKE/RAKE/c-TF-IDF class of
  methods — check licences per ADR-0012 before adopting any).
- Selection: **greedy submodular max-coverage** over mention mass under `|A| ≤ N`. Gives the
  **(1 − 1/e) ≈ 63.2%** guarantee (Nemhauser–Wolsey–Fisher 1978); Feige 1998 proves nothing
  polynomial beats it.
- Emit, alongside the labels: the **marginal-gain curve**, the **per-corpus optimality certificate**
  (`f(OPT_N) ≤ f(A_i) + Σ of the N largest remaining gains` — usually far tighter than 63.2%), the
  empirical coverage `C(N)`, the fitted Zipf α, and the **Good–Turing** estimate of residual
  uncovered mention mass.

**Report honestly, in the artifact:** there is **no published standard** for a "90% coverage" or
"marginal gain < x%" cutoff. If the fitted α ≈ 1, coverage grows logarithmically and **no sharp knee
exists** — in which case N is a budget decision and the artifact must say so rather than implying a
discovered boundary. Select N by Kneedle (`kneed`, BSD-3-Clause) on the gain curve and report both
N\* and the capped 40.

**Acceptance:**
1. Deterministic: same corpus + same seed ⇒ byte-identical candidate space.
2. A test on a synthetic corpus with a known-optimal cover, asserting greedy is within (1 − 1/e).
3. The emitted artifact carries curve + certificate + `C(N)` + Good–Turing residual.
4. Zero network calls — assert it.

### T-WIZ-2 — Wire the lexicon inducer as the non-LLM induction path · M

**Now:** `induce` is hard-coded to `label_inducer.llm` (`recipe.py:775-799`).
`label_inducer.lexicon` exists and is registered but is unreachable from setup.

**Do:** let the wizard emit `label_inducer.lexicon` with T-LABEL-1's output as `candidates`, keeping
`granularity: "chunk"`. Expose `min_hits` and `whole_word` — `whole_word=False` matters for German
compounding, which this corpus will contain.

**Acceptance:**
1. Emitted stack runs induction with **zero** LLM calls — assert no network.
2. A test asserting each chunk's induced set is a **subset** of the candidate pool and is
   **smaller than the pool** on a corpus where that must hold. This is the whole point; if every
   chunk gets all 40, the ticket has failed.
3. The ADR-0044 financial-PII floor still unions onto the redactor. Assert it.

### T-LABEL-2 — Measure N rather than assuming 40 · M · GPU-ONLY

GLiNER2 is a **uni-encoder** — labels and text share one sequence, cost O((n+m)²). Author guidance is
**20** labels; the docs say degradation past **30**; training cap is 25. Measured: +60% CPU latency
5→50 labels, −98.7% throughput 1→1024, and scores are **not permutation-invariant** in the label
list. **No controlled accuracy-vs-label-count ablation exists in the literature.**

**Do:** run extraction at N ∈ {10, 20, 30, 40} on a held-out slice; report entity/relation quality
and throughput per N. Also run one **label-order permutation** at fixed N to quantify the
order-sensitivity on our data.

**Acceptance:** a table, and a stated choice of N with the number that justifies it. If 40 costs
accuracy, we will have the measurement that says so — and that is a result, not a failure.

### T-WIZ-3 — Make the GLiNER2 performance knobs reachable · S

`microbatch_max_texts`, `microbatch_token_budget`, `intra_op_threads`, `dtype` are read by the
providers (`perf.py:983-997`, `:103-124`, `:733`) but the wizard emits none of them.

Leave `compile`/`quantize` on `"auto"` — `recipe.py:850-853` says the omission is deliberate and it
is right. This ticket is about the four knobs that must be tunable on a 24 GB card.

**Acceptance:** field → flag → YAML → provider, one test per hop, defaults byte-identical.

---

## Lane D — graph

### T-KG-1 — Canonicalize relations, then prove PPR isn't already enough · M

**Two gates, in order, before any link prediction.**

1. **Relation canonicalization is a hard prerequisite.** Open extraction yields |R| ≈ |V| —
   ReVerb45K has 21,623 relation phrases against 27,008 noun phrases. ULTRA's relation graph has |R|
   nodes and up to 4|R|² edges; at |R| ~ 10⁴ that is prohibitive. We already have the
   Type-Consolidation stage (v17b, `type_consolidation.cascade`). Measure |R| before and after and
   assert a bounded vocabulary (hundreds, not tens of thousands).
2. **Run Personalized PageRank over the raw graph first.** HippoRAG reports up to **+20% on
   multi-hop QA from PPR over an OpenIE graph with zero link prediction**, at 10–30× lower cost. If
   PPR closes our multi-hop gap, the entire link-prediction workstream is unnecessary — and that is
   the cheapest possible refutation of the premise. We already ship `graph_features.pagerank`; its
   config is emitted as literally `{}` (`recipe.py:1129-1130`), so no damping factor or iteration cap
   is reachable. Fix that as part of this ticket.

**Acceptance:** |R| before/after canonicalization; a multi-hop retrieval measurement with PPR on and
off. Both numbers reported whatever they say.

### T-KG-2 — A precision gate on predicted edges · M

`graph_completion.ultra` is already wired (`recipe.py:1153-1167`) with `target_precision: 0.9` and
`scorer: "auto"` as literals. Use `mgalkin/ultra_50g` (MIT weights, MIT code). Zero-shot first, as a
**re-ranker over candidates**, never an all-pairs scorer. Fine-tune only if needed: the paper's own
fine-tuning ran on a **single RTX 3090** in 1,000–2,000 steps.

**The honest ceiling:** on extracted open-schema KGs, SOTA **Hits@1 is 0.26–0.31**. Unfiltered top-1
densification means **~7 in 10 added edges are wrong**, and a false edge in a traversal graph
manufactures a spurious path that pulls unrelated documents in. Densification failure is *worse* than
recall failure.

**Do:** hold out a manually-verified sample **from our own graph** (never calibrate on FB15k-237 —
Akrami et al. SIGMOD 2020 is the standing proof that benchmark performance does not transfer).
Plot precision vs threshold **per relation type**. Admit only the top ~1–5% at ≥0.8 measured
precision. Cap additions per node so we densify the sparse periphery, not the hubs. Tag every added
edge with provenance so it is ablatable and reversible.

**Acceptance:** precision-vs-threshold curve on our own sample; edge count admitted; the ablation
(retrieval with and without predicted edges). **Gate on end-to-end retrieval, not on MRR** — a
densification that improves MRR and degrades retrieval is a failure.

---

## Lane E — corpus

### T-CORPUS-1 — Acquire the three-axis corpus · M

Sources and licence posture in DECISIONS.md §D4. **The office-format question is closed** —
GovDocs1 supplies genuine file-type diversity under CC0, so no synthetic slice is needed.

| Axis | Source | Pull |
|---|---|---|
| Format diversity | GovDocs1, 333 GB, CC0 | `aws s3 sync --no-sign-request s3://digitalcorpora/corpora/files/govdocs1/zipfiles/` |
| Scanned realism | IIT-CDIP @ NIST, 1.41 TB, 656 shards | `curl -O https://data.nist.gov/od/ds/ark:/88434/mds2-2531/cdip-images/images.<a>.<b>.tar` |
| Scored control | OHR-Bench, 1.84 GB, CC-BY-4.0 | `hf download opendatalab/OHR-Bench --include "pdfs.zip"` |

Start with GovDocs1's ten pre-built 1,000-file `threads/` subsets (3.09 GB) as a smoke test before
committing to the full pull.

**Two hard operational constraints, both non-negotiable:**

1. **GovDocs1 contains live malware, deliberately retained** — *"the malware will not be removed
   from the files because it is legitimately part of the corpus."* The pod is untrusted: no host
   mounts, no credential material in the environment, expect AV noise. We only ever *parse* these
   files, never execute them — but this must be a deliberate posture, not a surprise at 2am.
2. **CDIP/IDL material never leaves the pod** and nothing derived from it enters a public artifact.
   Its rights posture is fair use, not a grant. Anything we publish comes from GovDocs1 (CC0),
   OHR-Bench (CC-BY-4.0) or DocLayNet (CDLA-Permissive).

**Watch for the `by_type/` trap:** there is no `pdf.zip` and no `html.zip` — the per-type archives
omit exactly the two largest classes. Take `zipfiles/`.

**Acceptance:** a manifest with per-source counts, bytes, licence, and the **measured** file-type
histogram — so the write-up cannot overclaim coverage it does not have. Specifically record that
office formats are legacy OLE (DOC/XLS/PPT), not modern OOXML: GovDocs1 has only 164 DOCX / 39 XLSX
/ 220 PPTX, so **modern OOXML at volume remains untested in this run**. Say so rather than letting
"many file types" imply it.

---

## Lane F — retrieval and benchmark

The audit changed the shape of this lane substantially. Three findings drive it:

- **SPLADE cannot be a signal today.** The query side is real; the *load* side does not exist.
- **The KG is a post-fusion rescorer, not a retrieval leg** — architecturally it cannot recover a
  document the first stage missed, which is the exact claim we want to make.
- **The prior proof's headline is close to tautological**, and we must not repeat that shape.

### T-RET-1 — Make SPLADE a real signal: write named sparse vectors to Qdrant · L

**Now:** `sink.py:767-782` creates **dense-only** `VectorParams`. There is no `sparse_vectors_config`,
no `SparseVectorParams`, and grep returns **zero Python hits** for either across the repo. The CLI has
no `--vector sparse` flag (`cli.py:47-68`). `FakeQdrantClient` contains **zero occurrences of
"sparse"**, so no test could ever have caught this.

The Export's `sparse_indices` / `sparse_values` columns ride through into the **payload** as ordinary
lists (`export_reader.py:193`) — stored, never queryable. Meanwhile the query side is genuinely
complete: `QdrantBackend` builds `models.SparseVector` and issues
`query_points(..., using=sparse_vector_name)` (`qdrant.py:245-265`). We have a query for an index
that is never built.

This is `issues/dogfood/qdrant-native-sparse.md`, still open, all acceptance criteria unchecked.

**Do:** `sparse_vectors_config` at collection creation; `--vector sparse` / `--sparse-source` on the
CLI; **teach `FakeQdrantClient` sparse vectors first** — a fake that cannot express the feature makes
the tests structurally unable to fail.

**Acceptance:**
1. `FakeQdrantClient` models named sparse vectors and **rejects** what a real Qdrant rejects.
2. A test asserting the created collection carries a named sparse vector. Red before.
3. `GPU-ONLY`, **against a real Qdrant binary**: a sparse-only query returns documents a dense-only
   query does not. Per the standing rule, the fake is offline-CI only — this leg is proven live.

Also fix, in the same lane, the two harness bugs that would have made SPLADE silently useless even
once loaded: `run_benchmark.py:516` and `build_questions.py:653` pass `{"model_id": ...}` but
`SpladeCompositionEmbedder` reads `cfg.get("model", "")` (`provider.py:177`) and does **not** reject
unknown keys — so the model id silently resolves to `""`. And `build_questions.py:678-683` passes a
`SparseVector` object where `RetrievalQuery.sparse_terms` is `dict[str, float]`
(`contracts.py:79`); `run_benchmark.py:519-521` converts correctly, `build_questions.py` does not.

### T-RET-2 — The KG must be able to RETRIEVE, not only reorder · L

**This is the architectural one, and it decides whether the KG claim is provable at all.**

`MultiHopExpander` (`multihop.py:503`) and `Expander` (`expand.py:71`) are real, tested and
DuckDB-backed — but both are `CandidateProcessor`s applied **after** fusion
(`orchestrator.py:266-269`). They add a hop-decayed bonus to already-fused scores and **cannot
introduce a candidate the first stage missed** (`multihop.py:621-628`). `Capability`
(`contracts.py:21-35`) has no `GRAPH` member.

So if the benchmark's premise is *"graph traversal recovers documents dense+sparse never
retrieved"* — **the current architecture cannot express that claim.** Every KG "win" it can produce
is a reordering of a pool the graph did not choose.

**Do:** add a graph **retrieval leg** — seed from query entities, traverse to bounded depth, return a
ranked candidate list that enters fusion as a peer of dense/sparse/BM25. Keep the existing expander:
rescoring after fusion is a legitimate *second* use, just not evidence for the recovery claim.

**Acceptance:**
1. A test where the gold document is **unreachable** by dense and sparse but reachable by traversal,
   asserting the fused result contains it. This is impossible to pass today — that is the point.
2. `Capability.GRAPH` declared, and the orchestrator's refuse-if-unserveable path
   (`orchestrator.py:291-297`) honours it.
3. The ablation of D7 becomes runnable: retrieval with and without the graph leg.

### T-FUSE-1 — Wire the fuser we already have, and tune it · M

**Good news:** `WeightedFuser` (`fuser.py:158-216`) already does per-list **min-max-normalised
weighted sum** — the Bruch-recommended family. It is tested (22 tests) and registered
(`fuser.weighted`).

**The gap is wiring, not building.** Every fusion call in every harness is `RrfFuser()` with default
`k=60` and **uniform weights** — `run_benchmark.py:396`, `build_questions.py:690`,
`rfc_bench.py:82`. No harness passes weights; no code tunes them. `normalization` accepts only
`"min_max"` (`fuser.py:176-182`), so theoretical-min-max is a small addition if we want the
query-independent variant.

**Do:** run both fusers over the same candidate lists; sweep weights on a small labelled slice
(Bruch: "a handful of labeled queries" suffices); report both. Keep RRF as the no-labels fallback.

**Acceptance:** a table of RRF-k=60 vs tuned convex combination on the same questions. Whichever
wins, wins — we report the comparison, not a preferred outcome.

### T-BENCH-1 — Question generation with hop stratification · L

**Now:** two harnesses that disagree, and the shipped proof came from the *ad-hoc* one.

`benchmark/rfc/build_questions.py` uses fixed string templates; `gen_semantic` emits **one identical
sentence for every RFC** (`:351-372`) carrying no topic information at all — that flavour cannot be a
semantic probe. `results/dogfood-rfc/report/rfc_bench.py` — the script that actually produced the
proof — uses gpt-4.1-mini, a different gate, and a different stack (**no SPLADE, no Packer**).

**Do:** one harness. Generate 2–4 as serial depth chains and 5–7 as fan-out/constraint-intersection
(D1b), from the gold graph, with the shape recorded per question. Report **stratified by hop count
and by shape**, never pooled.

**Acceptance:** every question carries its verified hop count and shape; the report has no pooled
"multi-hop accuracy" number anywhere.

### T-BENCH-2 — The necessity gates · L

**Now: essentially none exist.** The current gate is
`recall[fused] > best_single + 0.24` (`build_questions.py:495`) — a *retrieval-lift* condition, not a
hop-necessity one. A question passes if fusion happens to win, for any reason.

Worse, the multi-hop template **names the whole chain in the question text** —
`"RFC {a} obsoletes RFC {b}, which {rel} RFC {c}"` (`:310`) — and gold is defined as chunks literally
containing `"RFC {c}"` (`:302-303`). **The bridge and terminal entities are handed to the retriever
verbatim.** A plain BM25 query for `RFC 5226` finds those chunks; nothing is forced to traverse.

**And the prior headline must be read in that light.** `rfc_bench.py:120` gates on
`rf > rd and rf > rb` — it **selects** the rows where fused beats both singles. On that kept set,
"dense 0.000 / bm25 0.000 / fused 1.000" is close to definitional given single-passage gold and
binary recall@10. The strict set was **7 questions**. We do not repeat that shape.

**Do:** implement G1–G8 from D9. Report the **funnel** — enumerated → structural → G3 → G4 →
human-audited. MuSiQue's tail-survival was 26.5%; expect similar and publish it.

**Acceptance:** a question that names its own bridge entity is **rejected** by G5. Red before.

### T-BENCH-3 — The KG ablation · M

Depends on T-RET-2. Retrieval with and without the graph leg, stratified by hop count. Report
whatever it says — D7's literature says the honest expectation is a **narrow** win concentrated in
multi-hop, and a possible *loss* on single-hop through relevance dilution.

### T-BENCH-4 — External benchmarks: StrategyQA + BEIR/HotpotQA · L

**Now: ABSENT.** Searched `hotpot`, `beir`, `musique`, `2wiki`, `msmarco` — the only hits are prose in
our own DECISIONS.md. `latence-benchmark` has suites for parse, redaction and relation extraction;
**there is no retrieval suite at all**.

StrategyQA is the best fit (official Recall@10 scorer; stale 2022 SOTA of **0.457**; nobody has
published retrieval numbers since). Its licence is **UNVERIFIED** — AI2's dataset page is dead —
confirm before any external claim.

**Blocker to size first:** `Bm25Rescorer` is a **rescorer, not an index** (`bm25.py:9-11`). Both
harnesses hand it the *entire corpus* as candidates (`run_benchmark.py:331-333`). At 3.5k chunks that
is fine; at BEIR scale it is O(corpus) pure-Python tokenisation per query and a hard wall. Either
build a first-stage index or scope to a corpus size where the rescorer is honest — and say which.

### T-RET-3 — Prove the retrieval leg against a real Qdrant · M

**Now:** `test_live_smoke.py` is the only `LATENCE_QDRANT_URL`-gated test and it covers
**load/converge/purge, not search**. Every `QdrantBackend` test monkeypatches the client. The
retrieval path has never been proven against a real Qdrant by any test — only by the ad-hoc
`rfc_bench.py` run.

That is precisely the standing rule: fakes are offline-CI only and must reject what the real
dependency rejects.

**Acceptance:** a `LATENCE_QDRANT_URL`-gated search test — dense, sparse, and filtered — against the
real Qdrant binary. `GPU-ONLY`/pod.

### T-RET-4 — Fix the three latent breakages on the pod path · S

All three sit inside `# pragma: no cover - pod path`, so the green suite says **nothing** about them:

1. **Wrong graph-edge columns.** `run_benchmark.py:462-464` selects `source, target`; the actual
   schema is `source_node_id` / `target_node_id`. DuckDB binder error → the whole hub-filter and
   multi-hop leg dies before the first query. (`DuckDBGraphSource` itself is correct at
   `multihop.py:193-194` — only the harness is wrong.)
2. SPLADE `model_id` vs `model` — see T-RET-1.
3. `SparseVector` vs `dict[str, float]` — see T-RET-1.

**Acceptance:** the pod path gets offline coverage against the **real** parquet schema, so a column
rename cannot pass again. Red before.

### T-RET-5 — Metadata filters: index them, and actually measure them · M

Plumbing exists — `RetrievalQuery.filter`, `_build_filter` (`qdrant.py:267-281`), and the sink writes
every enrichment column into the payload so `kg_node_community`, `classification`, `provenance` are
filterable in principle. Three gaps:

- Filters are **exact equality, AND-ed only** — no ranges, no `in`, no boolean composition.
- **No payload indexes are ever created** (zero `create_payload_index` in code, and
  `README.md:100-103` says this is left to the operator). Every filter is an unindexed payload scan.
- **No benchmark ever sets a filter.** Filtered retrieval is entirely unmeasured.

Since the request explicitly asks for "metadata management through extraction and enrichment", this
has to be measured, not merely present.

---

## Run order

```
T-DEV-1 ─┬─ T-DEV-2 ──────────────────────────┐
         └─ T-OCR-1 ─ T-OCR-2 ─ T-OCR-3 ──────┤
T-WIZ-1 ─── T-LABEL-1 ─ T-WIZ-2 ─ T-LABEL-2 ──┼─ staged pod run ─ Lane F
T-CORPUS-1 ───────────────────────────────────┤
T-KG-1 ─── T-KG-2 ────────────────────────────┘
```

Lane A first: every later measurement is invalid while the JIT kill and the 32 GB-derived batch
bounds are silently in force.
