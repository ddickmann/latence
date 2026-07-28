# Extended end-to-end GPU run — decisions and their evidence

Status: **PREPARATION**. Nothing launches until the gap analysis closes and the check-in happens.

Target hardware: **1× RTX 3090, 24 GB, Ampere sm_86** — not Blackwell, and **8 GB less VRAM than
the 32 GB card every current default was tuned on**. That single fact drives half of what follows.

Every decision below carries the evidence that produced it. Where the evidence contradicts the
original request, the contradiction is stated plainly rather than quietly resolved.

---

## D1 — Two premises in the request are wrong, and acting on them would cost us

### D1a. "longest side 1536 → shorter side must not be < 768" does not apply to LightOnOCR

That rule is **Qwen2-VL `smart_resize` semantics**. LightOnOCR does not inherit them. Verified from
the model's own config files:

- `processor_config.json` → `"image_processor_type": "PixtralImageProcessorFast"`,
  `"size": {"longest_edge": 1540}`. **`longest_edge` is the only size key** — no `min_pixels`, no
  `max_pixels`, no `shortest_edge`.
- `config.json` → `"model_type": "mistral3"`, `vision_config.model_type: "pixtral"`.

LightOn's published preprocessing guidance is two lines: render PDFs at **200 DPI**, target longest
dimension **1540 px**, preserve aspect ratio. There is no minimum-shorter-side rule anywhere in the
model card, the LightOn blog, the HF blog, or the paper. Implementing a 768 floor would upscale
narrow pages for no benefit.

**The real constraint, which nobody documents and which we would otherwise have shipped past:**
`Mistral3PatchMerger.forward` does 2×2 merging with `torch.nn.functional.unfold(kernel=2, stride=2)`,
which emits **only complete windows**. If the 14-px patch grid has an odd height or width, the
trailing patch row/column is **silently discarded** — the bottom or right ~14 px of the page is lost.
1540 is safe (1540/14 = 110, even), but a US-Letter page scales to width 1190 → 85 patches → **odd**.

→ **Round both sides to a multiple of 28** (patch 14 × merge 2). See T-OCR-2.

Severity note, stated honestly: the *mechanism* is verified in vLLM/transformers source; the
*consequence* is inferred, not LightOn-confirmed. It is cheap to satisfy and cheap to test, so we
satisfy it and add the test rather than argue about it.

### D1b. "2–7 hop" conflates two different tasks

**No published, human-validated benchmark exceeds 4 hops of serial depth.** HotpotQA and
2WikiMultihopQA are 2-hop. MuSiQue is 2–4. MultiHop-RAG tops out at 4 evidence pieces. HopWeaver —
the strongest 2025 *automatic* generator — does 2 only.

The benchmarks quoting 7–15 get there by **breadth**: FanOutQA's ~7 is fan-out ("for each of N
entities fetch X, then aggregate"); FRAMES' 2–15 is constraint-intersection. Neither is a 7-link
dependency chain.

Difficulty saturates well before 7 — MuSiQue's best model scores **57.9 / 47.9 / 28.1** at 2 / 3 / 4
hops. A 7-deep chain measures near-zero, which discriminates nothing. Khashabi et al.
(arXiv 1901.02522) additionally give a formal argument that beyond logarithmic depth over a noisy
language-grounded graph, no system is likely to answer correctly — so a 7-hop failure would not tell
us our retrieval is bad.

→ **2–4 = serial depth chains. 5–7 = fan-out / constraint-intersection over the KG. Reported
stratified by hop count AND by shape, never pooled.** Pooling a 7-wide fan-out with a 3-deep chain
into one "multi-hop accuracy" is exactly the move the shortcut literature exists to puncture. See
T-BENCH-1.

---

## D2 — The Blackwell remedies are an unconditional Ampere tax

Verified by reading the code, not by relaying a report.

| What | Where | Gated on | Verdict |
|---|---|---|---|
| `PYTORCH_JIT=0` + `torch.jit._state.disable()` | `providers/perf.py:539`, called at `:306`, `:679`, `:1147` and `latence-disambig-glinker/backends.py:338` | **nothing** — no device check, no capability check, no config knob | **UNCONDITIONAL** |
| `VLLM_USE_FLASHINFER_SAMPLER=0` | `deploy/env/served-ocr-server/serve.sh:27` | defaults to `0` for every host | **UNCONDITIONAL** |

`neutralize_torch_jit()` exists for **FM-NVRTC-JIT** — `libnvrtc-builtins.so.13.0` missing on a
Blackwell/CUDA-13 box. `VLLM_USE_FLASHINFER_SAMPLER=0` exists for **FM-FLASHINFER-SM120** —
FlashInfer misparsing compute capability `12.0` as the integer `12`, which compares below 75.

**Neither failure mode exists on sm_86.** `86` parses fine; the nvrtc path is healthy.

The JIT one is the expensive one: it fires on every GLiNER/GLiNER2/GLinker load, i.e. on `extract`,
`redact` and `disambiguate` — **the three stages that consumed 1,604 s of the prior run's 1,743 s**.
The docstring is honest that this is "throughput only, byte-identical output", so it is safe, not
wrong. It is simply free performance we are declining to take.

**Design constraint on the fix (T-DEV-1):** the fault surfaces on the *first forward*, so no
load-time `try/except` can catch it. A wrong "don't neutralize" is a **crash**; a wrong "neutralize"
is merely **slow**. That asymmetry means the gate must **fail safe**: neutralize when the capability
is unknown or unreadable, and skip only on a positively-identified known-good architecture.
`test_perf_fp32_regression_guard.py:611` asserts no Provider may set `PYTORCH_JIT` itself, so
`perf.py` is the only place this can change — which is correct and should stay that way.

---

## D3 — The 24 GB card is a downgrade, and the defaults do not know it

Every current default was measured on a **32 GB** Blackwell pod:

- `DEFAULT_MICROBATCH_MAX_TEXTS = 4` and `DEFAULT_MICROBATCH_TOKEN_BUDGET = 3072` (`perf.py:915`,
  `:924`) — derived because `batch_size=32` OOM'd and `batch_size=4` peaked at **21.6 GiB**.
- `DEFAULT_INTRA_OP_THREADS = 8` (`perf.py:733`) — derived from a **128-core** host.
- vLLM `--gpu-memory-utilization 0.72` / `--max-num-seqs 16` (`serve.sh:35`, `:40`) — derived from a
  32 GB card, after OOM at 0.90.

21.6 GiB peak on a 32 GB card leaves **2.4 GiB of headroom on a 24 GB card** — before the OCR server,
before fragmentation. The batch-4 setting is not conservative here; it is close to the ceiling.

→ **The run must be STAGED, not co-resident.** Serve OCR → parse the whole corpus → tear the server
down → run the learned stages. Attempting to co-host a vLLM server and GLiNER2 on 24 GB is the most
likely way to lose an afternoon. See T-RUN-1.

→ The microbatch and thread defaults must become **VRAM- and core-aware**, not 32 GB/128-core
literals. They are already operator-overridable (`microbatch_max_texts`,
`microbatch_token_budget`, `intra_op_threads`) — the defect is that the *defaults* encode one
specific pod. See T-DEV-2.

---

## D4 — Corpus

The request is "very large, corporate-like, many different files, data types and various content".
The honest finding is that **the corporate-document space is license-hostile**, and the one axis we
cannot satisfy from public data is genuine file-type diversity.

### Chosen

| Source | Why | Size | License |
|---|---|---|---|
| **docling-project/DocLayNet-v1.2** | The license-clean corporate core: financial reports, manuals, government tenders, laws, patents. **Embeds the original PDF bytes** *and* hand-annotated layout GT — so it grounds parse quality, not just throughput. | 41 GB full; ~5 GB for a 10-shard slice | **CDLA-Permissive-1.0 — commercial use permitted.** The only corporate-domain corpus surveyed where the publisher actually grants rights. |
| **pixparse/idl-wds** | UCSF Industry Documents Library — real corporate memos, contracts, correspondence, marketing plans, released through litigation. Decades of **scanned** paper, typewritten memos, faxes, stamps, handwriting. Zero academic contamination. This is the closest public proxy for an enterprise archive that exists. | ~830 MB/shard; take 25 shards ≈ 21 GB | Custom "IDL-Train": US copyright, **fair-use / court-approved-agreement basis, user bears compliance**. Not a commercial grant. |
| **opendatalab/OHR-Bench** | Small, CC-BY-4.0, and 5 of its 7 domains are corporate/institutional (Finance, Law, Manual, Administration, Newspaper). Ships **raw PDFs + QA ground truth**, so it is a scored control alongside the unscored bulk. | 1.84 GB | **CC-BY-4.0** |

**IDL's real PII is a feature, not a hazard, for this run** — it is the only way to exercise the
GLiNER2 PII redaction and guardrail stages against genuine personal data rather than synthetic
fixtures. It must never leave the pod, must never be redistributed, and nothing derived from it goes
into a public artifact. Recorded here so that constraint is explicit rather than assumed.

### Rejected, with reasons

- **OmniDocBench** — research-only / non-commercial. Card: "for research purposes only and not for
  commercial use."
- **Enron** — real PII with no fair-use framing and no corporate-document diversity beyond email.
- **SEC EDGAR** — public domain and huge, but HTML/XBRL-native, not PDF; we would be benchmarking our
  own renderer.
- **DocGenome (2.84 TB), BigDocs ArxivOCR, PaperPDF** — academic papers. Wrong genre.
- **ESG/sustainability report corpora** — surveyed thoroughly; no free, large, clearly-licensed
  corpus of original PDFs exists. The largest archives are scrape-only with unverified terms.

### The gap we cannot close from public data — stated, not hidden

**All three sources are PDF/image.** None supplies DOCX, XLSX, PPTX or HTML at scale. The request
asks for "many different file types" and for "file preprocessing converting any filetype to image
batches", and public corpora do not provide that.

Options, in preference order:
1. Generate a **modest, clearly-labelled office-format slice** (LibreOffice headless round-trip of a
   DocLayNet subset into DOCX/PPTX/XLSX) purely to exercise the `parser.render` path, and report it
   as synthetic — never folded into the quality numbers.
2. Accept a PDF/image-dominant corpus and state that office-format handling is **untested at scale**
   in this run.

Do NOT silently pick (2) and let the write-up imply full file-type coverage. See T-CORPUS-1.

---

## D5 — Labels without an LLM

The request forbids LLM-induced labels ("too expensive") and asks for a sampled top-N (N ≤ 40) on a
Pareto frontier. Two findings shape the design.

### D5a. The seam already exists

`predefined_labels` (T1) makes the generator emit **no `induce` stage at all** — not a disabled one,
not a short-circuited Provider. Every extraction stage then runs each chunk against the operator's
fixed label set, and the ADR-0044 financial-PII floor still unions onto the redactor. A
corpus-sampled top-40 set is, from the pipeline's point of view, a *predefined* set. **The plumbing
is built and tested; only the profiler that produces the 40 is missing.**

### D5b. 40 is above the safe ceiling for a uni-encoder, and GLiNER2 is a uni-encoder

Verified from the GLiNER2 paper and source: the input format is
`[Task Prompt] ⊕ [SEP] ⊕ [Input Text]` — labels and text share **one** transformer sequence, cost
O((n+m)²). A GitHub code search for "bi-encoder" across `fastino-ai/GLiNER2` returns **zero hits**;
there is no label encoder in the tree.

Published guidance for uni-encoder GLiNER, from the maintainers:

- urchade (author): *"For the vanilla GLINER model, I recommend a limit of **20 labels**."*
- `docs/architectures.md`: *"performance degrades when number of entity labels exceed **30**."*
- Training cap: `max_types = 25` (GLiNER config default). Inference does **not** silently drop
  labels beyond this — you simply leave the training distribution.
- Measured latency, GLiNER2 CPU classification: 5 labels 130 ms → 50 labels 208 ms (**+60%**).
- Measured throughput collapse, uni-encoder at scale: **−98.7% from 1 → 1024 labels**.
- Scores are **not permutation-invariant in the label list** — confirmed by the author as a
  positional-encoding effect ([GLiNER issue #192](https://github.com/urchade/GLiNER/issues/192)).

**No controlled accuracy-vs-label-count ablation exists in the literature.** The "degrades > 30"
claim is a maintainer assertion plus a confirmed order-sensitivity mechanism, not a measured curve.

→ Build the profiler to produce a **ranked** label set and make N a measured choice, not an assumed
one: run the extraction stage at N ∈ {10, 20, 30, 40} on a held-out slice and pick N from the curve.
If 40 costs accuracy, we will have the number that says so. See T-LABEL-1 and T-LABEL-2.

### D5c. The selection algorithm is a solved problem — use the solved version

Label selection under a cardinality cap is **monotone submodular maximization**: choose A ⊆ L with
|A| ≤ N maximizing the mention mass covered. Greedy gives the **(1 − 1/e) ≈ 63.2%** guarantee
(Nemhauser–Wolsey–Fisher 1978), and Feige 1998 proves no polynomial algorithm does better unless
P = NP. Greedy also yields, for free, a **per-corpus optimality certificate** far tighter than the
worst case — "these 40 labels are within X% of the best possible 40 **on this corpus**".

"Pareto frontier" in the request maps onto the **marginal-gain curve**, which submodularity
guarantees is monotonically non-increasing — so a knee is well-defined. Select N with Kneedle
(Satopää et al. 2011; `kneed`, BSD-3-Clause) and report it alongside the capped 40.

**Report honestly:** there is **no published standard** for a "90% coverage" or "marginal gain < x%"
threshold. Under Zipf with α ≈ 1, coverage grows only logarithmically and *no sharp knee exists* — in
which case the cap is a budget decision and must be described as one. Good–Turing gives the estimate
of residual uncovered mention mass; report it rather than implying the 40 are sufficient.

---

## D6 — Fusion: not RRF-by-default

RRF's `k = 60` is not a principled constant. From the original 2-page SIGIR'09 paper: *"k = 60 was
fixed during a pilot investigation and not altered during subsequent validation"*, and the paper's own
Table 1 sweep shows MAP .2139 (k=30) → **.2145 (k=60)** → .2147 (k=80) → .2142 (k=100). k=60 is a
flat-plateau point, and the optimum is actually at 80. It was measured once, over **30 homogeneous
TREC runs** — not over 4 heterogeneous signals of query-dependent reliability, which is our case.
RRF was explicitly invented *as a baseline*: *"we came up with Reciprocal Rank Fusion to serve as a
baseline."*

Bruch et al. (TOIS 2023) show a **normalized convex combination beats RRF** in-domain and
out-of-domain (MS MARCO nDCG@1000 0.454 vs 0.425; HotpotQA 0.699 vs 0.675), that RRF's parameters
are domain-sensitive and **flip direction** between in-domain and out-of-domain, and that RRF lacks
Lipschitz continuity because rank conversion discards score distance. Tuning needs "just a handful of
labeled queries". Use **theoretical min-max** normalization, which is query-independent and therefore
safe at serving time.

Vendor-bias disclosure: the authors are from Pinecone, a vector-DB vendor with an interest in "dense
scores carry information". The theory is checkable and independent of their product; discount the
empirical margins accordingly.

→ Implement fusion as a **normalized weighted sum with one interpretable weight per signal**, keep
RRF as the no-labels fallback, and report both. See T-FUSE-1.

---

## D7 — The KG claim must be earned, not assumed

The independent literature is substantially more sceptical than the GraphRAG marketing:

- **DIGIMON** (VLDB, 12 methods re-implemented in one framework): no graph method consistently beats
  VanillaRAG. On Quality, VanillaRAG 60.83 vs best graph method 62.55 — **+1.7 points** for an entire
  graph pipeline. G-retriever *underperformed ZeroShot by 14.17 points* on the same dataset.
- **GraphRAG-Bench** (ICLR 2026): graph methods achieve **higher recall but lower relevance**,
  producing noisier contexts and *lower* generation accuracy. That is precisely the trade a graph
  layer would impose on a precision-tuned hybrid.
- **Han et al.**: plain RAG beats graph RAG on single-hop factual QA (NQ F1 64.78 vs 63.01 vs 50.27),
  entity extraction **loses ~34% of answer entities**, and KG index build took **7,702 s vs 135 s**.
  The genuinely real number: **13.6%** of MultiHop-RAG queries are answered *only* by GraphRAG —
  routing captures **+1.1%**, running both and merging **+6.4% at 2× compute**.
- **Microsoft's own LazyGraphRAG** reports indexing at **0.1%** of full GraphRAG cost at comparable
  quality — i.e. 99.9% of their flagship's indexing spend bought nothing.
- **Most published win rates are LLM-judge artifacts.** Position bias swings win rates >30%; a
  25-token length gap swings them >50%. De-biased, **LightRAG vs NaiveRAG inverts** from 72/28 to
  NaiveRAG slightly ahead.

**Nobody in that literature compared against a strong hybrid with a reranker.** Every baseline is
dense-only or BM25-only. Our SPLADE+BM25+dense+metadata stack is stronger than every baseline those
papers beat, so their measured deltas are an **upper bound** on what we can expect, and probably a
loose one.

→ The KG stays in the run, because the request asks for it and because the multi-hop slice is where
it can genuinely win. But it ships with an **ablation**, not an assertion: measure retrieval with and
without the graph signal, stratified by hop count. If it does not earn its weight, that is a finding
worth reporting, not a failure to hide. See T-BENCH-3.

---

## D8 — Link prediction

**ULTRA** is the choice, and it is unusually clean: **MIT code and MIT weights**, verified
separately. It is fully inductive (new entities *and* new relation types), which is the only class
that survives an open, growing extracted schema. Checkpoint **`mgalkin/ultra_50g`** — it closes the
transductive gap the ICLR paper conceded (0.389/0.549 vs supervised 0.371/0.511 on 16 transductive
graphs). Decisive practical fact: **the paper's own fine-tuning ran on a single RTX 3090, and
1,000–2,000 steps sufficed.** That is a sub-hour experiment on exactly our hardware.

The better-performing successors are all legally encumbered: **KG-ICL is GPL-3.0**; **TRIX and MOTIF
carry no license at all** (verified `null` via the GitHub API — all rights reserved).

**Two hard prerequisites, in order:**

1. **Canonicalize the relation vocabulary first.** Open extraction yields |R| ≈ |V| — ReVerb45K has
   21,623 relation phrases against 27,008 noun phrases. ULTRA's relation graph has |R| nodes and up
   to 4|R|² edges; at |R| ~ 10⁴ that is prohibitive. This is not an optimization, it is a gate. We
   already have a Type-Consolidation stage (v17b) — it must run and be measured before ULTRA.
2. **Prove PPR over the raw graph doesn't already solve it.** HippoRAG gets up to **+20% on
   multi-hop QA from Personalized PageRank over an OpenIE graph with zero link prediction**, at
   10–30× lower cost. Run that first. If it closes the gap, the entire link-prediction workstream is
   unnecessary — and that is the cheapest possible refutation of the premise.

**The honest ceiling:** on automatically-extracted open-schema KGs, SOTA **Hits@1 is 0.26–0.31**. Add
top-1 predictions unfiltered and **~7 in 10 added edges are wrong** — and a false edge in a traversal
graph does not merely fail to help, it manufactures a spurious path that pulls unrelated documents
into the neighbourhood. Densification failure is *worse* than recall failure.

→ Admit only the top ~1–5% of predicted edges at **≥0.8 measured precision on our own held-out
sample** (never calibrated on FB15k-237), cap additions per node so we densify the sparse periphery
rather than the hubs, tag every added edge with provenance so it is ablatable, and **gate on
end-to-end retrieval, not on MRR**. See T-KG-1 and T-KG-2.

---

## D9 — What the benchmark must prove, and how

The prior RFC proof recorded, honestly, that only **7 of 90 questions were genuine multi-hop wins**.
A "fused 1.000" headline without a necessity control will not survive a hostile technical reader.

Every generated question passes these gates before it counts. Each is a published control, not an
invention:

| Gate | Condition | Source |
|---|---|---|
| **G1 closed-book** | A no-retrieval LLM scores at/near chance. Report the number. | FRAMES reports 0.408 |
| **G2 single-hop baseline** | BM25 top-k + reader, and dense + reader, both well below the fused system and below the oracle. | Min et al. 2019; FRAMES 0.474 / 0.66 / 0.729 |
| **G3 per-edge necessity** | For every edge, a strong reader given full context but with the predecessor's answer **masked out of the sub-question** must fail. | MuSiQue S3 — only **26.5%** of candidates survive; expect a similar kill rate |
| **G4 DiRe** | Split supporting chunks, run on each half independently, combine trivially. Whatever survives was never multi-hop. | Trivedi et al. 2020. Reference values: HotpotQA 68.8 / 2Wiki 63.4 / MuSiQue 37.8 |
| **G5 bridge leakage** | No intermediate entity's name or alias appears in the question. | MuSiQue `a2 ∉ q1`; 2Wiki structural rules |
| **G6 adversarial distractors** | Distractors are gold chunks of *other* paths sharing entity type — not random. | MuSiQue "positive distractors" |
| **G7 evidence accessibility** | Report Recall@k **and Support F1 separately**. The gap is the evidence that assembling the *complete* set is hard. | HopWeaver: 0.705 vs 0.222 |
| **G8 human audit** | Stratified by hop count. Report % judged genuinely multi-hop. | Comparables: HopWeaver 96.4–98.9, HotpotQA 92.8, MuSiQue 91.2 |

**Report the funnel** — candidates enumerated → survived structural filters → survived G3 → survived
G4 → human-audited. MuSiQue's 26.5% tail-survival is the norm to imitate. A generator that reports
only its output count is hiding its selection rate.

### External benchmarks

- **StrategyQA** is the best fit: real open-domain scale, implicit multi-hop (the strategy is not
  stated in the question, so KG traversal should shine), an **official Recall@10 scorer**, and a
  **stale 2022 SOTA of 0.457** with nobody publishing retrieval numbers since. Cost of entry: index
  the 2020-05-11 Wikipedia Cirrus dump ourselves. License probably MIT but **UNVERIFIED** — AI2's
  dataset page is dead; confirm before any commercial claim.
- **BEIR/HotpotQA** for the clean apples-to-apples retrieval comparison.
- **Bamboogle: dropped.** 125 questions, no corpus, no gold evidence — recall@k is not computable.
- **LoCoMo: excluded.** **CC BY-NC 4.0**, non-commercial.

---

## Open items blocking the check-in

1. Wizard emit-surface audit — in progress.
2. Retrieval/benchmark surface audit — in progress.
3. Office-format decision (D4) — needs an explicit call.
4. SSH access to the 3090 — deliberately not requested until the above close.
