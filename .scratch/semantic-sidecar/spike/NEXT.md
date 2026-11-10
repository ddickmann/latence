# Ticket 20 spike: where it stands and what runs next (on the GPU pod)

Status on 2026-11-08. The model arms run on a Runpod pod: 1× L40S 48 GB, at least 64 GB system RAM,
about 150 GB volume, a PyTorch/CUDA image (owner, 2026-11-08; replaces the M3 plan). Everything
so far ran on a 4-CPU Linux box without a GPU. Read this file, then
[`../issues/20-resolution-feasibility-spike.md`](../issues/20-resolution-feasibility-spike.md) and
[`../design/resolution-at-scale.md`](../design/resolution-at-scale.md).

## Final model set (owner, 2026-11-08)

| Role | Model (Hugging Face id unless noted) | Runtime |
|---|---|---|
| English NER and relation extraction | `knowledgator/gliner-relex-large-v1.0` | PyTorch, CUDA |
| Same-entity / type / relation decisions | `fastino/GLiNER2.5-Decide` **vs** `fastino/GLiDE-*` (9B or smaller), head to head | CUDA (vLLM optional for GLiDE) |
| Candidate retrieval (blocking ANN and linking retriever) | `google/embeddinggemma-2`, 256-d Matryoshka, together with char-ngram TF-IDF | CUDA |
| Pairwise reranking | `Qwen/Qwen3-Reranker-0.6B`, zero-shot **vs** LoRA fine-tuned | CUDA |
| Structured matching backbone | Splink 5 on DuckDB (Spark later) | CPU |
| Definitions and type-induction proposals | `claude-opus-5-5` (hosted; Batches API for bulk) | `llm.py`, tier `deep` |
| Explanations of review items | `claude-haiku-5-5` (hosted) | `llm.py`, tier `fast` |
| Graph completion (not this ticket) | `mgalkin/ultra_50g` **vs** Flock (github `jw9730/flock`) | CUDA (torch-scatter) |
| German phase (later) | `gliner-relex-multi-v1.0`, `fastino/gliner2.5-multi-v1`, `fastino/GLiNER2.5-multi-Decide` | CUDA |

The 2021–2023 models first downloaded (bge-small, bge-reranker-base, MiniLM-L6) are dropped.
`run_b.py` still references them and has to be rebuilt (step 4).

**Why this set:**
- **Licences:** Fastino Labs is a partner, and the owner arranges the licence once the spike shows the approach works. GLiDE's community licence therefore does not exclude it. Record the partner-agreement exception to ADR-0012 when the agreement exists.
- **Decide / GLiDE:** their probability is one *feature* of the calibrated score, not the final say. The Learn-then-Test bands still decide. They also give each review-band item a first verdict.
- **Splink stays the backbone.** It scales to warehouse size, explains each score field by field and needs no labels. Only the pairs it leaves uncertain are re-scored by the neural models.
- **The generative model is hosted** and sits behind the provider-neutral `TextLLM` protocol in `llm.py`, so a deployment can route calls through its own tenant. **The spike sends only synthetic or public data.** Sending client data needs an explicit egress decision per deployment. Still to check: whether Databricks model serving offers these models.

## Results so far (4-CPU Linux, Part A only)

| Run | Records | Auto-accept precision (realised) | Pairwise F1, machine only | Pairwise F1, with expert | Plain closure F1 | Notes |
|---|---|---|---|---|---|---|
| Estate 12k | 12,004 | 0.963 (Wilson LB 0.959) | 0.913 | 0.957 | 0.815 | existing cascade baseline F1 0.540 in 400 s |
| Splink `historical_50k` (public) | 50,578 | 0.971 | 0.868 | 0.904 | 0.874 | no reject band; 21.7% review; pair completeness 0.694 |
| Estate 100k, before the acronym cap | 98,971 | 0.979 (LB 0.978) | 0.867 | 0.961 | 0.799 | union pair completeness 0.931; 37 s; 1.9 GB |
| Estate 100k, with the acronym cap | 98,971 | 0.990 | 0.858 | 0.956 | n/a | union pair completeness 0.915; 22 s; 1.6 GB |

- **Result files:** `results/estate_4k_with_baseline.json`, `results/historical_50k.json`, `results/estate_100k.json`, `results/estate_100k_acronym_cap.json`.
- **The acronym cap** (commit `7c9f590`) is the version meant to scale. The REPORT shows both 100k results side by side.
- **The 1M run has not finished.** The latest attempt (2026-11-09, 4 CPUs, 15 GB RAM, code at
  `7c9f590`) was killed for lack of memory: blocking ended after 74 s at 12.2 GB peak RSS, and
  scoring then died. A count inside DuckDB shows why:
  - 989,658 records give **29.5M candidate pairs** (about 30 per record), against 518k at 100k
    records (about 5 per record). Union pair completeness is 0.898.
  - Two passes produce almost all of them: `name_core` 15.9M and `first_tok+city` 14.5M.
  - The pairs are pulled into pandas with string ids, which is what takes the memory.
  - Part of the growth comes from the generator: it has about 1,400 name stems, so names collide
    more as the estate grows. Real estates also have heavy name collisions, so this is worth
    reporting, not hiding.
  - The run needs the pod's RAM (64 GB or more). Keeping pairs in DuckDB or Arrow with integer
    ids would cut the memory without changing the method.
- **Part B (text)** has only a lexical baseline: top-1 accuracy 0.784, 0% on acronyms, and LTT abstained.

## Pod setup

```bash
git clone -b claude/blissful-maxwell-bpln7a https://github.com/ddickmann/latence.git
cd latence/.scratch/semantic-sidecar/spike
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match the image's CUDA
pip install splink duckdb pandas pyarrow scikit-learn scipy jellyfish transformers \
  sentence-transformers torchvision pillow gliner "gliner2[local]" peft accelerate anthropic pytest psutil
export ANTHROPIC_API_KEY=...           # set in the shell; never commit it
PYTHONHASHSEED=0 python -m pytest -q -p no:cacheprovider .   # 23 pass, no network, no model
python smoke_models.py                 # loads each model once; proves the code path, not quality
hf download fastino/GLiDE-9B --local-dir models/GLiDE-9B && pip install -r models/GLiDE-9B/requirements.txt
python smoke_models.py glide           # GLiDE ships its own `glide` package and pinned requirements
```

GLiDE pins its own torch and transformers versions. If they clash with the rest, give GLiDE its
own venv and run only the `glide` arms from it.

`data/`, `out/` and `models/` are git-ignored. Regenerate the data:

```bash
python generate.py --entities 4000   --seed 11 --out data/estate_4k.parquet
python generate.py --entities 33000  --seed 13 --out data/estate_100k.parquet
python generate.py --entities 330000 --seed 17 --out data/estate_1m.parquet
python generate_text.py --out data/text_2k
python -c "from splink import splink_datasets as s; s.historical_50k.to_parquet('data/historical_50k.parquet')"
```

`run_scale.sh` runs all of Part A in sequence. Its 1M step is the open one.

## Exports for the workbench prototype (ticket 22)

`run_a.py --export-review <file>` writes the review band as items a reviewer can decide
(`review_export.py`, schema `review-export/1`), and `--export-graph <file>` writes the machine
clusters as a graph (`graph_export.py`, schema `graph-export/1`; it needs `--export-review`).
Both are synthetic-estate only. `../workbench/README.md` has the exact command and where the
files go. On the pod, exports from the re-scored arms give the workbench items with the
neural scores in their evidence; that wiring is not written yet.

## What the CPU smoke test showed (2026-11-09; not results)

All four small models load and run through `models.py` on a CPU: EmbeddingGemma 2 (256-d,
unit norm; it needs `torchvision` and `pillow` even for text), Qwen3-Reranker-0.6B, GLiNER-Relex
(entities with character offsets, and relations) and GLiNER2.5-Decide. GLiDE has not been
loaded anywhere yet.

One thing to watch when the real arms run: on three hand-written record pairs, zero-shot
GLiNER2.5-Decide gave nearly the same P(yes) for a clear match and a clear non-match, and its
answer flipped with the wording of the labels. Three examples prove nothing, and Decide is only
a feature of the calibrated score. If the measured arm confirms it, the options are the GLiDE
arm and fine-tuning Decide (its model card documents that).

## Next steps, in order

0. **Check the GPU:** `python -c "import torch; print(torch.cuda.get_device_name(0))"`.
1. **Pin the models.** Download every model in the table and write each one's id, revision hash and licence to `results/models.json`.
2. **Part A at scale on the pod:**
   - Rerun 100k, then 1M, with `run_a.py`.
   - Report wall time, peak RSS and pairs per second.
3. **Part A, uncertain pairs** (code: `rescore.py`, wired into `run_a.py --rescore`):
   - The pairs between the first-stage reject and accept bands are written out as text and
     scored by the named scorers. `[splink_logit, <scorers>]` feed a logistic calibrator, then
     Learn-then-Test on a labelled sample of the uncertain region, then the constrained clusterer.
   - Each stage runs at δ/2, so the combined accepted set keeps the 0.95 claim at δ = 0.05.
   - Arms to run at 100k (and at 1M once it fits):
     ```bash
     python run_a.py --dataset estate --path data/estate_100k.parquet --baseline-max 0 --out out/a_100k_splink.json
     python run_a.py ... --rescore rerank        --out out/a_100k_rerank.json
     python run_a.py ... --rescore decide        --out out/a_100k_decide.json
     python run_a.py ... --rescore glide         --out out/a_100k_glide.json
     python run_a.py ... --rescore rerank,decide --out out/a_100k_rerank_decide.json
     python run_a.py ... --rescore rerank,glide  --out out/a_100k_rerank_glide.json
     ```
   - **Not written yet:** the LoRA fine-tune of the reranker on a disjoint labelled split. It
     ships only if it clears 0.95 where zero-shot does not.
4. **Part B** (`run_b.py` is rebuilt on the agreed models):
   - Arms, each adding one thing to the one before:
     ```bash
     python run_b.py --data data/text_2k --scorer lexical                          --out out/b_lexical.json
     python run_b.py --data data/text_2k --scorer rerank                           --out out/b_rerank.json
     python run_b.py --data data/text_2k --scorer rerank --embed                   --out out/b_rerank_embed.json
     python run_b.py --data data/text_2k --scorer rerank --embed --decide decide   --out out/b_decide.json
     python run_b.py --data data/text_2k --scorer rerank --embed --decide glide    --out out/b_glide.json
     python run_b.py --data data/text_2k --scorer rerank --embed --mentions relex  --out out/b_end_to_end.json
     ```
   - `--mentions relex` extracts mentions from the raw documents with GLiNER-Relex and reports
     extraction precision and recall against the span gold; linking then works on the extracted
     surfaces.
   - **Not done yet:** one permissively licensed English entity-linking benchmark (check the
     licence before use).
5. **Review items:** `llm.explain_review_item` (fast tier) explains a sample of review-band pairs, synthetic data only. These are the inputs for ticket 16. Spot-check them; they gate nothing.
6. **`REPORT.md`:**
   - B-cubed and pairwise P/R/F1;
   - LTT auto-accept precision with δ and the Wilson lower bound;
   - review rate and expert minutes per 1,000 records (reported, not gated);
   - throughput at 10⁵ and 10⁶ records;
   - the cascade baseline;
   - every model with its revision and licence note.
7. **Close out:**
   - Resolve ticket 20 as go or no-go and update `../map.md`.
   - Record the model set in tickets 13 and 15.
   - Commit with neutral messages and no attribution trailers, then push to `claude/blissful-maxwell-bpln7a` (PR #6).

**Go bar:**
- realised auto-accept precision ≥ 0.95 under Learn-then-Test (δ = 0.05), with its confidence stated;
- 10⁶ records within a time and cost budget agreed once the 100k timing on the pod is known;
- no expert-capacity bar during development.
