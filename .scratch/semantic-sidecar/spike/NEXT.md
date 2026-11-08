# Ticket 20 spike: where it stands and what runs next (on the M3)

Status on 2026-11-08. The model arms run on the owner's Apple Silicon M3 (32 GB unified memory).
Everything so far ran on a 4-CPU Linux box without a GPU. Read this file, then
[`../issues/20-resolution-feasibility-spike.md`](../issues/20-resolution-feasibility-spike.md) and
[`../design/resolution-at-scale.md`](../design/resolution-at-scale.md).

## Final model set (owner, 2026-11-08)

| Role | Model (Hugging Face id unless noted) | Runtime |
|---|---|---|
| English NER and relation extraction | `knowledgator/gliner-relex-large-v1.0` | PyTorch on MPS |
| Same-entity / type / relation decisions | `fastino/GLiNER2.5-Decide` **vs** `fastino/GLiDE-*` (9B or smaller), head to head | MPS |
| Candidate retrieval (blocking ANN and linking retriever) | `google/embeddinggemma-2`, 256-d Matryoshka, together with char-ngram TF-IDF | MPS or MLX |
| Pairwise reranking | `Qwen/Qwen3-Reranker-0.6B`, zero-shot **vs** LoRA fine-tuned | MPS |
| Structured matching backbone | Splink 5 on DuckDB (Spark later) | CPU |
| Definitions and type-induction proposals | `claude-opus-5-5` (hosted; Batches API for bulk) | `llm.py`, tier `deep` |
| Explanations of review items | `claude-haiku-5-5` (hosted) | `llm.py`, tier `fast` |
| Graph completion (not this ticket) | `mgalkin/ultra_50g` **vs** Flock (github `jw9730/flock`) | CPU |
| German phase (later) | `gliner-relex-multi-v1.0`, `fastino/gliner2.5-multi-v1`, `fastino/GLiNER2.5-multi-Decide` | MPS |

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
- **The acronym cap** (commit `20c69f7`) is the version meant to scale. The REPORT shows both 100k results side by side.
- **The 1M run has not finished.** Earlier attempts ran out of memory or disk on the Linux box. The fixes are in: DuckDB-side union, EM trained on a sample of at most 150k records, SQL passes, and the cap.
- **Part B (text)** has only a lexical baseline: top-1 accuracy 0.784, 0% on acronyms, and LTT abstained.

## M3 setup

```bash
git clone -b claude/blissful-maxwell-bpln7a https://github.com/ddickmann/latence.git
cd latence/.scratch/semantic-sidecar/spike
python3.12 -m venv .venv && source .venv/bin/activate
pip install splink duckdb pandas pyarrow scikit-learn scipy torch transformers \
  sentence-transformers gliner gliner2 peft accelerate anthropic pytest psutil
export PYTORCH_ENABLE_MPS_FALLBACK=1   # ops without an MPS kernel fall back to CPU
export ANTHROPIC_API_KEY=...           # set in the shell; never commit it
PYTHONHASHSEED=0 python -m pytest -q -p no:cacheprovider test_er.py test_llm.py   # 13 pass
```

`data/`, `out/` and `models/` are git-ignored. Regenerate the data:

```bash
python generate.py --entities 4000   --seed 11 --out data/estate_4k.parquet
python generate.py --entities 33000  --seed 13 --out data/estate_100k.parquet
python generate.py --entities 330000 --seed 17 --out data/estate_1m.parquet
python generate_text.py --out data/text_2k
python -c "from splink import splink_datasets as s; s.historical_50k.to_parquet('data/historical_50k.parquet')"
```

`run_scale.sh` runs all of Part A in sequence. Its 1M step is the open one.

## Next steps, in order

1. **Pin the models.** Download every model in the table and write each one's id, revision hash and licence to `results/models.json`.
2. **Part A at scale on the M3:**
   - Rerun 100k, then 1M, with `run_a.py`.
   - Report wall time, peak RSS and pairs per second.
3. **Part A, uncertain pairs:**
   - Write out the pairs between the reject and accept bands as text (`name | city | postcode | reg_id …`).
   - Score them with Qwen3-Reranker-0.6B and with Decide or GLiDE.
   - Feed `[splink_logit, rerank, decide]` into a logistic calibrator, then `er.learn_then_test_bands`, then `er.ConstrainedClusterer.try_union_checked`. Keep `er.py` as it is.
   - Run these arms: Splink only; plus zero-shot scores; plus a LoRA-fine-tuned reranker trained on a disjoint labelled split.
   - The fine-tuned arm ships only if it clears 0.95 where zero-shot does not.
4. **Part B rebuilt in `run_b.py`:**
   - GLiNER-Relex extracts mentions from the raw documents; score extraction P/R against the span gold in `mentions.jsonl`.
   - Retrieve candidates with TF-IDF plus EmbeddingGemma 2, rerank them with Qwen3-Reranker, and add the Decide or GLiDE probability.
   - A logistic model, then LTT, sorts mentions into link, NIL or review. NIL mentions are then clustered.
   - Add one permissively licensed English entity-linking benchmark, checking the licence before use.
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
- 10⁶ records within a time and cost budget agreed once the 100k timing on the M3 is known;
- no expert-capacity bar during development.
