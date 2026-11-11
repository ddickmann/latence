# Ticket 20 spike: where it stands and what is left

Status on 2026-11-11. **The model arms have run; the results and the recommendation are in
[`REPORT.md`](REPORT.md).** They ran on a rented pod with one 24 GB GPU (RTX PRO 4000
Blackwell, 48 CPU threads, 125 GB RAM; owner, 2026-11-11; replaces the L40S plan). This file
keeps the setup, the earlier CPU-only results and the list of what is left. Read the report
first, then
[`../issues/20-resolution-feasibility-spike.md`](../issues/20-resolution-feasibility-spike.md) and
[`../design/resolution-at-scale.md`](../design/resolution-at-scale.md).

## Final model set (owner, 2026-11-08)

| Role | Model (Hugging Face id unless noted) | Runtime |
|---|---|---|
| English NER and relation extraction | `knowledgator/gliner-relex-large-v1.0` | PyTorch, CUDA |
| Same-entity / type / relation decisions | `fastino/GLiNER2.5-Decide` **vs** `fastino/GLiDE-*`, head to head, smallest first (owner, 2026-11-11). Measured: GLiDE 0.4B wins every comparison; larger variants not run | CUDA; GLiDE in its own environment |
| Candidate retrieval (blocking ANN and linking retriever) | `google/embeddinggemma-2`, 256-d Matryoshka, together with char-ngram TF-IDF | CUDA |
| Pairwise reranking | `Qwen/Qwen3-Reranker-0.6B`, zero-shot **vs** LoRA fine-tuned | CUDA |
| Structured matching backbone | Splink 5 on DuckDB (Spark later) | CPU |
| Definitions and type-induction proposals | `claude-opus-5-5` (hosted; Batches API for bulk) | `llm.py`, tier `deep` |
| Explanations of review items | `claude-haiku-5-5` (hosted) | `llm.py`, tier `fast` |
| Graph completion (not this ticket) | `mgalkin/ultra_50g` **vs** Flock (github `jw9730/flock`) | CUDA (torch-scatter) |
| German phase (later) | `gliner-relex-multi-v1.0`, `fastino/gliner2.5-multi-v1`, `fastino/GLiNER2.5-multi-Decide` | CUDA |

The 2021–2023 models first downloaded (bge-small, bge-reranker-base, MiniLM-L6) are dropped;
`run_b.py` is rebuilt on the agreed set.

**Why this set:**
- **Licences:** Fastino Labs is a partner, and the owner arranges the licence once the spike shows the approach works. GLiDE's community licence therefore does not exclude it. Record the partner-agreement exception to ADR-0012 when the agreement exists.
- **Decide / GLiDE:** their probability is one *feature* of the calibrated score, not the final say. The Learn-then-Test bands still decide. They also give each review-band item a first verdict.
- **Splink stays the backbone.** It scales to warehouse size, explains each score field by field and needs no labels. Only the pairs it leaves uncertain are re-scored by the neural models.
- **The generative model is hosted** and sits behind the provider-neutral `TextLLM` protocol in `llm.py`, so a deployment can route calls through its own tenant. **The spike sends only synthetic or public data.** Sending client data needs an explicit egress decision per deployment. Still to check: whether Databricks model serving offers these models.

## Earlier results (4-CPU Linux, no GPU, structured side only)

| Run | Records | Auto-accept precision (realised) | Pairwise F1, machine only | Pairwise F1, with expert | Plain closure F1 | Notes |
|---|---|---|---|---|---|---|
| Estate 12k | 12,004 | 0.963 (Wilson LB 0.959) | 0.913 | 0.957 | 0.815 | existing cascade baseline F1 0.540 in 400 s |
| Splink `historical_50k` (public) | 50,578 | 0.971 | 0.868 | 0.904 | 0.874 | no reject band; 21.7% review; pair completeness 0.694 |
| Estate 100k, before the acronym cap | 98,971 | 0.979 (LB 0.978) | 0.867 | 0.961 | 0.799 | union pair completeness 0.931; 37 s; 1.9 GB |
| Estate 100k, with the acronym cap | 98,971 | 0.990 | 0.858 | 0.956 | n/a | union pair completeness 0.915; 22 s; 1.6 GB |
| Estate 1M, 5,000 labels | 989,658 | none certified | 0 (nothing merged) | 0.833 | 0.738 | union pair completeness 0.898; 72 s; 5.4 GB |
| Estate 1M, 50,000 labels | 989,658 | 0.980 (LB 0.980) | 0.857 | 0.857 | 0.738 | no review band left; 83 s; 5.5 GB |

- **Result files:** `results/estate_4k_with_baseline.json`, `results/historical_50k.json`, `results/estate_100k.json`, `results/estate_100k_acronym_cap.json`.
- **The acronym cap** (commit `7c9f590`) is the version meant to scale. The REPORT shows both 100k results side by side.
- **The 1M run finished on 2026-11-10** (989,658 records, 4 CPUs, 15 GB RAM, no GPU): 72 s end
  to end at 5.4 GB peak RSS, about 13,700 records per second and 1.26M pairs per second through
  the matcher. Before, blocking alone took 12.2 GB and scoring died. What changed is how pairs
  and clusters are held (integer arrays, the prediction table left in DuckDB, one sorted
  neighbour table for the clusterer), not the method: on the 12k estate, the 100k estate and
  the public 50k set the rewrite gives the same pairs, bands and clusterings as the code before
  it (`test_equivalence.py`, `equivalence.py`).
  - **The candidate set:** 29.5M pairs, about 30 per record, against 5 per record at 100k.
    Union pair completeness is 0.898. Two passes produce almost all of it: `name_core` 15.9M
    and `first_tok+city` 14.5M. Part of the growth comes from the generator (about 1,400 name
    stems, so names collide more as the estate grows); real estates also have heavy name
    collisions, so this is reported, not hidden.
  - **With the label budget used so far (5,000 labelled pairs) no accept band is certified.**
    Learn-then-Test abstains, and the machine alone merges nothing. The cause is the candidate
    set: only 2.8% of candidate pairs are true matches at 1M, against 16% at 100k, so 3,000
    uniformly drawn test labels hold about 84 matches, fewer than the 100 pairs the test needs
    above a threshold before it may certify it.
  - **With 50,000 labelled pairs the bars are met:** accept at 0.505, realised auto-accept
    precision 0.980 (Wilson lower bound 0.980), machine-only pairwise precision 0.986 and
    recall 0.758 (F1 0.857). 150,000 labels give the same. No review band is left at either.
  - **Where the recall goes:** 10% of true pairs are never candidates, and the reject band's
    1% tolerance, applied to 28.8M rejected pairs, lets through 0.53% of them as missed
    matches, about 153,000 pairs or 17% of all true pairs.
  - **What this means for the design (ticket 13):** uniform labelling over candidate pairs does
    not scale; 50,000 labels is about 280 hours of review at 20 s each, and 97% of those labels
    are obvious non-matches. Labels have to be drawn where the scores are uncertain or high,
    with the test adjusted for that sampling, and the reject tolerance has to be stated
    against true pairs, not against rejected pairs. Neither is built.
  - **Result files:** `results/estate_1m.json` (5,000 labels), `results/estate_1m_labels_50k.json`,
    `results/estate_1m_labels_150k.json`.
- **The text side** had only a lexical baseline then: top-1 accuracy 0.784, 0% on acronyms, and LTT abstained. The model arms are in `REPORT.md`.

## Pod setup

```bash
git clone -b claude/blissful-maxwell-bpln7a https://github.com/ddickmann/latence.git
cd latence/.scratch/semantic-sidecar/spike
python3 -m venv --system-site-packages .venv && source .venv/bin/activate   # keeps the image's CUDA torch
pip install splink duckdb pandas pyarrow scikit-learn scipy jellyfish transformers \
  sentence-transformers torchvision pillow gliner "gliner2[local]" peft accelerate anthropic pytest psutil
PYTHONHASHSEED=0 python -m pytest -q -p no:cacheprovider .   # no network, no model
export HF_HOME=/workspace/hf
python pin_models.py --glide 0.4B      # downloads every model, writes results/models.json (not tracked)
python smoke_models.py                 # loads each model once; proves the code path, not quality

# GLiDE pins its own torch and transformers: a second environment, used only for the glide arms
python3 -m venv .venv-glide && source .venv-glide/bin/activate
pip install -r models/GLiDE-0.4B/requirements.txt
pip install splink duckdb pandas pyarrow scikit-learn scipy jellyfish psutil pytest
pip install -c <(printf 'torch==2.14.0\ntransformers==5.17.0\n') sentence-transformers torchvision
python smoke_models.py glide
```

- **In the GLiDE environment the embedder does not load** (its pinned transformers does not
  know the architecture), so GLiDE text arms run without `--embed`.
- **The hosted model's key** (`ANTHROPIC_API_KEY`) is set in the shell and never committed. It
  was not set on the 2026-11-11 run, so no review-item explanations were generated.
- **A pod's `/workspace` may sit on the container disk.** Then everything is lost when the pod
  stops: copy `out/*.json` off the pod before stopping it.

`data/`, `out/` and `models/` are git-ignored. Regenerate the data:

```bash
python generate.py --entities 4000   --seed 11 --out data/estate_4k.parquet
python generate.py --entities 33000  --seed 13 --out data/estate_100k.parquet
python generate.py --entities 330000 --seed 17 --out data/estate_1m.parquet
python generate_text.py --out data/text_2k
python generate_text.py --entities 2000 --threads 1200 --seed 29 --out data/text_threads   # the acronym set
python -c "import pyarrow.parquet as pq; from splink import splink_datasets as s; pq.write_table(s.historical_50k, 'data/historical_50k.parquet')"
```

`run_scale.sh` runs all of Part A in sequence. Its 1M step now also runs on a 15 GB machine.

## Exports for the workbench prototype (ticket 22)

`run_a.py --export-review <file>` writes the review band as items a reviewer can decide
(`review_export.py`, schema `review-export/1`), and `--export-graph <file>` writes the machine
clusters as a graph (`graph_export.py`, schema `graph-export/1`; it needs `--export-review`).
Both are synthetic-estate only. `../workbench/README.md` has the exact command and where the
files go. On the pod, exports from the re-scored arms give the workbench items with the
neural scores in their evidence; that wiring is not written yet.

## A suggester for the review band (2026-11-10)

`suggest.py` fits a decision tree of depth 3 on the matcher's agreement level per field, using
labelled review pairs, two folds, so each pair's suggestion comes from a tree that did not see
its label. On the 12k estate's review band (1,424 pairs) the suggestion matches the truth for
96.4% of pairs out of sample; calling every pair the same would be right for 69.8%. The
workbench shows the suggestion with its conditions and that record.

- **Why it works here:** the matcher weighs each field on its own; what separates the review
  band is combinations (same postcode area, similar name, industry given: different in 164 of
  164 labelled pairs). A tree sees combinations.
- **What it does not show:** the labels are the generator's truth, and the generator's hard
  negatives follow a few patterns a tree can learn. A real estate will be harder. No real
  labels have been used.
- **For ticket 13:** a second stage over field-agreement combinations, fitted on the decisions
  reviewers record, may shrink the review band before any neural scorer is needed. It has to
  go through the same calibrated bands as every other score; a suggestion decides nothing.

## What the CPU smoke test showed (2026-11-09; not results)

All four small models load and run through `models.py` on a CPU: EmbeddingGemma 2 (256-d,
unit norm; it needs `torchvision` and `pillow` even for text), Qwen3-Reranker-0.6B, GLiNER-Relex
(entities with character offsets, and relations) and GLiNER2.5-Decide. GLiDE was first loaded
on the pod (2026-11-11).

One thing to watch when the real arms run: on three hand-written record pairs, zero-shot
GLiNER2.5-Decide gave nearly the same P(yes) for a clear match and a clear non-match, and its
answer flipped with the wording of the labels. Three examples prove nothing, and Decide is only
a feature of the calibrated score. **The measured arms confirmed it** (2026-11-11): zero-shot
Decide carries almost no weight on records. GLiDE 0.4B does the job; fine-tuning Decide (its
model card documents that) is the other option and has not been tried.

## What is left

Done on 2026-11-11 and reported in `REPORT.md`: the models are pinned, the structured arms ran
at 10⁵ and 10⁶ and on the public person dataset, the text arms ran on both text sets, and
acronyms have a resolver (`aliases.py`, `run_b.py --aliases`).

Open, in the order they matter for the decision:

1. **Done:** the owner decided go on 2026-11-11, with a budget of 4 hours for 10⁶ records on
   one 24 GB GPU.
2. **One permissively licensed English entity-linking benchmark.** Candidates and their
   licence questions are in the report; none is cleared.
3. **A real estate and real text,** even a small hand-labelled slice. Everything so far is
   synthetic except one public person dataset.
4. **A second stage without a model** (labels drawn from the undecided pairs, no scorer), and
   **a model pass just below the reject line** at 10⁶, where the missed matches are.
5. **The reviewer's alias loop:** a reviewer's choice for a bare acronym becomes a stored
   alias with its scope. Acronyms that are not organisations ("VP") need the extractor's
   typing measured.
6. **Larger GLiDE variants** (0.8B, 2B) only if the review band has to shrink further;
   **fine-tuning** only if a real estate misses the bar zero-shot.
7. **Review-item explanations** from the hosted fast tier (`llm.explain_review_item`), for
   ticket 16. They gate nothing.
8. **Close out:** record the model set in tickets 13 and 15.

**Go bar:**
- realised auto-accept precision ≥ 0.95 under Learn-then-Test (δ = 0.05), with its confidence stated;
- 10⁶ records within 4 hours on one 24 GB GPU (owner, 2026-11-11);
- no expert-capacity bar during development.
