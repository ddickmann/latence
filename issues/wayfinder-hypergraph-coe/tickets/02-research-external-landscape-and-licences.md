# 02 — Research: external landscape, datasets, dependency licences

Type: research
Status: resolved
Blocked by: —

## Question

External facts the decision tickets wait on (research-diligence directive applies — licences, IP, regulation in an explicit cited section; unverifiable = UNVERIFIED, never guessed):

a. **HyCE-RAG (arXiv 2607.22597)** — verify the paper exists as cited; its actual eval protocol and reported gains; adjacent SOTA lines (HippoRAG-2, hypergraph-RAG variants, budget-constrained evidence selection) and where the spec's zero-LLM adaptation sits vs them.
b. **Datasets** — MuSiQue and HotpotQA licences, redistribution constraints, canonical supporting-fact eval definitions; any precedent for "recall@token-budget" curves we should match for comparability.
c. **Dependency licences** (weights vs code separately where models are involved): pysbd, spaCy (senter model licences!), scipy, an LMDB binding (lmdb / plyvel alternatives), scikit-learn (isotonic regression), DuckDB. GLiREL relation-type scorer: code licence AND weight licence.
d. **Regulatory** — anything in the EU-AI-Act/GDPR direction that the EU-text sidecar (verbatim text spans + provenance in LMDB/parquet) changes vs the existing chunk store. Expected: nothing new beyond existing Purge obligations — verify, don't assume.

## Answer

Researched 2026-07-30 against primary sources (arXiv pages, upstream repos/LICENSE files via GitHub API, PyPI metadata, HF model cards, spaCy model release pages). Per-item verdicts first, cited licence section at the end.

### (a) HyCE-RAG — the paper is REAL, but its eval is not the one we should match

**Verdict: EXISTS as cited.** [arXiv:2607.22597](https://arxiv.org/abs/2607.22597) — *"HyCE-RAG: Hypergraph Chain-of-Evidence Retrieval-Augmented Generation for Explainable Multi-hop Question Answering"* (An, Zhang, Liang et al., submitted 2026-06-12). Not a placeholder.

**Its actual protocol** ([HTML v1](https://arxiv.org/html/2607.22597v1)):
- Datasets: HotpotQA, 2WikiMultihopQA, MuSiQue, plus GraphRAG-Bench Medical + Novel subsets.
- Metrics: **LLM-as-judge** — Answer Accuracy, Context Relevance, Faithfulness. It does **not** report the canonical EM/F1/supporting-fact metrics.
- Baselines: No-Context, vector RAG, GraphRAG, LightRAG (HyperGraphRAG cited but not compared).
- Headline: MuSiQue Acc 56.73% vs 27.12% best baseline (+29.6 abs); Faithfulness 44.57%→79.23%.
- **LLM usage: heavy.** gpt-oss-120b for entity/relation extraction at indexing; LLM calls for query entity extraction at retrieval. Evidence budget is a *count* cap (6 segments), not a token budget. **No recall@token-budget curves anywhere in the paper.**

**Consequence for the spec:** our zero-LLM adaptation diverges from HyCE-RAG on both the offline and online paths, and HyCE-RAG's reported gains are measured with LLM-judge metrics we cannot (and should not) reproduce for comparability. Part 5.6 should anchor on the datasets' canonical metrics + HippoRAG-2-style recall@k, not on HyCE-RAG's Acc/Rel/Faith numbers.

**Adjacent SOTA lines (where we sit):**
- **HippoRAG 2** ([arXiv:2502.14802](https://arxiv.org/abs/2502.14802), "From RAG to Memory"): passage **recall@5** + token-F1 (MuSiQue-style) on NQ/PopQA/MuSiQue/2Wiki/HotpotQA/LV-Eval/NarrativeQA. MuSiQue R@5 74.7, HotpotQA R@5 96.3; QA F1 48.6 / 75.5 with Llama-3.3-70B reader. Indexing uses **LLM OpenIE** (Llama-3.3-70B) — so it is not a zero-LLM comparator either, but its recall@5 numbers are the de-facto comparison line for graph-RAG retrieval quality.
- Hypergraph-RAG family, all LLM-extraction-based: HyperGraphRAG ([arXiv:2503.21322](https://arxiv.org/pdf/2503.21322v2.pdf)), HyperRAG n-ary facts ([arXiv:2602.14470](https://arxiv.org/html/2602.14470)), HKVM-RAG ([arXiv:2606.07218](https://arxiv.org/pdf/2606.07218)), "Optimizing Hypergraph-Based RAG" ([arXiv:2607.20506](https://arxiv.org/abs/2607.20506)).
- **No published zero-LLM (small-model-extraction) hypergraph-RAG line was found** — the spec's GLiNER-family adaptation is genuinely differentiated, and honest positioning is "recall-per-token efficiency at zero marginal LLM cost", not head-to-head Acc parity with LLM-indexed systems.

### (b) MuSiQue + HotpotQA — both usable; HotpotQA carries share-alike

- **MuSiQue**: **CC BY 4.0** (LICENSE in [StonyBrookNLP/musique](https://github.com/StonyBrookNLP/musique)). Redistribution fine with attribution. Canonical metrics (`evaluate_v1.0.py`): **answer F1 + supporting-fact F1** (Answerable variant); Full variant adds group answer/support sufficiency F1.
- **HotpotQA**: **CC BY-SA 4.0** ([hotpotqa.github.io](https://hotpotqa.github.io/)). Attribution + **share-alike**: any redistributed or derivative *dataset* artifacts (e.g. a re-chunked eval corpus we ship) must stay CC BY-SA 4.0 — do not fold them into the framework's code licence; eval *code* and reported *numbers* are unaffected. Canonical metrics: Answer EM/F1, **Supporting-fact EM/F1 (sentence-level)**, Joint EM/F1, in distractor (10-paragraph) and fullwiki settings.
- **recall@token-budget precedent: none canonical.** HippoRAG-2 and the graph-RAG line report count-based recall@k. The closest budgeted-evaluation precedent is the context-compression literature — LLMLingua/LongLLMLingua ([llmlingua.com](https://llmlingua.com/llmlingua.html), [LlamaIndex writeup](https://www.llamaindex.ai/blog/longllmlingua-bye-bye-to-middle-loss-and-save-on-your-rag-costs-via-prompt-compression-54b559b9ddf7)), RECOMP, AttentionRAG ([arXiv:2503.10720](https://arxiv.org/pdf/2503.10720)) — which report QA quality at fixed compression ratios/token budgets. **Recommendation:** report supporting-fact recall as a *curve over token budget* (our native protocol, no direct competitor to mimic) **plus** recall@5 and supporting-fact F1 at a standard budget so the numbers are comparable to HippoRAG-2 and the HotpotQA/MuSiQue leaderboards.

### (c) Dependency licences — one real blocker: GLiREL

All green except GLiREL; full table with sources in the licence section below. Summary verdicts:

| Dependency | Licence | Verdict |
|---|---|---|
| pysbd | MIT | OK |
| spaCy (code) | MIT | OK |
| spaCy senter — `xx_sent_ud_sm` | **CC BY-SA 3.0** (model artifact; trained on UD v2.8, CC BY-SA 3.0) | OK for use; share-alike only if we *redistribute* the model artifact — prefer install-at-setup, don't vendor |
| spaCy `de_core_news_sm` / `en_core_web_sm` | MIT / MIT | OK (preferred senter route for DE/EN) |
| scipy | BSD-3-Clause | OK |
| scikit-learn (isotonic) | BSD-3-Clause | OK |
| DuckDB | MIT | OK |
| lmdb (py-lmdb binding + LMDB C lib) | OLDAP-2.8 (permissive, OSI-approved, BSD-like) | OK |
| plyvel (LevelDB alternative) | BSD-3-Clause | OK as fallback |
| **GLiREL code** | **CONFLICTING**: no LICENSE file in repo; README says CC BY-NC-SA 4.0; PyPI metadata says Apache-2.0 | **UNVERIFIED / legal risk** |
| **GLiREL weights** (`jackboyla/glirel-large-v0`) | **CC BY-NC-SA 4.0** (HF model card) | **BLOCKER for commercial use** |

**GLiREL detail (the ticket's headline finding):** the [GitHub repo](https://github.com/jackboyla/GLiREL) contains **no LICENSE file** (verified via GitHub API contents listing; `/license` endpoint 404s). The README and the [HF model card](https://huggingface.co/jackboyla/glirel-large-v0) both state "GLiREL by Jack Boylan is licensed under **CC BY-NC-SA 4.0**" (NonCommercial-ShareAlike), while [PyPI metadata](https://pypi.org/project/glirel/) (v1.2.1) declares `license: Apache-2.0`. The weights are unambiguously NC per the model card. For an OSS framework meant to be commercially usable, **GLiREL weights cannot ship as a default dependency**. Options for the decision tickets: (1) ask the author to clarify/relicense; (2) make GLiREL an optional opt-in extra with an NC warning (mirroring the EmbeddingGemma opt-in pattern); (3) train/substitute a permissively-licensed relation scorer (e.g. GLiNER2's own relation extraction, Apache-2.0 ecosystem). Note the spec already names GLiNER2 relation extraction as the primary extractor — GLiREL is used for argument offsets; option 3 may be nearly free.

### (d) Regulatory — expected result CONFIRMED: no new obligation class, Purge must cover the sidecar

- **GDPR is format-neutral.** "Processing" (Art. 4(2) GDPR) covers storage in any form; the chunk store already holds verbatim text, so a verbatim-span sidecar is the *same category* of processing, not a new one. No new legal basis, DPIA trigger, or obligation class arises from moving spans into LMDB/parquet.
- **The existing obligations extend to the new copy.** Art. 17 (erasure) and Art. 5(1)(e) (storage limitation) attach to every copy: the **Purge path must also delete EU text spans + provenance rows** in the sidecar. Engineering note, not a legal delta: LMDB deletes in place; **parquet is immutable, so purge = file rewrite/compaction or tombstoning** — this must be designed in, or the sidecar silently violates the same Purge promise the chunk store keeps. Provenance pointers (doc IDs, offsets) can themselves be personal data when documents concern individuals — purge them with the spans.
- **EDPB Opinion 28/2024** ([edpb.europa.eu PDF](https://www.edpb.europa.eu/system/files/2024-12/edpb_opinion_202428_ai-models_en.pdf)) concerns when *AI models* count as anonymous and legitimate-interest bases for *model training*. A retrieval index storing verbatim text is plainly not anonymous — but that was already true of the chunk store; the opinion changes nothing between the two designs.
- **EU AI Act**: the framework is not a GPAI model provider (no Art. 53 obligations); Art. 10 data-governance duties apply only if the framework is embedded in a *high-risk system*, and then identically to chunk store and sidecar. **No delta.**

### Licences / IP / Regulation (cited)

- MuSiQue — CC BY 4.0: https://github.com/StonyBrookNLP/musique (LICENSE file in repo)
- HotpotQA — CC BY-SA 4.0: https://hotpotqa.github.io/ (licence statement on the official dataset page)
- pysbd — MIT: https://github.com/nipunsadvilkar/pySBD (GitHub API licence endpoint: `MIT`)
- spaCy — MIT: https://github.com/explosion/spaCy (GitHub API: `MIT`)
- spaCy `xx_sent_ud_sm` — CC BY-SA 3.0 (model + UD v2.8 training data): https://github.com/explosion/spacy-models/releases/tag/xx_sent_ud_sm-3.8.0
- spaCy `de_core_news_sm` — MIT: https://github.com/explosion/spacy-models/releases/tag/de_core_news_sm-3.8.0
- spaCy `en_core_web_sm` — MIT: https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.8.0
- scipy — BSD-3-Clause: https://github.com/scipy/scipy (GitHub API: `BSD-3-Clause`)
- scikit-learn — BSD-3-Clause: https://github.com/scikit-learn/scikit-learn (GitHub API: `BSD-3-Clause`)
- DuckDB — MIT: https://github.com/duckdb/duckdb (GitHub API: `MIT`)
- py-lmdb — OLDAP-2.8: https://raw.githubusercontent.com/jnwatson/py-lmdb/master/LICENSE ("The OpenLDAP Public License Version 2.8"); PyPI `lmdb` metadata: `OLDAP-2.8`
- plyvel — BSD-3-Clause: https://github.com/wbolster/plyvel (LICENSE.rst; PyPI classifier `BSD License`)
- GLiREL code — UNVERIFIED (no LICENSE file; README CC BY-NC-SA 4.0 vs PyPI Apache-2.0): https://github.com/jackboyla/GLiREL , https://pypi.org/project/glirel/
- GLiREL weights — CC BY-NC-SA 4.0: https://huggingface.co/jackboyla/glirel-large-v0 (model card licence statement)
- GDPR Arts. 4, 5(1)(e), 17 — https://eur-lex.europa.eu/eli/reg/2016/679/oj
- EU AI Act (Reg. 2024/1689), Arts. 10, 53 — https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- EDPB Opinion 28/2024 (AI models & personal data) — https://www.edpb.europa.eu/system/files/2024-12/edpb_opinion_202428_ai-models_en.pdf
- HyCE-RAG — https://arxiv.org/abs/2607.22597 ; HippoRAG 2 — https://arxiv.org/abs/2502.14802 ; HyperGraphRAG — https://arxiv.org/abs/2503.21322 ; HyperRAG — https://arxiv.org/abs/2602.14470 ; HKVM-RAG — https://arxiv.org/abs/2606.07218 ; Optimizing Hypergraph-Based RAG — https://arxiv.org/abs/2607.20506 ; AttentionRAG — https://arxiv.org/abs/2503.10720 ; LLMLingua — https://llmlingua.com/llmlingua.html

Everything above was verified at the linked primary source on 2026-07-30; the single UNVERIFIED item is the GLiREL *code* licence (contradictory declarations, no LICENSE file — treat as NC until the author clarifies).

## Comments

**2026-07-31 — GLiREL blocker MOOT (map session, verified against the repo).** The blocker in (c)
dissolves: the repo neither uses nor needs GLiREL. Relation extraction is the GLiNER2 unified fused
provider `fastino/gliner2-multi-v1` (Apache-2.0 weights, verified 2026-07-10, THIRD-PARTY-LICENSES.md:235)
with predecessor `knowledgator/gliner-relex-multi-v1.0` (Apache-2.0, :234). GLiNER2 relation output
natively carries per-relation head/tail confidence + char offsets resolved to true original offsets
(ADR-0031; `latence-extract-gliner2/provider.py`), persisted as
`RelationMention.head_mention_id`/`tail_mention_id` + `confidence` (`latence-core/contracts.py:1151`).
The mitigation "lean on GLiNER2's own relation extraction" is therefore not a fallback but the
existing state. spec.md carries an adaptation note; every GLiREL reference in it was replaced.
w_relq uses GLiNER2 zero-shot relation-type extraction on the query string.
