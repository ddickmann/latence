# latence-label-profiler

The non-LLM corpus profiler behind `label_inducer.lexicon` (ticket **T-LABEL-1**).

It samples a corpus, mines a candidate type space, selects it under a cardinality budget by
**greedy submodular max-coverage**, and emits the `candidates` block of a `label_inducer.lexicon`
config — plus an artifact that reports how much it does *not* know.

**Zero dependencies. Zero network. Zero model. Byte-reproducible.**

---

## Why it emits a pool and not a schema

The obvious reading of "sample the corpus, take the top 40 types, no LLM" is a trap, and the trap is
documented in `issues/e2e-extended/DECISIONS.md` §D5a:

- With no `induce` stage, `effective_labels()` returns the config labels unchanged
  (`latence-core/src/latence_core/induced.py:146-179`), so a corpus-level top-40 would be scored
  against **every chunk**.
- That is exactly the corpus-schema-applied-per-chunk behaviour the T1–T7 extraction redesign
  removed, and exactly the zero-shot degradation §D5b measures: GLiNER2 is a **uni-encoder**
  (labels and text share one sequence, cost `O((n+m)²)`); its author recommends **20** labels, its
  docs report degradation past **30**, its training cap is 25, and measured CPU latency rises
  **+60%** from 5 to 50 labels.

So this package does deliberately **half** the job:

| Step | Who | What |
|---|---|---|
| 1 | **this package** | corpus-level candidate **pool** (≤ N types per slot), from a sample |
| 2 | `label_inducer.lexicon` (already shipped) | narrows the pool **per chunk** to the types that chunk evidences |
| 3 | extraction | unions each chunk's induced set onto the config floor; ADR-0044's PII floor stays non-bypassable |

Only step 1 was missing.

## Method

Every method is a published algorithm implemented here from the paper — no library was adopted.

| Concern | Method | Source |
|---|---|---|
| Candidate phrases | stopword-segmented content runs | RAKE — Rose, Engel, Cramer & Cowley (2010) |
| Relation phrases | `V \| VP \| VW*P` between two argument phrases, plus the lexical constraint | ReVerb — Fader, Soderland & Etzioni (2011) |
| Selection | greedy submodular max-coverage over mention mass, `\|A\| ≤ N` | Nemhauser, Wolsey & Fisher (1978) — `(1 − 1/e) ≈ 63.2%` |
| Optimality of that ratio | nothing polynomial beats it unless P = NP | Feige (1998) |
| Knee of the coverage curve | Kneedle | Satopää, Albrecht, Irwin & Raghavan (2011) |
| Residual unseen mass | `p₀ = N₁ / N` | Good (1953) |
| Rank-frequency fit | discrete power-law MLE + log-log rank regression | Clauset, Shalizi & Newman (2009) |

**Coverage is defined as the downstream matcher's semantics**, not as a convenient proxy: a
candidate covers a surface exactly when its token sequence is a contiguous sub-sequence of that
surface — which is precisely when `LexiconLabelInducer`'s compiled `\b(?:cue)\b` fires. The coverage
number reported here is therefore the coverage the pipeline will actually deliver.

Selection uses **naive** greedy rather than CELF/lazy greedy on purpose. CELF returns the same set,
but only naive greedy evaluates *every* candidate's marginal gain at *every* step — and those gains
are the certificate and the gain curve.

## What the artifact refuses to claim

There is **no published standard** for a "90% coverage" or "marginal gain < x%" label-budget cutoff.
Under a Zipf distribution with `α ≈ 1`, coverage of the top-N types grows like `H(N) ~ ln N` and
**no sharp knee exists**. When the fit lands in that regime, the artifact says so outright:

> No sharp knee: the Kneedle difference peak is 0.248, below the 0.5 convention declared in
> `stats.SHARPNESS_THRESHOLD`. N is a budget decision, not a discovered boundary. There is no
> published standard for a coverage or marginal-gain cutoff.

The 0.5 sharpness line is **our stated convention**, not Kneedle's — the algorithm locates a knee, it
does not adjudicate whether one is real — so `max_difference` is always reported and a reader can
draw a different line. The 40 cap is likewise labelled a budget; **T-LABEL-2** measures the real N on
the pod.

Each slot reports: the marginal-gain curve, `C(N)`, the per-corpus optimality certificate
(`f(OPT_N) ≤ f(A_i) + Σ of the N largest remaining gains`, minimised over prefixes and clamped to the
total mass — usually far tighter than 63.2%), the fitted Zipf exponent, and the Good–Turing residual.

## PII safety

- A structurally-detected PII value is **masked out of the text before phrase mining**, so no account
  number or personal name can become an entity type.
- A detected value is **never** echoed into a cue list. Emitted cues are contextual keywords only.
- When a type's whole mass came from its detector, it is flagged `detector_only` with a note saying
  per-chunk induction will fire on a chunk that *talks about* the type, not on one carrying a bare
  value. Bare values are ADR-0044's non-bypassable redaction floor's job, not this pool's.

## Known limits, stated rather than discovered later

- **Relation recall is bounded by a POS-tagger-free heuristic** (closed verb set + conservative
  suffix morphology). A thin relation pool is evidence about the miner, not necessarily about the
  corpus. The artifact carries this as a per-slot note.
- `-ing` is excluded from the verb suffixes because "heading" / "building" / "meeting" / "training"
  are nouns and common in corporate text; a genuine gerund relation is missed in exchange.
- `-ed` is kept, but only trims a phrase-leading token when a copula precedes it, so
  `combined nomenclature` stays a noun phrase while `is classified under` becomes a relation.
- The PII slot's candidates are **declared** (a fixed table) and their coverage sets are disjoint, so
  greedy is exactly optimal there by construction — its certificate ratio of 1.0 is not evidence of a
  good corpus fit, and the artifact says so.

## Usage

```python
from latence_label_profiler import CorpusProfiler, ProfilerConfig

profile = CorpusProfiler(ProfilerConfig(seed=0, max_types=40)).profile(texts)

candidates = profile.lexicon_candidates()   # -> the label_inducer.lexicon `candidates` block
artifact = profile.to_json()                # -> curves, certificate, Zipf fit, residual, caveats
```

```yaml
induce:
  provider: label_inducer.lexicon
  config:
    granularity: chunk
    whole_word: false        # German compounding
    candidates: <profile.lexicon_candidates()>
```

Or from the shell:

```bash
latence-label-profile ./corpus --pattern '*.txt' --max-types 40 \
    -o profile.json --candidates-out candidates.json
```

## Integration: three release-gate edits this package requires, in files it does not own

Adding a 33rd package trips three `latence-core` release gates **by design** — they exist precisely
so a new distribution cannot ship without the release evidence being updated. Verified by causation,
not inference: with `packages/latence-label-profiler/` removed, both files pass in full; with it
present, exactly these three fail.

| Test | Why it fires | Fix |
|---|---|---|
| `latence-core/tests/test_release_integrity.py::test_releasing_dry_run_evidence_matches_the_tree` | `docs/RELEASING.md` records a dry run that built **32** packages; the tree now holds 33. `release.yml` builds `for pkg in packages/*/`, so the release genuinely changed. | Re-run the dry run and update the evidence block. |
| `…::test_releasing_sbom_component_arithmetic_is_consistent` | The SBOM evidence block records **32** workspace path entries (`docs/RELEASING.md:155`). | Re-run `uv export --all-packages`, update the count and the component arithmetic. |
| `latence-core/tests/test_publish_readiness.py::test_project_urls_did_not_swallow_dependencies` | It asserts `[project.dependencies]` is non-empty for **every** package, as a proxy for "a `[project.urls]` sub-table did not re-parent the `dependencies` key". This package has **genuinely zero** runtime dependencies, which that proxy cannot distinguish from the footgun. | Give the guard a way to express "deliberately dependency-free" — e.g. a `_zero_dependency_packages = {"latence-label-profiler"}` exemption alongside the existing `_standalone_roots`, still asserting the KEY is present (`"dependencies" in proj`) so the footgun stays caught. |

The third one was not worked around by declaring an unused `latence-core` dependency. Metadata that
claims a dependency the code never imports is a lie that survives forever, and this package's whole
claim — zero network, zero cost, zero model, reproducible anywhere — is exactly the claim a spurious
dependency would make unverifiable.

## Licensing

Framework code, **Apache-2.0**. Ships **no weights and no model**, so there is no weights licence to
verify (ADR-0012 requires weights and code to be checked separately; here the weights side is empty).

The reference *implementations* of the methods above were checked and **not adopted**: `kneed`
(BSD-3-Clause per its PyPI metadata) requires numpy + scipy and the CPU-first dev env is deliberately
numpy-free (ADR-0016); `yake` / `rake-nltk` / c-TF-IDF pull nltk or scikit-learn, and nltk's runtime
corpus download would violate the zero-network bar outright. A licence we do not rely on is not
load-bearing.

The stopword lists are hand-written English and German function words. Function words are not a
copyrightable creative work and the lists are not copied from any package's data file.
