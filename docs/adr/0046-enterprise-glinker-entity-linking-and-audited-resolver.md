# Enterprise GLinker entity-linking + audited resolver, ported from dataset-intelligence

`disambiguation.glinker` (package `latence-disambig-glinker`) adds the **enterprise** entity-linking
+ resolution engine behind the existing S8 `Disambiguator` seam (ADR-0026): the neural GLinker
L2→L3→L4→L0 linking pipeline + the audited 5-strategy resolver, ported PORT-AS-IS from the private
`latenceai-dataset-intelligence` repo. It is a **drop-in** for the in-core `disambiguation.cascade`
/ `disambiguation.embedding` — same `DisambiguationRecord` contract, swapped by one config line
(ADR-0007). This ADR records why it is a new Provider package (not a core change), how the two
ported halves map onto the framework contracts, and the enterprise-honesty posture.

## Why a new Provider package, not another in-core reference

ADR-0026 ported the *CPU-viable reference* of these assets into `latence_core.disambiguation`: the
5-strategy `EntityResolver` with **lexical `token_similarity`** standing in for the biencoder, and a
`KnowledgeBase` doing the GLinker cascade shape with lexical scoring. That reference is deliberately
dependency-free and deterministic — the CPU path the S8 AC requires.

`disambiguation.glinker` is the *other* end of that same seam: the **real neural engine**. It runs
the actual `knowledgator/gliner-linker-large-v1.0` biencoder (L3) + `gliner-linker-rerank-v1.0` (L4)
through the **`glinker`** library's `ProcessorFactory` — the Knowledgator package (PyPI `0.1.x`,
Apache-2.0) that provides `ProcessorFactory` and the L2/L3/L4 processors, the direct import
`from glinker import ProcessorFactory` in `backends.py`. (This is a **distinct package** from the
`gliner` zero-shot extraction library used by the NER/relation/PII providers; the audit caught the
pyproject declaring `gliner` while the code imports `glinker` — corrected to `glinker>=0.1,<0.2`,
and `gliner` dropped since nothing imports it here.) Its resolver's embedding-by-label merge uses
a **real learned STS embedder** (IBM Granite Embedding r2, ADR-0045) with an optional FAISS-GPU path
for large label groups. That stack (`glinker`/`flashDeBERTa`/`transformers`/`torch`/`faiss`) is
heavy and GPU-bound, so — exactly like the GLiNER extractor + Granite embedder packages (ADR-0016) —
it ships in its OWN package, NOT in the thin core and NOT in the CPU-first dev/CI env. The framework
never names a model; installing the package registers `disambiguation.glinker` under
`latence.providers` (ADR-0004) and any Pipeline opts in with one config line.

## The seam mapping (two ported halves → one Stage)

The Provider subclasses `CascadeDisambiguator` + `AdapterBase` and overrides ONLY the
blocking/merge step (`_resolve_clusters`), inheriting all record building, external-KB linking
(`kb_id`), relation normalisation, page-accurate Provenance/Evidence, and deterministic emission —
the same override pattern the `disambiguation.embedding` Provider uses (W1 rework, #94). Inside:

1. **Corpus-internal KB + neural linking** (`kb_builder` + `backends` + `linker`, ported from
   `linking/*`) — a self-referencing KB is built from the run's `EntityMention`s (dominant-label
   vote, aliases, context descriptions); the GLinker L2→L3→L4→L0 pipeline links each mention to its
   canonical entry over ±window-context inputs (the ±200-char context is sourced from the run's
   `ChunkRecord`s, coordinate-correct through the chunk `offset_map` per ADR-0031 — the source read
   it from document markdown); a **direct-KB text match** catches what the neural pass misses and
   **provisional entities** hold the unresolved. Produces canonical entity clusters.
2. **Audited resolver** (`entity_resolver`, ported from `resolution/entity_resolver.py`) — fuses
   those clusters by exact → alias → acronym → substring → **fuzzy edit-distance** → *(opt-in)*
   Granite-STS embedding-by-label, with transitive-merge + cycle detection. The **`MergeDecision`
   audit log** (§19: no silent over-merge) records every merge with reason + confidence +
   provenance. Type-constrained: the acronym/substring/fuzzy rungs never cross entity labels; the
   embedding rung groups by label. **Precision default:** the embedding rung is OFF by default and
   the conservative fuzzy rung is ON — see *Runtime fixes* below.

The resolved canonical entities are lifted onto the engine's `ResolvedCluster` (with the audited
`MergeDecision`s mapped to engine `MergeAudit`s — attached to the surviving entity **by surface
name**, robust to the evidence-chosen survivor id), so the inherited `_entity_record` stamps the
identical `DisambiguationRecord` contract. Model→framework record mapping: `MentionRecord` ←
`EntityMention`; `CanonicalEntityRecord` → `ResolvedCluster` → `CanonicalEntity`.

The **DI adaptation** is confined to two seams — context comes from `ChunkRecord`s (not document
markdown), and the STS embedder is the resolved framework `Embedder` (Granite r2, NOT
EmbeddingGemma/vllm-factory). The linking + fusion ALGORITHM is byte-for-byte the enterprise one.

## Enterprise-honesty posture

- **`compute="gpu"`, `deterministic=False`.** The neural gliner-linker biencoder needs a GPU, so on
  a CPU-only host the device seam skips-with-flag (conformance records it device-skipped alongside
  the OCR Parsers — never faked). `deterministic=False` is declared honestly: the neural biencoder +
  FlashDeBERTa is not bitwise-reproducible across BLAS/hardware. The string-cascade resolver rungs
  ARE deterministic and the cluster emission ORDER is sorted (canonical-text), so the record stream
  is stably ordered.
- **Robust fallbacks — never crash.** A backend load/inference failure (no GPU, missing weights,
  import error) degrades to the deterministic direct-KB + string-cascade path; the FAISS-GPU
  embedding-merge path falls back to brute-force when `faiss` is absent; a mention with no locatable
  chunk falls back to the bare surface.
- **No silent over-merge.** Every applied merge carries reason + confidence + provenance-bearing
  Evidence, surfaced as contract `MergeDecision`s in the Quality Report.

## What ships vs what defers

The **transformers path is the complete enterprise default** — the full L2→L3→L4→L0 pipeline + the
audited resolver ship today. The alternate **vLLM serving backend** (`VLLMGLiNERBackend`, which needs
the vllm-factory `deberta_gliner_linker` plugin) is a **documented, seam-compatible follow-on**: the
same `InferenceBackend` seam, so a stack swaps to it with no Provider change once the plugin is
vendored. This is NOT a sacrifice of the algorithm — the full GLinker + audited resolver runs; only
the alternate serving backend defers, and ProviderProfile `device=cuda` device-honesty covers it.

## License diligence (weights AND code, separately; verified 2026-07-14)

Per the research-diligence directive / ADR-0012, weights and code are verified separately with
citations (see `THIRD-PARTY-LICENSES.md`). All permissive — **no non-permissive finding**:

| Dependency | Type | SPDX | Cite |
|---|---|---|---|
| `knowledgator/gliner-linker-large-v1.0` | weights | Apache-2.0 | model card `license: apache-2.0` — https://huggingface.co/knowledgator/gliner-linker-large-v1.0 |
| `knowledgator/gliner-linker-rerank-v1.0` | weights | Apache-2.0 | model card `license: apache-2.0` — https://huggingface.co/knowledgator/gliner-linker-rerank-v1.0 |
| `glinker` (the imported `ProcessorFactory` package, Knowledgator) | code | Apache-2.0 | PyPI `glinker` (Apache Software License) + https://github.com/Knowledgator/GLinker (LICENSE) |
| `flashdeberta` | code | Apache-2.0 | PyPI classifier "Apache Software License" + https://github.com/Knowledgator/FlashDeBERTa |
| `transformers` / `huggingface-hub` | code | Apache-2.0 | HF project licenses |
| `faiss` (optional; `faiss-cpu`) | code | MIT | https://github.com/facebookresearch/faiss (LICENSE) + https://pypi.org/project/faiss-cpu/ |
| `ibm-granite/granite-embedding-{311m,97m}-multilingual-r2` (STS merge) | weights | Apache-2.0 | ADR-0045 |

**Caveat (recorded honestly):** the official GPU Faiss build ships via conda
(`conda install -c pytorch faiss-gpu`); the PyPI `faiss-gpu` wheel is a third-party-maintained build
(declared MIT). Prefer conda for GPU provenance, or `faiss-cpu` (MIT, the actively-maintained PyPI
wheel). Faiss is OPTIONAL — the resolver falls back to brute-force when it is absent.

## Enterprise-audit corrections (fixed in the port, not carried over)

A three-dimension enterprise audit of the port surfaced correctness gaps — two are bugs the port
**corrects vs the reference source** (the source has them too), documented here so the divergence
from PORT-AS-IS is deliberate and traceable:

- **Confusable-surface record_id collision (A-F1) — FIXED.** The framework derives a canonical
  entity's `record_id` as `entity:{normalize_text(canonical_text)}` (NFKC + casefold), but the
  resolver grouped exact-name/alias by `str.strip().lower()`. Two compatibility-confusable surfaces
  (the ligature `ﬁle` vs `file`, fullwidth Latin, super/subscripts) therefore survived as **two**
  resolver clusters that the framework then folded onto the **same** `record_id`, silently dropping
  one cluster's provenance. Fixed by grouping on the framework's own `normalize_text`, injected into
  `EntityResolver(name_normalizer=…)` so agreement is **guaranteed by construction** (same callable);
  the engine keeps a byte-identical in-package default for standalone use. Regression-tested with a
  ligature surface (no id collision, no dropped mention provenance).
- **Dangling `canonical_entity_id` after an evidence-flipped merge (A-F2) — FIXED (also a source
  bug).** `_apply_entity_merges` keeps the **highest-evidence** member as the surviving id, but
  `_update_mention_ids` remapped mentions to the **merge-map transitive root**. When a low-evidence
  root absorbs a higher-evidence entity (e.g. the acronym `IT` folds the higher-evidence
  `Information Technology`, whose id becomes the survivor), mentions pointed at a non-emitted id.
  Fixed by threading the actual **survivor map** from `_apply_entity_merges` into
  `_update_mention_ids`. Regression-tested (evidence=3 acronym absorbing evidence=1 full name →
  every mention points at the emitted survivor).

The audit also confirmed a deliberate **asymmetry vs `CascadeDisambiguator` (B-F3)**: the in-core
cascade records *near-miss / not-applied* merge candidates as `MergeAudit(applied=False)` for
transparency, whereas the ported resolver logs **only applied merges** (its rungs short-circuit on
the first accepted merge and never materialise a scored not-applied candidate). This is disclosed,
not silently divergent: every *applied* merge still carries reason + confidence + provenance (§19 no
silent over-merge holds); only the negative (below-policy) candidates are absent from the audit. A
future enhancement can emit `applied=False` near-misses from the embedding rung for full parity;
until then the Provider maps an unmapped resolver reason to `EMBEDDING` **with a logged warning**
(B-F5) rather than silently, so a newly-added rung cannot mislabel its audit unnoticed.

## Runtime fixes (pod-validated on a German legal/shareholder corpus)

The port's tests used a faithful **fake** backend + a deterministic fake STS embedder — which hid
three runtime defects that only surface against the real `glinker`/`transformers` stack and real
short entity names. All three are fixed in the package (tested with faithful, anti-false-green
fakes) and are wired into the defaults so the CLI is correct out of the box:

- **FIX-1 — embedding-by-label rung OFF by default (precision).** The Granite-STS embedding rung
  catastrophically over-merged on real data: every distinct person collapsed into one, distinct
  dates and orgs merged (49 false merges). Granite is a **retrieval** embedder — on short entity
  names its cosines for genuinely-distinct names (0.85–0.89) overlap the right merges (0.86–0.95), so
  **no clean threshold exists**. `use_embedding_merge` now defaults **`False`** in `EntityResolver`,
  the Provider config, this ADR, `stacks/gpu-sota-glinker.yaml`, and (by the Provider default) any
  hand-written or wizard-emitted glinker stage. Enabling it stays opt-in, defaults to a safe **≥0.93**
  cutoff, and is documented to REQUIRE a discriminative (non-retrieval) embedder. Recall is carried
  by the neural linker + the string cascade + the fuzzy rung (FIX-3), not the embedding rung.
- **FIX-2 — the neural gliner-linker was DEAD at runtime (`token_lengths`).** `gliner` 0.2.x injects
  `token_lengths` (and `word_lengths`/`packing_config`/`pair_attention_mask`) into the model input and
  its `base.py` forwards them into the encoder, but the gliner-linker backbone is a plain
  `transformers` DeBERTa whose `forward()` **rejects** `token_lengths` → every neural batch raised →
  silent degrade to direct-KB (`entities_linked: 0`). The backbone does not use those kwargs, so
  stripping them is safe; validated live (stripping `token_lengths` → 227/232 mentions neural-linked).
  `GLinkerBackend._init_executor` now applies a **contained** monkeypatch (never a site-package edit):
  it walks the just-built executor for the encoder module(s) and wraps each `forward` to drop only the
  kwargs its real signature lacks — a forward-compatible no-op once a backbone accepts them, logged
  once at DEBUG. A faithful test (`test_glinker_encoder_forward_patch_strips_token_lengths`) asserts an
  un-patched forward raises and the patched one succeeds — the test that would have caught the dead
  neural path.
- **FIX-3 — conservative fuzzy edit-distance rung (OCR-variant recall).** Four OCR variants of one
  person (`Kai Kölsch / Kai Kolsch / Kai Koisch / Kai Kusch`) stayed four entities: the string cascade
  can't (not exact/substring), the embedding rung over-merges, and the corpus-internal KB is
  per-unique-text so the neural pass won't consolidate. The reference documented an edit-distance rung
  via `min_similarity` but left it a **dead knob** (audit B-F4). It is now implemented in
  `EntityResolver` (batch + incremental): same-label only, normalized `difflib` char-similarity ≥
  `min_similarity` (default **0.84**), optional shared-doc guard, type-constrained, transitive-safe,
  and audited (`reason="fuzzy_edit_distance"`, surfaced in the Quality Report under the SUBSTRING
  strategy bucket — the closest lexical kin — with the exact technique legible in the decision reason).
  It merges `Kai Kölsch ↔ Kai Kolsch` (0.90) and the legit `Kai Kusch ↔ Kai Kölsch` two-edit variant
  (0.84) but never `Kai … ↔ Dennis Dickmann` / `Max Mustermann` (ratios far below); heavier OCR noise
  (the full four-variant set) is consolidated by lowering `min_similarity` into its precision-safe band
  (the negatives sit ≤ 0.35). Enabled by default.
- **FIX-3b — distinguishing-token guard (fuzzy rung; pod-observed on a German legal/shareholder
  corpus).** The char-ratio alone cannot separate an OCR spelling-variant from two entities that
  differ ONLY in a *distinguishing token*: `§ 3 Abs. 2 GwG` vs `§ 11 Abs. 6 GwG` (0.83) are distinct
  legal paragraphs, and `K² Ventures UG` vs `D² Ventures UG` (0.93) are distinct orgs — yet the latter
  scores HIGHER than the legit `Kai Kusch ↔ Kai Kölsch` merge (0.84), so no threshold can split them.
  Before a fuzzy merge is allowed (same-label + ratio ≥ `min_similarity`) the rung now additionally
  requires BOTH sub-guards (each individually unit-tested): (1) **numeric tokens equal** —
  `_numeric_tokens` NFKC-folds (`§ 3`→`3`, `K²`→`K2`) and extracts the digit set; `{3,2}` ≠ `{11,6}`
  ⇒ BLOCK. (2) **leading token is not a distinct short code** — `_leading_code_differs` fires when the
  first tokens are equal length, differ only in the leading char, and are ≤3 chars or all-caps
  (`k2`/`d2` ⇒ BLOCK). `Kai Kölsch`/`Kai Kolsch` has no numbers and shares the leading token `Kai`, so
  both sub-guards pass and the OCR variant still merges; `Notare`/`Notar` and `Fragebogens`/`Fragebogen`
  (length-mismatched leading tokens, no numbers) still merge. The guard only ever **blocks** a merge —
  it never creates one — so it can only prevent a false merge, never introduce one. Applied in both the
  batch and incremental fuzzy paths.

## Consequences

- The roster grows to 44 Providers; the conformance sweep device-skips `disambiguation.glinker` on
  CPU (still passing — a SKIP is never a FAIL) and validates it on a GPU pod.
- A new `stacks/gpu-sota-glinker.yaml` = `gpu-sota-pod.yaml` with `disambiguation.embedding` →
  `disambiguation.glinker` (Granite embedder retained for the resolver's STS merge). Added to the
  Disambiguation bake-off matrix as the enterprise (GPU, skip-with-flag on CPU) candidate.
- The CPU-first, offline, deterministic dev/CI env is unchanged (the heavy stack installs `--no-deps`
  from source for type-checking + roster discovery; the tests exercise the ported algorithm with a
  faithful fake backend + a deterministic fake STS embedder — anti-false-green).
