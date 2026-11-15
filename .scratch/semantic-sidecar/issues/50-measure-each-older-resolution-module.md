# Task: measure each older resolution module and keep or retire it

Type: task
Mode: AFK for what runs on a CPU; HITL for the linker path (its model)
Status: measured on CPU (15 Nov): cascade retire, embedding (hashing) retire, Type Consolidation keep; open: the GPU runs (GLinker, learned embedder) as a live request
Blocked by: nothing
Emerges from: 13 (decided by the owner, 14 Nov: "an older module (the disambiguation cascade,
embedding and linker paths, type consolidation) is kept only where a measurement shows it adds
value, and retired otherwise")

## Question

Four older modules resolve on the text path: the string cascade of `disambiguation.cascade`,
the embedding Disambiguator (`latence-disambig-embedding`), the GLinker linker
(`latence-disambig-glinker`) and Type Consolidation. Does each add value over what the new
resolution does?

## Goal

One measurement per module on a gold set it can be scored on, against the same set without the
module, with the decision rule stated before the run: kept if it adds at least the stated gain
at no loss of precision, retired otherwise.

## Done check

- A recorded result per module, with the command that produced it.
- A keep-or-retire line per module, and a ticket for each retirement.

## State (14 Nov): not built in this workstream

- **What is measured already:** the string cascade, on the spike's 12k synthetic record estate:
  pairwise F1 0.540 in 400 s, against 0.913 for the Fellegi-Sunter matcher alone on the same
  records ([spike report](../spike/REPORT.md)). That is the warehouse side; the text side, where
  the cascade, the embedding Disambiguator and Type Consolidation actually run, has no gold set in
  the repository (`matrix/disambiguation.yaml` says so: its quality column is merge counts, not
  precision and recall).
- **What this needs first:** a text-side gold set of mentions and their canonical entities (the
  spike's `generate_text.py` writes one for its own runs) and a decision rule stated before the
  run (proposed: keep a module only if it adds at least 0.02 pairwise F1 at no loss of
  precision at the accept line, else retire it).
- **CPU-only, so buildable here later:** the cascade, `disambiguation.embedding` (on the in-core
  hashing embedder; the learned embedder needs a GPU) and Type Consolidation. **Live request:**
  `disambiguation.glinker` needs its model on a GPU (`LATENCE_CUDA=1 latence bake-off
  matrix/disambiguation.yaml` on the owner's machine once the gold set is declared on the stack).

## Result (15 Nov)

The decision rule, fixed in the harness before the run: **kept when pairwise F1 rises by at
least 0.02 over the same run without the module and precision does not fall; retired
otherwise.** Harness, gold sets and recorded result: `benchmark/resolution-modules/`
(`measure.py`, `gold-mentions.json`, `gold-types.json`, `results.json`). Command:
`PYTHONHASHSEED=0 uv run python benchmark/resolution-modules/measure.py`; a second run gives the
same scores.

- **The gold sets.** Mentions: 684 mentions of 179 organisations in 500 synthetic documents
  (the spike's `generate_text.py`, seed 7, 150 entities: noisy surfaces and homonyms that differ
  by city or industry). Type labels: 65 labels of 16 types, as free-form extractors write them,
  with two pairs that must stay apart.
- **Without a module**: for mentions, the cascade restricted to exact normalised surfaces
  (P 0.946, R 0.265, F1 0.414); for type labels, labels equal once lower-cased (P 1.000, R 0.242,
  F1 0.390).

| Module | P / R / F1 with it | F1 gain | Precision change | Verdict |
|---|---|---|---|---|
| string cascade (`disambiguation.cascade`) | 0.754 / 0.304 / 0.433 | +0.019 | −0.192 | **retire** (ticket 60) |
| embedding Disambiguator, hashing embedder | 0.616 / 0.265 / 0.371 | −0.043 | −0.330 | **retire** on this embedder (ticket 61) |
| Type Consolidation, lexical rungs | 1.000 / 0.879 / 0.936 | +0.546 | 0.000 | **keep** |
| GLinker linker (`disambiguation.glinker`) | not run: needs its model on a GPU | | | live request (61) |

What this does not show: the gold set is template prose, easier than real text; the cascade's
fuzzy rungs fire on the homonyms the generator plants, which is where its precision goes; the
learned embedder may behave otherwise (ticket 61's live run decides it). Type Consolidation
missed the synonyms no lexical rule can see (people, place) and never merged a pair that must
stay apart.

### Live request

On the owner's machine (GPU): the GLinker linker and the embedding Disambiguator on the learned
embedder over the same gold set, as ticket 61 states.
