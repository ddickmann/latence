# Task: measure each older resolution module and keep or retire it

Type: task
Mode: AFK for what runs on a CPU; HITL for the linker path (its model)
Status: open
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
