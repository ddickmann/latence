# Task: the embedding Disambiguator on its hashing default is retired; its learned run decides the rest

Type: task
Mode: HITL (the learned run needs a GPU: the owner's machine)
Status: open
Blocked by: the learned run (a live request)
Emerges from: 50 (measured 15 Nov: retire on the hashing embedder)

## Question

On the text-side gold set (`benchmark/resolution-modules/`), the embedding Disambiguator on the
in-core hashing embedder (its default when no embedder is configured) lowers pairwise F1 by
0.043 and precision from 0.946 to 0.616 against exact surfaces: retired under the owner's rule.
The GPU stacks (`stacks/gpu-sota.yaml`, `stacks/gpu-sota-pod.yaml`) name it with a learned
embedder, which was not measured.

## Goal

- Its hashing default is no default any more: configured without an embedder it refuses with a
  typed error naming the embedder to configure, or the default becomes exact surfaces; an ADR
  states which.
- The learned run on the same gold set decides the GPU stacks: kept if it meets the rule
  (at least 0.02 F1 over exact surfaces at no loss of precision), retired otherwise.

## Live request

On the owner's machine, with a GPU: run `benchmark/resolution-modules/measure.py` with the
embedding Disambiguator configured as `stacks/gpu-sota.yaml` configures it
(`embedding.sentence_transformers` on `cuda`), and the GLinker linker as
`stacks/gpu-sota-glinker.yaml` does, and record both results beside `results.json`.
