# S8 — Disambiguation (algorithmic crown)

**Parent**: S5 · **Phase 1 · Baseline bar**

## What to build
The corpus-level **Disambiguation** Capability: link entity mentions to canonical entities and merge duplicates across documents, with auditable, confidence-weighted merge decisions (no silent over-merge) and Evidence back to source mentions. This is the differentiator versus parse-only tools — it ports the strongest algorithmic IP in the stack. Keep it CPU-viable for the demo; the FAISS-GPU blocking path is an upgrade.

## Internal sub-tasks (do in order)
1. **Page-map integration** — wire PageOffsetIndex so every canonical entity keeps page-accurate Provenance.
2. **Entity resolver port** — EntityResolver 5-strategy cascade (exact → alias → acronym → substring → embedding) + Union-Find merge map + audited merge log.
3. **Linking port** — GLinker pipeline L2 candidate-gen → L3 biencoder → L4 rerank → L0 threshold, with KB-text fallback; permissive `knowledgator/gliner-linker-*` reference Providers.
4. **Relation normalizer port** — batched embedding resolution, type filtering, inverse detection + direction swap.
5. **Integration** — behind the single Disambiguation Capability; affected-set-ready (records provenance keys for S11).

## Ports (algorithmic assets — all PORT-AS-IS)
- EntityResolver (`resolution/entity_resolver.py`, 5/5) · GLinker (`linking/backends.py`, 4.5/5) · RelationNormalizer (`relations/normalizer.py`, 4.5/5) · PageOffsetIndex.

## Acceptance criteria
- [ ] Cross-document mentions resolve to canonical entities; each merge decision records reason, confidence, and Evidence (source mentions/offsets).
- [ ] No silent over-merge: merges below the confidence policy are not applied and are logged.
- [ ] Entity linking runs against a configurable KB with a graceful fallback for unlinked mentions.
- [ ] CPU reference path works on the sample corpus; GPU/FAISS blocking is a pluggable Provider option, not required.
- [ ] Merge/link statistics (created/merged/split, confidence) land in the Quality Report.
- [ ] Baseline bar met.

## Blocked by
S5.
