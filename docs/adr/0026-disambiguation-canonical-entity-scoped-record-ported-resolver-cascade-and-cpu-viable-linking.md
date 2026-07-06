# Disambiguation emits scoped DisambiguationRecords over ported resolver/linking/normaliser assets, CPU-viable with an audited, no-silent-over-merge log

S8 adds **Disambiguation** (CONTEXT): the corpus-level Stage — the algorithmic crown — that
links entity mentions to canonical entities and merges duplicates across documents, with
auditable, confidence-weighted merge decisions and Evidence back to source mentions. Five
design questions had non-obvious answers; they extend ADR-0004/0007 (the Capability +
CPU-first-reference pattern) and mirror how ADR-0025 handled the S7 corpus-level Stage, with
the concrete Disambiguation contract, the corpus-level multi-input seam, the ported
algorithmic assets, and the "no silent over-merge" audit discipline.

## A DisambiguationRecord is a per-scope inter-Stage Record, like the FeatureRecord

Disambiguation is corpus-level and produces two kinds of output — **canonical entities**
(the entity-linking + merge half) and **normalized relations** (the relation-normalisation
half). Rather than two record types, a single `DisambiguationRecord` carries a
`DisambiguationScope` tag (`ENTITY` / `RELATION`) and exactly one populated payload
(`CanonicalEntity` / `NormalizedRelation`), exactly as `FeatureRecord` carries a
`FeatureScope` (ADR-0025). It is a versioned core contract (`SCHEMA_VERSION` → 8), a subclass
of `Record`, so it carries Provenance and Classification like every other inter-Stage
carrier, threads through the DAG, is checkpointed/resumed uniformly, and exports to the
corpus directly (the canonical entities + normalized relations are the AI-ready artifact a
downstream Graph Assembly Stage reads). The scope/payload consistency is contract-enforced,
so a Provider cannot emit a mis-scoped record.

**Evidence is its own contract, distinct from Provenance.** CONTEXT separates the two:
Provenance is *source lineage* (which file/page/offset a record came from); **Evidence** is
*justification for an inference* (which mentions made us believe two surfaces name one
entity, and how strongly). A `CanonicalEntity` and a `MergeDecision` each carry `Evidence`
back to the source mentions/offsets — so no merge is a silent, unexplained act — while their
Provenance remains the canonical mention's page-accurate lineage. This is the first Stage
where the CONTEXT `Evidence` term is instantiated as a contract; Graph Assembly (S9) reuses
it verbatim for per-edge justification.

## The canonical entity keeps page-accurate Provenance (the S8 page-map sub-task)

A `CanonicalEntity`'s Provenance is inherited from its **representative mention** — the
first (sorted) member — whose Provenance already carries the exact `char_start`/`char_end`
and the source `page_start`/`page_end` resolved through `PageOffsetIndex` at S4. So a
canonical entity cites "page 7 of the contract", not "character 41,213 of an opaque blob" —
the S8 "wire PageOffsetIndex so every canonical entity keeps page-accurate Provenance"
sub-task, satisfied by *inheriting* the upstream resolution rather than re-deriving it (the
mention offsets are the coordinate system the whole pipeline shares). `source_document_ids`
records the cluster's cross-document reach — the affected-set key S11 will key incremental
recompute off.

## Corpus-level and multi-input, off Entity Extraction (required) + Relation Extraction (optional)

Disambiguation depends on **Entity Extraction** for the `EntityMention`s it resolves and
links — **required**, because a Disambiguation with no mentions has nothing to disambiguate
(unlike Profiling, whose mention side is optional, ADR-0025). It **optionally** depends on
**Relation Extraction** for the `RelationMention`s it normalises onto the canonical entities;
an entity-only Disambiguation pipeline is valid. A *fused* Entity+Relation parent (ADR-0013)
supplies both from one mixed carrier. The Runner partitions the gathered inputs by carrier
and errors only if a non-empty input carries no `EntityMention` (a mis-wired Entity edge).

## The ported algorithmic assets — PORT-AS-IS behind the Disambiguator seam

The three strongest v1 assets (ASSET-INVENTORY.md, all PORT-AS-IS) are re-homed faithfully
in the pure `latence_core.disambiguation` engine, keeping the *algorithmic value* verbatim
while dropping vendor coupling:

- **EntityResolver** (`resolution/entity_resolver.py`, 5/5): the **5-strategy cascade**
  (exact → alias → acronym → substring → embedding), a **Union-Find** merge map, and a
  **confidence-weighted, audited merge log**. The cascade is tried in descending precision so
  the strongest signal wins and the reason is auditable; Union-Find points every cluster at
  its lexicographic-min canonical, so the canonical text is deterministic regardless of merge
  order.
- **GLinker linking pipeline** (`linking/backends.py`, 4.5/5): the L2 candidate-gen → L3
  biencoder → L4 rerank → L0 threshold cascade with a **KB-text fallback**. The CPU reference
  keeps the cascade *shape* verbatim (candidate generation by normalised-token overlap over
  an inverted index, a deterministic lexical similarity standing in for the biencoder/rerank
  score, and an L0 acceptance threshold); the FAISS/GPU biencoder is the pluggable Provider
  upgrade.
- **RelationNormalizer** (`relations/normalizer.py`, 4.5/5): GLinker-style relation L2
  resolution — **fuzzy + alias** label normalisation, **type filtering**, and **inverse
  detection + direction swap** (an `employs` relation whose configured inverse is `works_for`
  is rewritten to the canonical direction with head/tail swapped).

`PageOffsetIndex` (the fourth named asset for this slice) is reused from core, not re-ported.

## No silent over-merge, and CPU-viable with no embedder and no KB (the S8 acceptance bar)

Two acceptance criteria shaped the reference Provider. **No silent over-merge:** every
candidate merge the cascade proposes is recorded as a `MergeDecision` with its strategy,
confidence, and Evidence; a decision below the `min_confidence` policy is recorded with
`applied=False` and is **not** folded into a canonical entity. The audit is attached to the
entity and rolled up in the Quality Report (applied vs below-policy, by strategy). **CPU
reference required, GPU optional:** the `disambiguation.cascade` Provider ships in core,
pure-Python and deterministic, and works with **no embedder** (the lexical token-similarity
stand-in for the EMBEDDING strategy, which can be disabled for the strict no-embedder path)
and **no KB** (an empty KB makes every entity fall back to unlinked — the "configurable KB
with a graceful fallback for unlinked mentions" criterion). A FAISS/GPU blocking Provider is
a pluggable option behind the same `Disambiguator` seam, **not** required — so first
`git clone` → working disambiguation is hash-stable for the Baseline bar, with no new model,
no weights/code license to verify (ADR-0012), and no heavy dependency (ADR-0016). A future
learned biencoder/embedder is a post-v1 Provider behind this same seam.

## Providers shipped (CPU-first reference, ADR-0007)

* **`disambiguation.cascade`** ships in `latence-core`: the deterministic resolver cascade +
  Union-Find + audited merge log, GLinker-style KB linking with graceful fallback, and the
  relation normaliser — all pure-Python, no extra deps.

The merge/link statistics land in the Quality Report as `DisambiguationQuality`: mentions in
vs canonical entities out, the created/merged/singleton breakdown, applied vs below-policy
merges by strategy, KB-linked vs unlinked, and relations normalised — counts only (no raw
document text) — satisfying the S8 "merge/link statistics land in the Quality Report"
criterion.
