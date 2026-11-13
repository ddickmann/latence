# The semantic layer is a replay of an append-only ledger of Assertions and Decisions

**Status:** Accepted (2026-11-09)

The re-scoped engine (ADR-0067) keeps one source of truth: an append-only ledger of Assertions
and Decisions. The semantic layer anyone reads (Concepts, Entities, Bindings) is rebuilt by
replaying that ledger. This record also pins the vocabulary and who may accept what. The owner
decided each point on 2026-11-09 (map ticket *Destination and domain model*). Nothing here is
built yet; the terms enter `CONTEXT.md` with the first slice that implements them.

## Context

Research for the re-scope found that every comparable tool traces answers, not semantic
elements, and that auto-drafted semantic layers lose human curation when they are rebuilt. It
also found that no ontology-learning method is accurate enough to accept schema-level elements
unattended (taxonomy F1 of about 0.5–0.6), while calibrated entity matching can reach a stated
precision for an auto-accept band. The product has to keep human decisions across rebuilds and
delta runs, and has to say for every element where it came from and who accepted it.

## Decision

1. **Two levels, treated differently.**
   - The *instance level* holds Entities. The engine may accept merges and links there, only
     inside the calibrated accept band.
   - The *schema level* holds Concepts. Schema-level elements are only ever proposals that a
     human approves.
2. **Vocabulary.**
   - **Concept:** a schema-level element: a class, a property or relation, or a measure or
     dimension. A measure is a Concept bound to a Unity Catalog metric-view measure, not a
     separate kind.
   - **Entity:** one canonical real-world thing, made by merging records and mentions.
   - **Assertion:** one statement harvested from one source. It always carries a Locator and
     the Actor that produced it.
   - **Locator:** where an Assertion came from: a text span, a Unity Catalog object, a query, a
     lineage edge or a value sketch.
   - **Decision:** an accept, reject, merge, split or edit made on top of Assertions, with its
     Actor and the evidence it used.
   - **Actor:** the engine (rules and calibrated bands), a model (name and version) or a human.
   - **Binding:** a tie from the semantic layer to a Unity Catalog object, keyed on Unity
     Catalog object ids and carrying a role (for example *defines*, *identifies*, *measures*).
     A *Concept binding* ties a Concept to a table, column or metric-view measure. An *Entity
     binding* ties an Entity to the rows that represent it (object id plus primary-key value),
     which yields a cross-system crosswalk.
3. **The ledger.** Assertions and Decisions are append-only. Nothing in the semantic layer
   exists without at least one Assertion behind it. A correction is a new Decision, never an
   edit of an old record. The current layer is a deterministic replay of the ledger.
4. **Authority.**
   - The engine may accept instance-level merges and links, only inside the accept band.
   - A model's verdict accepts nothing by itself. It is a feature of the score or a suggestion
     shown to the expert.
   - Schema-level changes, splits of an existing Entity and any write-back to Unity Catalog
     need a human.
   - A human Decision outranks the engine and is never overwritten by a later run.
5. **The line to Unity Catalog.** The engine never defines a table or a metric itself. It binds
   to what Unity Catalog defines, adds what Unity Catalog lacks (synonyms, descriptions, the
   class hierarchy, relations between Concepts, the Entity crosswalk) and sends anything that
   would change Unity Catalog upstream as a proposal for a human to apply. Drafting new
   metric-view measures as proposals is in scope for v1.

## Alternatives considered

- **A mutable semantic layer with an audit log.** Familiar, and cheaper to build. Rejected: an
  audit log records that something changed but cannot rebuild the layer, so byte-identical
  replay and curation that survives a rebuild would both have to be bolted on afterwards.
- **Assertions only, with verdicts as more Assertions.** One record kind instead of two.
  Rejected: accept, reject, merge and split would have to be encoded inside statements, and the
  rule that a human verdict outranks the engine would have no place to live.
- **Instances only in v1.** Smaller. Rejected: the product would launch without a schema level,
  which is the ontology the re-scope promises.
- **Auto-accepting schema-level elements above a confidence threshold.** Less expert work.
  Rejected on the measured accuracy of ontology learning, and on the standing rule that an
  unknown stays unknown.
- **A separate Metric kind next to Concept.** Rejected: Unity Catalog already defines measures,
  so a measure here is a Concept with a Binding, and a second kind would invite a second
  definition.
- **Owning metric definitions where Unity Catalog has none.** Rejected: it creates a parallel
  source of truth. Proposals upstream keep Unity Catalog authoritative.

## Consequences

- The contracts need a structured Locator next to character offsets, an Actor on every
  Decision, and the ledger records themselves. Each is a `SCHEMA_VERSION` change.
- Replay has to be deterministic, so the ledger order and every tie-break are part of the
  contract.
- Proposing new measures overlaps with what Metis does with metric views. Where that line
  runs is decided in the map ticket *Boundary with Metis*.
- v1 is done when seven checks hold; they are listed in the map.
