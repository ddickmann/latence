# A Hyperedge is a third GraphScope emitted by Graph Assembly; its `tau_text` never holds what the corpus would not export

The m-ary Evidence Unit that chain-of-evidence retrieval selects over (ADR-0056) enters the domain
model as a **Hyperedge** — a third `GraphScope` on the existing `GraphRecord`, emitted by the
**Graph Assembly** Stage, not by a new Stage. It is **distinct from `Evidence`**, which keeps its
current meaning untouched. Its verbatim span text (`tau_text`) **follows the corpus redaction
variant**: the hypergraph may never carry text the exported RAG corpus would not carry.

## Context

The source spec calls the unit an "Evidence Unit (EU)". That name cannot enter this codebase.
CONTEXT already spends `Evidence` on a precise, load-bearing meaning — *the justification linking an
inference back to its source mentions, with confidence* — and already polices one collision around
it (`_Avoid_: Provenance`). A third overlapping term in the same neighbourhood would be a
domain-language defect, and this is the neighbourhood the retrieval layer leans on hardest.

The two concepts are genuinely different, not two names for one thing:

| | `Evidence` | Hyperedge |
|---|---|---|
| answers | *why did we believe this edge?* | *what text do we hand the model?* |
| holds | `mention_ids`, `document_ids`, `confidence`, short `snippet` | verbatim `tau_text`, entity set, relation set, `token_cost` |
| raw content | **forbidden** (`snippet` is a surface string or strategy name) | **is the point** |
| arity | attached to a binary edge | m ≥ 2 entities at once |
| consumer | audit | retrieval selection |

Collapsing them would break the audit contract in both directions: `Evidence.snippet` is
deliberately barred from raw content, while `tau_text` is nothing but raw content.

## Decision

- **The domain term is `Hyperedge`.** CONTEXT gains the entry, with `_Avoid_: Evidence Unit, EU,
  fact, claim`. `Evidence` is unchanged and keeps its meaning; a Hyperedge may *carry* Evidence for
  its relations.
- **It is a third `GraphScope` (`NODE` / `EDGE` / `HYPEREDGE`) on `GraphRecord`**, with a
  `GraphHyperedge` payload, following the established scoped-record pattern (ADR-0025 `FeatureScope`,
  ADR-0026 `DisambiguationScope`, ADR-0027 `GraphScope`). It therefore inherits, without re-earning
  any of it: Provenance/Classification threading, uniform checkpoint/resume, content-addressed
  deterministic ids (`make_graph_id`), the S11 affected-set delta key, and Export.
- **Graph Assembly emits it — no new Stage.** A Hyperedge needs canonical `entity_ids`, so it can
  only exist post-Disambiguation, which is exactly where Graph Assembly sits. It is the same input
  re-projected m-ary instead of binary: Graph Assembly already turns disambiguated entities and
  normalised relations into nodes and edges; hyperedges are a third projection of that same pass.
- **`tau_text` follows the corpus redaction variant.** The redaction floor is enforced **at the
  Hyperedge writer**, not inherited by luck from whichever stream extraction happened to read. When
  redaction is on, `tau_text` is masked; raw `tau_text` exists only where the exported corpus is
  itself raw. The Hyperedge keeps its span in **original coordinates** for Provenance, offset
  round-trip, and Purge.
- **Weak Hyperedges** (entities present, no extractable relation) are **emitted by default behind a
  config toggle**, carrying `has_relations=false` and a utility discount. They contribute
  propagation mass; they are also expected to outnumber relation-bearing units, so the toggle exists
  and the eval (ticket 17) decides the default's fate.
- **Upstream prerequisite:** per-mention **link confidence must be persisted** through Disambiguation.
  It is currently computed by glinker and dropped before it reaches any durable record, which leaves
  `gamma` (spec 1.5) and weighted incidence (spec 6.1) with no data source at all. This lands with
  M1.

## Considered alternatives

- **Reuse / extend `Evidence`.** Rejected: opposite content rules (raw forbidden vs raw required),
  opposite arity (binary vs m-ary), opposite consumer (audit vs retrieval). The overlap is thematic,
  not structural.
- **Name it "Evidence Unit" anyway.** Rejected: a third term colliding with `Evidence` and
  `Provenance` in the glossary's most load-bearing neighbourhood.
- **A new Stage producing its own Record type.** Rejected: it would re-implement Provenance
  threading, checkpointing, deterministic ids, delta keys, and Export — all of which the scoped
  `GraphRecord` already provides — for no gain, since it consumes exactly Graph Assembly's inputs.
- **Store raw `tau_text` and let the consumer redact at query time.** Rejected: it pushes a
  compliance guarantee into the adopter's integration code, and it leaves unredacted PII in sidecar
  files sitting next to corpus files that are redacted. That is the ADR-0042 truncation-leak defect
  class, reintroduced through a new door.

## Consequences

- `GraphRecord` gains a third scope, so `SCHEMA_VERSION` bumps and the scope/payload consistency
  validator grows a branch — the pattern is built for exactly this.
- Where `extract_on: masked` is **off**, entity/relation offsets are in unmasked coordinates while
  the stored text is masked, so the writer needs an offset alignment. This is the problem
  `offset_map` already solves for markup stripping (ADR-0031); copy that pattern, do not invent one.
- Retraction and Purge (ADR-0029) reach hyperedges through the same content-addressed keys as nodes
  and edges; the derived KV (ADR-0056) is rebuildable, so Purge may rebuild rather than edit in place.
- The Quality Report gains hyperedge counts (total, weak fraction, arity distribution, token-cost
  distribution) — the token-cost distribution is the one that predicts whether budgeted selection
  will behave, so it is worth reporting from the first run.
