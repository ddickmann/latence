# 12 — Task: M1 — Hyperedge segmentation + builder

Type: task
Status: open
Blocked by: 03, 04

**CHECK IN with Dennis before starting — first long-running implementation on this map.**

## Question

Implement spec 1.2–1.5 as a **Graph Assembly** extension per
**[ADR-0057](../../../docs/adr/0057-hyperedge-is-a-third-graphscope-emitted-by-graph-assembly-and-tau-text-follows-corpus-redaction.md)**
(domain model) and **[ADR-0056](../../../docs/adr/0056-hypergraph-evidence-sidecar-is-read-per-call-over-a-point-lookup-kv-no-global-edge-adjacency.md)**
(what the store does and does not hold). Terminology: **Hyperedge**, never "Evidence Unit"/"EU".

Scope:

1. **Persist per-mention link confidence** through Disambiguation (glinker computes it, it is
   currently dropped — [ticket 01](01-research-spec-vs-repo-audit.md)). Prerequisite: without it
   `gamma` has no data source. Do this first; it is a contract change.
2. **Sentence splitting** — abbreviation-safe DE/EN. Today's is a naive `[.!?]\s` regex
   (`chunking.py:226`); pysbd (MIT) or a spaCy senter (MIT for de/en pipelines — **not** the
   CC BY-SA multilingual one, [ticket 02](02-research-external-landscape-and-licences.md)).
3. **Hyperedge segmentation** — window extension driven by relation head/tail argument offsets
   (recoverable by joining `RelationMention.head_mention_id`/`tail_mention_id` to `EntityMention`
   spans; no schema change needed), minimal-cover spans, merge/split to the 40–120 token target,
   weak Hyperedges behind the config toggle, `m_max=12` with relation-group splitting.
4. **`GraphHyperedge` payload + `HYPEREDGE` GraphScope** on `GraphRecord`; `SCHEMA_VERSION` bump;
   scope/payload consistency validation.
5. **`tau_text` follows the corpus redaction variant**, enforced at the writer, span kept in
   original coordinates (ADR-0057). Offset alignment via the ADR-0031 `offset_map` pattern.
6. **`gamma`** from persisted link confidence + relation confidence + weakest-mention penalty;
   calibration itself is [ticket 09](09-grilling-calibration-ownership.md)'s call — uncalibrated
   weighted raw scores are acceptable for M1.
7. **Dedup/merge** on identical entity-set AND relation signature, `support` counter, highest-γ
   instance's `tau_text` kept primary.
8. **Quality Report**: hyperedge counts, weak fraction, arity distribution, **token-cost
   distribution** (the one that predicts whether budgeted selection will behave).

Tests (spec M1 + the ADR-0057 additions): every relation's arguments fall inside its Hyperedge span;
no Hyperedge exceeds 180 tokens; merge is idempotent; **`tau_text` never contains a span the exported
corpus masked** (the PII-bypass regression test); deterministic ids are stable across runs.
Repo DoD bar applies.
