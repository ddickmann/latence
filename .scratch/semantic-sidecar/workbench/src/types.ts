// The shapes the workbench reads and writes. The review and graph exports are what the engine
// writes (`review-export/1`, `graph-export/1`); decisions and intents are what a reviewer adds.

export type Band = "accept" | "review" | "reject";
export type Scalar = string | number | null;

export interface SourceRecord {
  unique_id: string;
  [field: string]: Scalar;
}

/** One comparison the matcher made between the two records, and what it contributed. */
export interface EvidenceRow {
  comparison: string;
  label: string;
  /** The record field the comparison was derived from. */
  field: string;
  /** -1: missing on at least one side; otherwise the agreement level, 0 the weakest. */
  level: number;
  level_label: string;
  /** Match weight in bits: positive speaks for a match, negative against. */
  weight: number;
  left: Scalar;
  right: Scalar;
}

export interface ReviewItem {
  id: string;
  group: string | null;
  band: Band;
  probability: number;
  match_weight: number;
  prior_weight: number;
  left: SourceRecord;
  right: SourceRecord;
  evidence: EvidenceRow[];
  why_unsure: string;
  vetoed: boolean;
  /** The generator's truth. Present on synthetic data only. */
  gold_same?: boolean;
}

export interface GroupEdge {
  left: string;
  right: string;
  band: Band;
  probability: number;
}

/** A connected set of review pairs: one cluster a reviewer settles together. */
export interface ReviewGroup {
  id: string;
  rank: number;
  priority: number;
  records: string[];
  labels: Record<string, string>;
  edges: GroupEdge[];
  items: string[];
}

export interface ReviewMeta {
  source: string;
  kind: string;
  synthetic: boolean;
  records: number;
  candidate_pairs: number;
  bands: { accept_at: number | null; reject_below: number | null };
  counts: Record<Band, number>;
  review_groups_total: number;
  review_groups_exported: number;
  record_fields: string[];
  comparisons: { comparison: string; label: string; field: string }[];
}

export interface ReviewExport {
  schema: string;
  meta: ReviewMeta;
  groups: ReviewGroup[];
  items: ReviewItem[];
  audit: ReviewItem[];
}

export type Verdict = "same" | "different" | "unsure";

export interface Actor {
  id: string;
  name: string;
}

/** One entry in the append-only decision log. A later entry for the same item supersedes. */
export interface Decision {
  item: string;
  verdict: Verdict;
  reason?: string;
  note?: string;
  actor: Actor;
  /** ISO 8601, UTC. */
  at: string;
}

export interface DecisionInput {
  item: string;
  verdict: Verdict;
  reason?: string;
  note?: string;
}

/** A reviewer's statement about a whole group. Recorded; applying it is the engine's job. */
export interface GroupIntent {
  group: string;
  intent: "merge" | "split";
  note?: string;
  actor: Actor;
  at: string;
}

export interface Ledger {
  decisions: Decision[];
  intents: GroupIntent[];
}

// ---------- Graph ----------

export type NodeType = "organisation" | "city" | "country" | "industry" | "source_system";

export interface Concept {
  id: NodeType;
  label: string;
  kind: "entity" | "reference" | "provenance";
  count: number;
  definition: string;
}

export interface Relation {
  id: string;
  from: NodeType;
  to: NodeType;
  label: string;
  count: number;
}

export interface ConceptAttribute {
  concept: NodeType;
  id: string;
  label: string;
  /** Share of instances with a value, 0 to 1. */
  filled: number;
}

export interface GraphMeta {
  source: string;
  synthetic: boolean;
  /** "declared": read off the source columns. "induced": proposed by the engine and approved. */
  schema_origin: string;
  records: number;
  organisations: number;
  organisations_in_sample: number;
  multi_record_organisations: number;
}

export interface CountedValue {
  value: string;
  records: number;
}

export interface GraphEntity {
  id: string;
  label: string;
  records: SourceRecord[];
  values: Record<string, CountedValue[]>;
  city: string | null;
  country: string | null;
  industry: string | null;
  systems: string[];
  possibly_same: { entity: string; pairs: number; probability: number; items: string[] }[];
}

export interface GraphReferenceNode {
  id: string;
  type: Exclude<NodeType, "organisation">;
  label: string;
  code?: string;
  organisations?: number;
  records?: number;
  in_sample: number;
  country?: string | null;
  cities?: string[];
}

export interface GraphExport {
  schema: string;
  meta: GraphMeta;
  concepts: Concept[];
  relations: Relation[];
  attributes: ConceptAttribute[];
  nodes: GraphReferenceNode[];
  entities: GraphEntity[];
}

export interface GraphSchema {
  meta: GraphMeta;
  concepts: Concept[];
  relations: Relation[];
  attributes: ConceptAttribute[];
}

/** A node as a neighbour or a list row: enough to draw and to link to. */
export interface NodeRef {
  id: string;
  type: NodeType;
  label: string;
  /** What makes it big: records for an organisation or a system, organisations for the rest. */
  size: number;
}

export interface NeighbourGroup {
  /** Stable key, e.g. "located_in" or "possibly_same". */
  relation: string;
  /** How the focus relates to these nodes, read as a sentence: "located in", "has organisations". */
  label: string;
  type: NodeType;
  /** "asserted": the graph says so. "open": the review band has not decided. */
  status: "asserted" | "open";
  nodes: (NodeRef & { items?: string[]; probability?: number; pairs?: number })[];
  /** How many there are in all; `nodes` may hold fewer. */
  total: number;
}

export interface NodeDetail {
  node: NodeRef;
  groups: NeighbourGroup[];
  /** Organisations only: the supporting records and what they say. */
  records?: SourceRecord[];
  values?: Record<string, CountedValue[]>;
  /** Reference nodes only: how many organisations in all, and how many this source can list. */
  total?: number;
  listed?: number;
}
