// A small review export and graph export in the engine's shapes, for tests.
import type { EvidenceRow, GraphExport, ReviewExport, ReviewItem, SourceRecord } from "../types";

function record(id: string, system: string, name: string, city: string | null, industry: string | null): SourceRecord {
  return { unique_id: id, source_system: system, name, city, postcode: null, country: "GB", industry, reg_id: null };
}

function evidence(weights: Record<string, number | null>): EvidenceRow[] {
  return Object.entries(weights).map(([comparison, weight]) => ({
    comparison,
    label: comparison,
    field: comparison,
    level: weight === null ? -1 : weight > 0 ? 1 : 0,
    level_label: "",
    weight: weight ?? 0,
    left: weight === null ? null : "a",
    right: weight === null ? null : weight > 0 ? "a" : "b",
  }));
}

function item(left: SourceRecord, right: SourceRecord, group: string, probability: number, weights: Record<string, number | null>): ReviewItem {
  const ev = evidence(weights);
  const prior = -12;
  return {
    id: `${left.unique_id}~${right.unique_id}`, group, band: "review", probability,
    match_weight: prior + ev.reduce((s, e) => s + e.weight, 0), prior_weight: prior,
    left, right, evidence: ev, why_unsure: "", vetoed: false, gold_same: probability > 0.5,
  };
}

const DIFFERENT = {
  verdict: "different" as const, like_this: 44, like_this_agree: 41,
  conditions: [
    { comparison: "industry", label: "Industry", levels: [0], text: "differs" },
    { comparison: "name", label: "Name", levels: [0, 1], text: "not the same" },
  ],
};
const SAME = {
  verdict: "same" as const, like_this: 30, like_this_agree: 29,
  conditions: [{ comparison: "name", label: "Name", levels: [1], text: "agrees" }],
};

const r1 = record("r1", "erp", "Acme Capital Inc", "Leeds", "CAP");
const r2 = record("r2", "crm", "Acme Consulting Inc", "Leeds", "CON");
const r3 = record("r3", "billing", "ACME CAPITAL", "Leeds", "CAP");
const r4 = record("r4", "erp", "Birch Labs Ltd", "Bristol", "LAB");
const r5 = record("r5", "procurement", "birch labs", null, "LAB");

export const reviewExport: ReviewExport = {
  schema: "review-export/1",
  meta: {
    source: "test", kind: "org", synthetic: true, records: 5, candidate_pairs: 6,
    bands: { accept_at: 0.585, reject_below: 0.46 },
    counts: { accept: 1, review: 3, reject: 2 },
    review_groups_total: 2, review_groups_exported: 2,
    record_fields: ["unique_id", "source_system", "name", "city", "postcode", "country", "industry", "reg_id"],
    comparisons: [{ comparison: "name", label: "name", field: "name" }, { comparison: "city", label: "city", field: "city" }, { comparison: "industry", label: "industry", field: "industry" }],
    suggester: { available: true, labelled_pairs: 300, out_of_sample_right: 288, out_of_sample_share: 0.96, always_same_share: 0.7, labels: "synthetic" },
  },
  groups: [
    {
      id: "r1", rank: 0, priority: 2.4, records: ["r1", "r2", "r3"],
      labels: { r1: "Acme Capital Inc", r2: "Acme Consulting Inc", r3: "ACME CAPITAL" },
      edges: [
        { left: "r1", right: "r2", band: "review", probability: 0.55 },
        { left: "r2", right: "r3", band: "review", probability: 0.5 },
        { left: "r1", right: "r3", band: "accept", probability: 0.97 },
      ],
      items: ["r1~r2", "r2~r3"],
    },
    {
      id: "r4", rank: 1, priority: 1.1, records: ["r4", "r5"],
      labels: { r4: "Birch Labs Ltd", r5: "birch labs" },
      edges: [{ left: "r4", right: "r5", band: "review", probability: 0.57 }],
      items: ["r4~r5"],
    },
  ],
  items: [
    { ...item(r1, r2, "r1", 0.55, { name: 2, city: 4, industry: -1.4 }), suggestion: DIFFERENT },
    // No suggestion for this one: a source may have none to give.
    { ...item(r2, r3, "r1", 0.5, { name: 1.5, city: 4, industry: -1.4 }), suggestion: null },
    { ...item(r4, r5, "r4", 0.57, { name: 6, city: null, industry: 2 }), suggestion: SAME },
  ],
  audit: [{ ...item(r1, r3, "r1", 0.97, { name: 8, city: 4, industry: 2 }), band: "accept", group: null }],
};

export const graphExport: GraphExport = {
  schema: "graph-export/1",
  meta: { source: "test", synthetic: true, schema_origin: "declared", records: 5, organisations: 3, organisations_in_sample: 3, multi_record_organisations: 2 },
  concepts: [
    { id: "organisation", label: "Organisation", kind: "entity", count: 3, definition: "A legal entity." },
    { id: "city", label: "City", kind: "reference", count: 2, definition: "A place." },
    { id: "country", label: "Country", kind: "reference", count: 1, definition: "A country." },
    { id: "industry", label: "Industry", kind: "reference", count: 3, definition: "A line of business." },
    { id: "source_system", label: "Source system", kind: "provenance", count: 4, definition: "A system of record." },
  ],
  relations: [
    { id: "located_in", from: "organisation", to: "city", label: "located in", count: 3 },
    { id: "operates_in", from: "organisation", to: "industry", label: "operates in", count: 3 },
    { id: "recorded_in", from: "organisation", to: "source_system", label: "recorded in", count: 5 },
    { id: "in_country", from: "city", to: "country", label: "lies in", count: 2 },
  ],
  attributes: [{ concept: "organisation", id: "name", label: "Name", filled: 1 }, { concept: "organisation", id: "reg_id", label: "Registration id", filled: 0.5 }],
  nodes: [
    { id: "city:Leeds", type: "city", label: "Leeds", organisations: 2, in_sample: 2, country: "GB" },
    { id: "city:Bristol", type: "city", label: "Bristol", organisations: 1, in_sample: 1, country: "GB" },
    { id: "country:GB", type: "country", label: "GB", organisations: 3, in_sample: 3, cities: ["Bristol", "Leeds"] },
    { id: "industry:CAP", type: "industry", label: "Capital", code: "CAP", organisations: 1, in_sample: 1 },
    { id: "industry:CON", type: "industry", label: "Consulting", code: "CON", organisations: 1, in_sample: 1 },
    { id: "industry:LAB", type: "industry", label: "Labs", code: "LAB", organisations: 1, in_sample: 1 },
    { id: "system:erp", type: "source_system", label: "erp", records: 2, in_sample: 2 },
    { id: "system:crm", type: "source_system", label: "crm", records: 1, in_sample: 1 },
    { id: "system:billing", type: "source_system", label: "billing", records: 1, in_sample: 1 },
    { id: "system:procurement", type: "source_system", label: "procurement", records: 1, in_sample: 1 },
  ],
  entities: [
    {
      id: "org:r1", label: "Acme Capital Inc", records: [r1, r3],
      values: { name: [{ value: "ACME CAPITAL", records: 1 }, { value: "Acme Capital Inc", records: 1 }], city: [{ value: "Leeds", records: 2 }] },
      city: "Leeds", country: "GB", industry: "CAP", systems: ["billing", "erp"],
      possibly_same: [{ entity: "org:r2", pairs: 2, probability: 0.55, items: ["r1~r2", "r2~r3"] }],
    },
    {
      id: "org:r2", label: "Acme Consulting Inc", records: [r2],
      values: { name: [{ value: "Acme Consulting Inc", records: 1 }], city: [{ value: "Leeds", records: 1 }] },
      city: "Leeds", country: "GB", industry: "CON", systems: ["crm"],
      possibly_same: [{ entity: "org:r1", pairs: 2, probability: 0.55, items: ["r1~r2", "r2~r3"] }],
    },
    {
      id: "org:r4", label: "Birch Labs Ltd", records: [r4, r5],
      values: { name: [{ value: "Birch Labs Ltd", records: 1 }, { value: "birch labs", records: 1 }], city: [{ value: "Bristol", records: 1 }] },
      city: "Bristol", country: "GB", industry: "LAB", systems: ["erp", "procurement"],
      possibly_same: [],
    },
  ],
};
