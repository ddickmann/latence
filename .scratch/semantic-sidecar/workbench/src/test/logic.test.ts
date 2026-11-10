import { describe, expect, it } from "vitest";

import { StaticGraphSource, StaticReviewSource, memoryStore } from "../dataSource";
import { bridge, explain, lean, markShared } from "../explain";
import { layers, layoutFocus, layoutSchema } from "../graphLayout";
import { translator } from "../i18n";
import { actionFor } from "../keys";
import { historyOf, inForce, pace, tally } from "../ledger";
import { DEFAULT_FILTER, hasConflict, nextOpen, order, step, visibleGroups } from "../queue";
import type { Decision, ReviewGroup } from "../types";
import { graphExport, reviewExport } from "./fixtures";

const actor = { id: "t", name: "Tester" };
const at = (minute: number) => new Date(Date.UTC(2026, 0, 1, 9, minute)).toISOString();
const decision = (item: string, verdict: Decision["verdict"], minute = 0): Decision => ({ item, verdict, actor, at: at(minute) });
const items = new Map(reviewExport.items.map((i) => [i.id, i] as const));

describe("the decision log", () => {
  it("keeps every entry and lets the latest one be in force", () => {
    const log = [decision("r1~r2", "same", 0), decision("r4~r5", "unsure", 1), decision("r1~r2", "different", 2)];
    expect(inForce(log).get("r1~r2")?.verdict).toBe("different");
    expect(historyOf(log, "r1~r2").map((d) => d.verdict)).toEqual(["same", "different"]);
    expect(tally(log)).toEqual({ same: 0, different: 1, unsure: 1 });
  });

  it("says nothing about pace until there is enough to go on", () => {
    const now = new Date(Date.UTC(2026, 0, 1, 9, 30));
    expect(pace([decision("a", "same", 0), decision("b", "same", 10)], now)).toBeNull();
    // 3 decisions over the 30 minutes since the first: 6 an hour.
    expect(pace([decision("a", "same", 0), decision("b", "same", 10), decision("c", "same", 20)], now)).toBe(6);
    // Decisions older than an hour do not count.
    expect(pace([decision("a", "same", 0), decision("b", "same", 1), decision("c", "same", 2)], new Date(Date.UTC(2026, 0, 1, 11, 0)))).toBeNull();
  });
});

describe("the queue", () => {
  it("lists groups by rank and drops decided pairs from the open view, except the one that is open", () => {
    const decided = inForce([decision("r1~r2", "same")]);
    const open = visibleGroups(reviewExport.groups, items, decided, DEFAULT_FILTER);
    expect(order(open)).toEqual(["r2~r3", "r4~r5"]);
    expect(open[0]?.decided).toBe(1);
    expect(order(visibleGroups(reviewExport.groups, items, decided, DEFAULT_FILTER, "r1~r2"))).toEqual(["r1~r2", "r2~r3", "r4~r5"]);
    expect(order(visibleGroups(reviewExport.groups, items, decided, { ...DEFAULT_FILTER, status: "decided" }))).toEqual(["r1~r2"]);
  });

  it("filters by source system and by conflict", () => {
    const none = new Map<string, Decision>();
    expect(order(visibleGroups(reviewExport.groups, items, none, { ...DEFAULT_FILTER, system: "procurement" }))).toEqual(["r4~r5"]);
    expect(order(visibleGroups(reviewExport.groups, items, none, { ...DEFAULT_FILTER, conflictsOnly: true }))).toEqual(["r1~r2", "r2~r3"]);
    expect(hasConflict(reviewExport.items[2]!)).toBe(false);
  });

  it("steps without wrapping and finds the next undecided pair with one wrap", () => {
    const ids = ["a", "b", "c"];
    expect(step(ids, undefined, 1)).toBe("a");
    expect(step(ids, "c", 1)).toBe("c");
    expect(step(ids, "b", -1)).toBe("a");
    const decided = inForce([decision("b", "same"), decision("c", "same")]);
    expect(nextOpen(ids, "b", decided)).toBe("a");
    expect(nextOpen(ids, "a", inForce([decision("a", "same"), decision("b", "same"), decision("c", "same")]))).toBeUndefined();
  });
});

describe("the keys", () => {
  it("gives each action one key and leaves typing and shortcuts alone", () => {
    expect(actionFor({ key: "a" })).toBe("same");
    expect(actionFor({ key: "R" })).toBe("different");
    expect(actionFor({ key: "u" })).toBe("unsure");
    expect(actionFor({ key: " " })).toBe("skip");
    expect(actionFor({ key: "j" })).toBe("next");
    expect(actionFor({ key: "k" })).toBe("previous");
    expect(actionFor({ key: "Escape", typing: true })).toBe("close");
    expect(actionFor({ key: "a", typing: true })).toBeNull();
    expect(actionFor({ key: "r", metaKey: true })).toBeNull();
    expect(actionFor({ key: "x" })).toBeNull();
  });
});

describe("the built-in explanation", () => {
  it("reads each comparison as for, against, missing or neutral", () => {
    const [name, , industry] = reviewExport.items[0]!.evidence;
    expect(lean(name!)).toBe("for");
    expect(lean(industry!)).toBe("against");
    expect(explain(industry!).key).toBe("explain.differ");
    expect(explain(name!).key).toBe("explain.same");
    const missing = reviewExport.items[2]!.evidence[1]!;
    expect(lean(missing)).toBe("missing");
    expect(explain(missing).key).toBe("explain.missingBoth");
    expect(lean({ ...name!, weight: 0.2 })).toBe("neutral");
  });

  it("marks the words two names do not share", () => {
    expect(markShared("Acme Capital Inc.", "ACME consulting inc")).toEqual([
      { word: "Acme", shared: true }, { word: "Capital", shared: false }, { word: "Inc.", shared: true },
    ]);
  });

  it("finds the pair that alone holds a group together", () => {
    const group = reviewExport.groups[0] as ReviewGroup;
    const none = new Map<string, Decision>();
    // r2 hangs on by two review links; neither is the only one.
    expect(bridge(group, { left: "r1", right: "r2" }, none)).toBeNull();
    // Once r2~r3 is called different, r1~r2 is all that ties r2 in.
    const decided = inForce([decision("r2~r3", "different")]);
    expect(bridge(group, { left: "r1", right: "r2" }, decided)).toEqual([["r1", "r3"], ["r2"]]);
  });
});

describe("the static review source", () => {
  it("appends decisions with actor and time and never rewrites one", async () => {
    const store = memoryStore();
    let minute = 0;
    const source = new StaticReviewSource(reviewExport, { actor, store, now: () => new Date(Date.UTC(2026, 0, 1, 9, minute++)) });
    await source.decide({ item: "r1~r2", verdict: "same" });
    await source.decide({ item: "r1~r2", verdict: "different", reason: "business", note: "two firms" });
    const ledger = await source.loadLedger();
    expect(ledger.decisions).toEqual([
      { item: "r1~r2", verdict: "same", actor, at: at(0) },
      { item: "r1~r2", verdict: "different", reason: "business", note: "two firms", actor, at: at(1) },
    ]);
    // A second source over the same store sees the same log: it survives a reload.
    expect((await new StaticReviewSource(reviewExport, { actor, store }).loadLedger()).decisions).toHaveLength(2);
    await expect(source.decide({ item: "nope", verdict: "same" })).rejects.toThrow("unknown item");
  });

  it("ignores a stored log that is damaged or belongs to another export", async () => {
    const store = memoryStore();
    store.setItem("rw.ledger.v1", "{not json");
    expect(await new StaticReviewSource(reviewExport, { actor, store }).loadLedger()).toEqual({ decisions: [], intents: [] });
    store.setItem("rw.ledger.v1", JSON.stringify({ decisions: [decision("gone~pair", "same"), decision("r4~r5", "same")], intents: [] }));
    expect((await new StaticReviewSource(reviewExport, { actor, store }).loadLedger()).decisions.map((d) => d.item)).toEqual(["r4~r5"]);
  });
});

describe("the static graph source", () => {
  const graph = new StaticGraphSource(graphExport);

  it("gives an organisation its neighbours by relation, the undecided ones marked open", async () => {
    const detail = await graph.loadNode("org:r1");
    expect(detail?.groups.map((g) => [g.relation, g.status, g.nodes.map((n) => n.id)])).toEqual([
      ["located_in", "asserted", ["city:Leeds"]],
      ["operates_in", "asserted", ["industry:CAP"]],
      ["recorded_in", "asserted", ["system:billing", "system:erp"]],
      ["possibly_same", "open", ["org:r2"]],
    ]);
    expect(detail?.groups[3]?.nodes[0]?.items).toEqual(["r1~r2", "r2~r3"]);
    expect(detail?.records).toHaveLength(2);
  });

  it("walks from a city up to its country and down to its organisations", async () => {
    const leeds = await graph.loadNode("city:Leeds");
    expect(leeds?.groups.map((g) => [g.relation, g.nodes.map((n) => n.id)])).toEqual([
      ["in_country", ["country:GB"]],
      ["has_organisation", ["org:r1", "org:r2"]],
    ]);
    expect((await graph.loadNode("country:GB"))?.groups[0]?.nodes.map((n) => n.label)).toEqual(["Leeds", "Bristol"]);
    expect((await graph.loadNode("industry:LAB"))?.groups[0]?.nodes.map((n) => n.id)).toEqual(["org:r4"]);
    expect(await graph.loadNode("city:Nowhere")).toBeUndefined();
  });

  it("searches names, finds a record's organisation and lists a concept's instances", async () => {
    expect((await graph.search("acme", 5)).map((n) => n.id)).toEqual(["org:r1", "org:r2"]);
    expect((await graph.search("  ", 5))).toEqual([]);
    expect((await graph.entityOfRecord("r5"))?.id).toBe("org:r4");
    expect(await graph.entityOfRecord("r99")).toBeUndefined();
    expect(await graph.listInstances("city", 1)).toEqual({ nodes: [{ id: "city:Leeds", type: "city", label: "Leeds", size: 2 }], total: 2 });
  });
});

describe("the drawings", () => {
  it("puts concepts in layers by the longest chain of relations", () => {
    const l = layers(graphExport.concepts, graphExport.relations);
    expect([...l]).toEqual([["organisation", 0], ["city", 1], ["country", 2], ["industry", 1], ["source_system", 1]]);
  });

  it("keeps every schema box inside the drawing and apart from its neighbours, wide and narrow", () => {
    for (const width of [326, 720, 1000]) {
      const layout = layoutSchema(graphExport.concepts, graphExport.relations, width);
      const boxes = [...layout.nodes.values()].map((p) => ({ l: p.x - layout.box.w / 2, r: p.x + layout.box.w / 2, t: p.y - layout.box.h / 2, b: p.y + layout.box.h / 2 }));
      for (const b of boxes) {
        expect(b.l).toBeGreaterThanOrEqual(-0.5);
        expect(b.r).toBeLessThanOrEqual(layout.width + 0.5);
        expect(b.t).toBeGreaterThanOrEqual(-0.5);
        expect(b.b).toBeLessThanOrEqual(layout.height + 0.5);
      }
      for (const a of boxes) for (const b of boxes) if (a !== b) expect(a.r <= b.l || b.r <= a.l || a.b <= b.t || b.b <= a.t).toBe(true);
      expect(layout.edges).toHaveLength(graphExport.relations.length);
    }
  });

  it("draws at most its budget of neighbours, at least one per relation, and counts the rest", () => {
    const many = Array.from({ length: 30 }, (_, i) => ({ id: `org:${i}`, type: "organisation" as const, label: `Org ${i}`, size: 1 }));
    const groups = [
      { relation: "in_country", label: "lies in", type: "country" as const, status: "asserted" as const, nodes: [{ id: "country:GB", type: "country" as const, label: "GB", size: 3 }], total: 1 },
      { relation: "has_organisation", label: "is the location of", type: "organisation" as const, status: "asserted" as const, nodes: many, total: 338 },
    ];
    const narrow = layoutFocus(groups, 326);
    expect(narrow.narrow).toBe(true);
    expect(narrow.spokes.map((s) => s.group.relation)).toEqual(["in_country", "has_organisation", "has_organisation", "has_organisation"]);
    expect(narrow.hubs.map((h) => h.hidden)).toEqual([0, 335]);
    const wide = layoutFocus(groups, 1000);
    expect(wide.spokes).toHaveLength(8);
    for (const s of [...narrow.spokes, ...wide.spokes]) {
      const layout = narrow.spokes.includes(s) ? narrow : wide;
      expect(s.at.x).toBeGreaterThan(40);
      expect(s.at.x).toBeLessThan(layout.width - 40);
      expect(s.at.y).toBeGreaterThan(30);
      expect(s.at.y).toBeLessThan(layout.height - 30);
    }
  });
});

describe("messages", () => {
  it("fills variables and picks the plural the way i18next does", () => {
    const t = translator();
    expect(t("queue.groupMeta", { count: 1 })).toBe("1 record");
    expect(t("queue.groupMeta", { count: 4 })).toBe("4 records");
    expect(t("review.of", { n: 2, total: 9 })).toBe("2 of 9");
    expect(t("no.such.key")).toBe("no.such.key");
  });
});
