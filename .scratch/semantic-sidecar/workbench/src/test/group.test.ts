import { describe, expect, it } from "vitest";

import { StaticReviewSource, memoryStore } from "../dataSource";
import { count, planMerge, planSplit, suggestParts } from "../groupActions";
import { historyOf, inForce, standing } from "../ledger";
import type { Decision, ReviewGroup } from "../types";
import { reviewExport } from "./fixtures";

const actor = { id: "t", name: "Tester" };
const items = new Map(reviewExport.items.map((i) => [i.id, i] as const));
// r1, r2, r3: the machine accepted r1~r3; r1~r2 and r2~r3 are in review.
const group = reviewExport.groups[0] as ReviewGroup;
const none = new Map<string, Decision>();
const decided = (item: string, verdict: Decision["verdict"]): Map<string, Decision> =>
  new Map([[item, { item, verdict, actor, at: "2026-01-01T09:00:00.000Z" }]]);

describe("a decision about a whole group", () => {
  it("suggests parts from what is already firm", () => {
    expect(suggestParts(group, none)).toEqual([["r1", "r3"], ["r2"]]);
    // A pair the reviewer called the same joins its two parts.
    expect(suggestParts(group, decided("r2~r3", "same"))).toEqual([["r1", "r2", "r3"]]);
    // A pair called different does not.
    expect(suggestParts(group, decided("r2~r3", "different"))).toEqual([["r1", "r3"], ["r2"]]);
  });

  it("merge decides every open pair as same and leaves decided pairs alone", () => {
    expect(planMerge(group, items, none)).toEqual({
      decisions: [{ item: "r1~r2", verdict: "same" }, { item: "r2~r3", verdict: "same" }],
      kept: 0, disagree: [], separatesAccepted: [],
    });
    const plan = planMerge(group, items, decided("r1~r2", "different"));
    expect(plan.decisions).toEqual([{ item: "r2~r3", verdict: "same" }]);
    expect(plan.kept).toBe(1);
    expect(plan.disagree).toEqual(["r1~r2"]);
  });

  it("split calls pairs inside a part the same and pairs across parts different", () => {
    const plan = planSplit(group, items, none, [["r1"], ["r2", "r3"]]);
    expect(plan.decisions).toEqual([
      { item: "r1~r2", verdict: "different", reason: "group_split" },
      { item: "r2~r3", verdict: "same" },
    ]);
    expect(count(plan)).toEqual({ same: 1, different: 1 });
    // r1~r3 was accepted by the machine and is not a review item: it is reported, not decided.
    expect(plan.separatesAccepted).toEqual([["r1", "r3"]]);
    expect(planSplit(group, items, none, [["r1", "r3"], ["r2"]]).separatesAccepted).toEqual([]);
  });

  it("split keeps a decided pair and says when it disagrees", () => {
    const plan = planSplit(group, items, decided("r1~r2", "same"), [["r1", "r3"], ["r2"]]);
    expect(plan.decisions).toEqual([{ item: "r2~r3", verdict: "different", reason: "group_split" }]);
    expect(plan.kept).toBe(1);
    expect(plan.disagree).toEqual(["r1~r2"]);
  });

  it("refuses parts that leave a record out, repeat one or name a stranger", () => {
    expect(() => planSplit(group, items, none, [["r1"], ["r2"]])).toThrow("r3 is in no part");
    expect(() => planSplit(group, items, none, [["r1", "r2"], ["r2", "r3"]])).toThrow("r2 is in two parts");
    expect(() => planSplit(group, items, none, [["r1", "r2", "r3"], ["r9"]])).toThrow("r9 is not in group");
  });
});

describe("withdrawing", () => {
  it("reopens a pair and keeps both entries in the log", () => {
    const log: Decision[] = [
      { item: "a", verdict: "same", actor, at: "t1", batch: "b1" },
      { item: "b", verdict: "same", actor, at: "t1", batch: "b1" },
      { item: "a", verdict: "same", actor, at: "t2", batch: "b1", withdraws: true },
    ];
    expect([...inForce(log).keys()]).toEqual(["b"]);
    expect(historyOf(log, "a")).toHaveLength(2);
    expect(standing(log, "b1").map((d) => d.item)).toEqual(["b"]);
  });
});

describe("the static source applies a group decision in one step", () => {
  const clock = () => {
    let minute = 0;
    return () => new Date(Date.UTC(2026, 0, 1, 9, minute++));
  };

  it("records the pair decisions and the group record under one batch id", async () => {
    const source = new StaticReviewSource(reviewExport, { actor, store: memoryStore(), now: clock() });
    const { intent, decisions } = await source.decideGroup({ group: "r1", intent: "split", parts: [["r1"], ["r2", "r3"]], note: "two firms" });
    expect(decisions.map((d) => [d.item, d.verdict, d.reason, d.batch])).toEqual([
      ["r1~r2", "different", "group_split", intent.batch],
      ["r2~r3", "same", undefined, intent.batch],
    ]);
    expect(intent).toEqual({
      group: "r1", intent: "split", actor, at: "2026-01-01T09:00:00.000Z", batch: "r1@2026-01-01T09:00:00.000Z",
      decided: 2, note: "two firms", parts: [["r1"], ["r2", "r3"]], separates_accepted: [["r1", "r3"]],
    });
    const ledger = await source.loadLedger();
    expect(ledger.decisions).toEqual(decisions);
    expect(ledger.intents).toEqual([intent]);
  });

  it("undo withdraws what still stands and leaves a pair decided again since", async () => {
    const source = new StaticReviewSource(reviewExport, { actor, store: memoryStore(), now: clock() });
    const { intent } = await source.decideGroup({ group: "r1", intent: "merge" });
    await source.decide({ item: "r1~r2", verdict: "different", reason: "business" });
    const withdrawn = await source.withdrawBatch(intent.batch);
    expect(withdrawn.map((d) => [d.item, d.withdraws])).toEqual([["r2~r3", true]]);
    const ledger = await source.loadLedger();
    expect([...inForce(ledger.decisions)].map(([item, d]) => [item, d.verdict])).toEqual([["r1~r2", "different"]]);
    // Nothing was removed: two from the merge, one individual decision, one withdrawal.
    expect(ledger.decisions).toHaveLength(4);
    expect(await source.withdrawBatch(intent.batch)).toEqual([]);
  });

  it("rejects an unknown group and malformed parts without writing", async () => {
    const source = new StaticReviewSource(reviewExport, { actor, store: memoryStore() });
    await expect(source.decideGroup({ group: "nope", intent: "merge" })).rejects.toThrow("unknown group");
    await expect(source.decideGroup({ group: "r1", intent: "split", parts: [["r1"]] })).rejects.toThrow("is in no part");
    expect(await source.loadLedger()).toEqual({ decisions: [], intents: [] });
  });
});
