// What a decision about a whole group means for its pairs. Pure: given the group, the pairs and
// what is already decided, these say which decisions to append. Nothing here writes anything.
import type { Decision, DecisionInput, GroupEdge, ReviewGroup, ReviewItem } from "./types";

/**
 * Whether a link still ties two records together: the machine accepted it, or it is in review
 * and no reviewer has called it different.
 */
export function holds(edge: GroupEdge, decisions: ReadonlyMap<string, Decision>): boolean {
  if (edge.band === "reject") return false;
  if (edge.band === "accept") return true;
  return decisions.get(`${edge.left}~${edge.right}`)?.verdict !== "different";
}

/** Every record that can be reached from `from` over the given links. */
export function reachable(records: readonly string[], links: readonly { left: string; right: string }[], from: string): Set<string> {
  const next = new Map<string, string[]>(records.map((r) => [r, []]));
  for (const e of links) {
    next.get(e.left)?.push(e.right);
    next.get(e.right)?.push(e.left);
  }
  const seen = new Set([from]);
  const todo = [from];
  while (todo.length > 0) {
    for (const n of next.get(todo.pop() as string) ?? []) {
      if (!seen.has(n)) {
        seen.add(n);
        todo.push(n);
      }
    }
  }
  return seen;
}

/**
 * A starting point for a split: records share a part when the machine accepted a link between
 * them or a reviewer called them the same. Parts keep the group's record order.
 */
export function suggestParts(group: ReviewGroup, decisions: ReadonlyMap<string, Decision>): string[][] {
  const firm = group.edges.filter((e) => e.band === "accept" || (e.band === "review" && decisions.get(`${e.left}~${e.right}`)?.verdict === "same"));
  const parts: string[][] = [];
  const placed = new Set<string>();
  for (const record of group.records) {
    if (placed.has(record)) continue;
    const together = reachable(group.records, firm, record);
    const part = group.records.filter((r) => together.has(r));
    for (const r of part) placed.add(r);
    parts.push(part);
  }
  return parts;
}

export interface GroupPlan {
  /** One decision per open pair of the group. */
  decisions: DecisionInput[];
  /** Pairs that already carry a decision, which a group action never replaces. */
  kept: number;
  /** Of those, the ones whose decision disagrees with what the group action says. */
  disagree: string[];
  /** Split only: machine-accepted pairs the split puts in different parts. */
  separatesAccepted: [string, string][];
}

function itemsOf(group: ReviewGroup, items: ReadonlyMap<string, ReviewItem>): ReviewItem[] {
  return group.items.map((id) => items.get(id)).filter((i): i is ReviewItem => i !== undefined);
}

/** "All one": every open pair of the group is the same. */
export function planMerge(group: ReviewGroup, items: ReadonlyMap<string, ReviewItem>, decisions: ReadonlyMap<string, Decision>): GroupPlan {
  const plan: GroupPlan = { decisions: [], kept: 0, disagree: [], separatesAccepted: [] };
  for (const item of itemsOf(group, items)) {
    const existing = decisions.get(item.id);
    if (existing === undefined) plan.decisions.push({ item: item.id, verdict: "same" });
    else {
      plan.kept += 1;
      if (existing.verdict !== "same") plan.disagree.push(item.id);
    }
  }
  return plan;
}

/** Throws unless every record of the group is in exactly one part. */
export function checkParts(group: ReviewGroup, parts: readonly (readonly string[])[]): Map<string, number> {
  const partOf = new Map<string, number>();
  parts.forEach((part, index) => {
    for (const record of part) {
      if (partOf.has(record)) throw new Error(`record ${record} is in two parts`);
      partOf.set(record, index);
    }
  });
  const known = new Set(group.records);
  for (const record of partOf.keys()) if (!known.has(record)) throw new Error(`record ${record} is not in group ${group.id}`);
  for (const record of group.records) if (!partOf.has(record)) throw new Error(`record ${record} is in no part`);
  return partOf;
}

/** A split: open pairs inside one part are the same, open pairs across parts are different. */
export function planSplit(
  group: ReviewGroup,
  items: ReadonlyMap<string, ReviewItem>,
  decisions: ReadonlyMap<string, Decision>,
  parts: readonly (readonly string[])[],
): GroupPlan {
  const partOf = checkParts(group, parts);
  const together = (a: string, b: string) => partOf.get(a) === partOf.get(b);
  const plan: GroupPlan = { decisions: [], kept: 0, disagree: [], separatesAccepted: [] };
  for (const item of itemsOf(group, items)) {
    const same = together(item.left.unique_id, item.right.unique_id);
    const existing = decisions.get(item.id);
    if (existing === undefined) {
      plan.decisions.push(same ? { item: item.id, verdict: "same" } : { item: item.id, verdict: "different", reason: "group_split" });
    } else {
      plan.kept += 1;
      if (existing.verdict !== (same ? "same" : "different")) plan.disagree.push(item.id);
    }
  }
  for (const e of group.edges) {
    if (e.band === "accept" && !together(e.left, e.right)) plan.separatesAccepted.push([e.left, e.right]);
  }
  return plan;
}

export function count(plan: GroupPlan): { same: number; different: number } {
  const same = plan.decisions.filter((d) => d.verdict === "same").length;
  return { same, different: plan.decisions.length - same };
}
