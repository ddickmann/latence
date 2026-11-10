// What the queue shows and in what order. Pure, so the keyboard flow can be tested without a DOM.
import type { Decision, ReviewGroup, ReviewItem } from "./types";

export interface QueueFilter {
  status: "open" | "decided" | "all";
  /** A source system, or "all". An item passes when either record comes from it. */
  system: string;
  /** Only pairs where at least one field speaks against a match. */
  conflictsOnly: boolean;
}

export const DEFAULT_FILTER: QueueFilter = { status: "open", system: "all", conflictsOnly: false };

/** Evidence weaker than this, either way, is treated as saying nothing. */
export const WEIGHT_FLOOR = 0.5;

export function hasConflict(item: ReviewItem): boolean {
  return item.evidence.some((e) => e.weight < -WEIGHT_FLOOR);
}

export function passes(item: ReviewItem, decided: boolean, filter: QueueFilter): boolean {
  if (filter.status === "open" && decided) return false;
  if (filter.status === "decided" && !decided) return false;
  if (filter.system !== "all" && item.left["source_system"] !== filter.system && item.right["source_system"] !== filter.system) return false;
  if (filter.conflictsOnly && !hasConflict(item)) return false;
  return true;
}

export interface QueueGroup {
  group: ReviewGroup;
  items: ReviewItem[];
  decided: number;
}

/** Groups in rank order, each with the items the filter lets through. Empty groups drop out. */
export function visibleGroups(
  groups: readonly ReviewGroup[],
  items: ReadonlyMap<string, ReviewItem>,
  decisions: ReadonlyMap<string, Decision>,
  filter: QueueFilter,
  /** An item kept visible whatever the filter says: the one that is open. */
  pinned?: string,
): QueueGroup[] {
  const out: QueueGroup[] = [];
  for (const group of [...groups].sort((a, b) => a.rank - b.rank)) {
    const all = group.items.map((id) => items.get(id)).filter((i): i is ReviewItem => i !== undefined);
    const shown = all.filter((i) => i.id === pinned || passes(i, decisions.has(i.id), filter));
    if (shown.length > 0) out.push({ group, items: shown, decided: all.filter((i) => decisions.has(i.id)).length });
  }
  return out;
}

export function order(groups: readonly QueueGroup[]): string[] {
  return groups.flatMap((g) => g.items.map((i) => i.id));
}

/** The neighbour in the list, without wrapping; the first item when nothing is open yet. */
export function step(ids: readonly string[], current: string | undefined, by: 1 | -1): string | undefined {
  if (ids.length === 0) return undefined;
  if (current === undefined) return ids[0];
  const at = ids.indexOf(current);
  if (at === -1) return ids[0];
  return ids[Math.min(ids.length - 1, Math.max(0, at + by))];
}

/** The next undecided item after the current one, wrapping once; undefined when none is left. */
export function nextOpen(ids: readonly string[], current: string | undefined, decisions: ReadonlyMap<string, Decision>): string | undefined {
  const start = current === undefined ? -1 : ids.indexOf(current);
  for (let n = 1; n <= ids.length; n += 1) {
    const id = ids[(start + n + ids.length) % ids.length];
    if (id !== undefined && id !== current && !decisions.has(id)) return id;
  }
  return undefined;
}

export function systemsOf(items: Iterable<ReviewItem>): string[] {
  const seen = new Set<string>();
  for (const i of items) {
    for (const r of [i.left, i.right]) {
      const s = r["source_system"];
      if (typeof s === "string" && s !== "") seen.add(s);
    }
  }
  return [...seen].sort();
}
