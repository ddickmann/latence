// The built-in explanation of a review item: computed from the matcher's own evidence, with no
// model call, so it is the same every time and can be checked against the numbers beside it.
import { holds, reachable } from "./groupActions";
import { WEIGHT_FLOOR } from "./queue";
import type { Decision, EvidenceRow, ReviewGroup, Scalar } from "./types";

export type Lean = "for" | "against" | "missing" | "neutral";

export function lean(e: EvidenceRow): Lean {
  if (e.level === -1) return "missing";
  if (e.weight > WEIGHT_FLOOR) return "for";
  if (e.weight < -WEIGHT_FLOOR) return "against";
  return "neutral";
}

function same(a: Scalar, b: Scalar): boolean {
  return a !== null && b !== null && String(a).trim().toLowerCase() === String(b).trim().toLowerCase();
}

/** The message key and its variables for one evidence row. */
export function explain(e: EvidenceRow): { key: string; vars: Record<string, string> } {
  const vars = { left: e.left === null ? "" : String(e.left), right: e.right === null ? "" : String(e.right) };
  const l = lean(e);
  if (l === "missing") {
    if (e.left === null && e.right === null) return { key: "explain.missingBoth", vars };
    return { key: e.left === null ? "explain.missingLeft" : "explain.missingRight", vars };
  }
  if (same(e.left, e.right)) return { key: l === "against" ? "explain.sameButCommon" : "explain.same", vars };
  if (l === "for") return { key: "explain.close", vars };
  if (l === "against") return { key: "explain.differ", vars };
  return { key: "explain.neutral", vars };
}

/** Words of `text`, each marked as shared with `other` or not (case and punctuation ignored). */
export function markShared(text: string, other: string): { word: string; shared: boolean }[] {
  const norm = (w: string) => w.toLowerCase().replace(/[^\p{L}\p{N}]/gu, "");
  const theirs = new Set(other.split(/\s+/).map(norm).filter(Boolean));
  return text.split(/\s+/).filter(Boolean).map((word) => ({ word, shared: theirs.has(norm(word)) }));
}

/**
 * Whether this pair is the only link between two parts of its group. If rejecting it would
 * leave the group in two pieces, returns the two sides; otherwise null. A link counts while the
 * machine accepted it, or it is in review and no reviewer has called it different.
 */
export function bridge(
  group: ReviewGroup,
  pair: { left: string; right: string },
  decisions: ReadonlyMap<string, Decision>,
): [string[], string[]] | null {
  const links = group.edges.filter((e) => {
    if ((e.left === pair.left && e.right === pair.right) || (e.left === pair.right && e.right === pair.left)) return false;
    return holds(e, decisions);
  });
  const reach = (from: string) => reachable(group.records, links, from);
  const a = reach(pair.left);
  if (a.has(pair.right)) return null;
  return [[...a].sort(), [...reach(pair.right)].sort()];
}
