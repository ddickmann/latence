// The decision log is append-only: a changed mind is a new entry, never an edit. These read it.
import type { Decision, GroupIntent, Verdict } from "./types";

/** The decision in force for each item: the latest entry, unless that entry withdraws. */
export function inForce(decisions: readonly Decision[]): Map<string, Decision> {
  const out = new Map<string, Decision>();
  for (const d of decisions) {
    if (d.withdraws === true) out.delete(d.item);
    else out.set(d.item, d);
  }
  return out;
}

/** The pairs a group action decided whose decision from that action is still the one in force. */
export function standing(decisions: readonly Decision[], batch: string): Decision[] {
  return [...inForce(decisions).values()].filter((d) => d.batch === batch);
}

export function historyOf(decisions: readonly Decision[], item: string): Decision[] {
  return decisions.filter((d) => d.item === item);
}

export function latestIntent(intents: readonly GroupIntent[], group: string): GroupIntent | undefined {
  let found: GroupIntent | undefined;
  for (const i of intents) if (i.group === group) found = i;
  return found;
}

export function tally(decisions: readonly Decision[]): Record<Verdict, number> {
  const out: Record<Verdict, number> = { same: 0, different: 0, unsure: 0 };
  for (const d of inForce(decisions).values()) out[d.verdict] += 1;
  return out;
}

/**
 * How often the reviewer's decisions in force went with the suggestion: of the pairs that had
 * a suggestion and were called same or different, how many got the suggested verdict. "Not
 * sure" neither agrees nor disagrees.
 */
export function agreement(decisions: readonly Decision[]): { agreed: number; of: number } {
  let agreed = 0;
  let of = 0;
  for (const d of inForce(decisions).values()) {
    if (d.suggested === undefined || d.verdict === "unsure") continue;
    of += 1;
    if (d.verdict === d.suggested) agreed += 1;
  }
  return { agreed, of };
}

/**
 * Decisions per hour over the last hour of work, or null while there is too little to say:
 * fewer than three decisions, or less than a minute between the first and now.
 */
export function pace(decisions: readonly Decision[], now: Date): number | null {
  const hourAgo = now.getTime() - 3_600_000;
  const recent = decisions.filter((d) => d.withdraws !== true).map((d) => Date.parse(d.at)).filter((t) => t >= hourAgo && t <= now.getTime());
  if (recent.length < 3) return null;
  const span = now.getTime() - Math.min(...recent);
  if (span < 60_000) return null;
  return Math.round((recent.length / span) * 3_600_000);
}
