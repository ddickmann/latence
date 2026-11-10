// The decision log is append-only: a changed mind is a new entry, never an edit. These read it.
import type { Decision, GroupIntent, Verdict } from "./types";

/** The decision in force for each item: the latest entry. */
export function inForce(decisions: readonly Decision[]): Map<string, Decision> {
  const out = new Map<string, Decision>();
  for (const d of decisions) out.set(d.item, d);
  return out;
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
 * Decisions per hour over the last hour of work, or null while there is too little to say:
 * fewer than three decisions, or less than a minute between the first and now.
 */
export function pace(decisions: readonly Decision[], now: Date): number | null {
  const hourAgo = now.getTime() - 3_600_000;
  const recent = decisions.map((d) => Date.parse(d.at)).filter((t) => t >= hourAgo && t <= now.getTime());
  if (recent.length < 3) return null;
  const span = now.getTime() - Math.min(...recent);
  if (span < 60_000) return null;
  return Math.round((recent.length / span) * 3_600_000);
}
