// The queue and the decision log, loaded once and kept in step with what the reviewer records.
import { useCallback, useEffect, useState } from "react";

import type { Queue, ReviewDataSource } from "./dataSource";
import type { Decision, DecisionInput, GroupIntent, Ledger } from "./types";

export interface ReviewState {
  queue: Queue | null;
  ledger: Ledger;
  failed: boolean;
  decide(input: DecisionInput): Promise<Decision>;
  propose(input: { group: string; intent: "merge" | "split"; note?: string }): Promise<GroupIntent>;
}

export function useReview(source: ReviewDataSource): ReviewState {
  const [queue, setQueue] = useState<Queue | null>(null);
  const [ledger, setLedger] = useState<Ledger>({ decisions: [], intents: [] });
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let live = true;
    setFailed(false);
    Promise.all([source.loadQueue(), source.loadLedger()]).then(
      ([q, l]) => {
        if (!live) return;
        setQueue(q);
        setLedger(l);
      },
      () => live && setFailed(true),
    );
    return () => {
      live = false;
    };
  }, [source]);

  const decide = useCallback(async (input: DecisionInput) => {
    const decision = await source.decide(input);
    setLedger((l) => ({ ...l, decisions: [...l.decisions, decision] }));
    return decision;
  }, [source]);

  const propose = useCallback(async (input: { group: string; intent: "merge" | "split"; note?: string }) => {
    const intent = await source.proposeGroup(input);
    setLedger((l) => ({ ...l, intents: [...l.intents, intent] }));
    return intent;
  }, [source]);

  return { queue, ledger, failed, decide, propose };
}
