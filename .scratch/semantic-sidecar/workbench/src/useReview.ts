// The queue and the decision log, loaded once and kept in step with what the reviewer records.
import { useCallback, useEffect, useState } from "react";

import type { Queue, ReviewDataSource } from "./dataSource";
import type { Decision, DecisionInput, GroupInput, GroupIntent, Ledger } from "./types";

export interface ReviewState {
  queue: Queue | null;
  ledger: Ledger;
  failed: boolean;
  decide(input: DecisionInput): Promise<Decision>;
  decideGroup(input: GroupInput): Promise<{ intent: GroupIntent; decisions: Decision[] }>;
  withdrawBatch(batch: string): Promise<Decision[]>;
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

  const decideGroup = useCallback(async (input: GroupInput) => {
    const done = await source.decideGroup(input);
    setLedger((l) => ({ decisions: [...l.decisions, ...done.decisions], intents: [...l.intents, done.intent] }));
    return done;
  }, [source]);

  const withdrawBatch = useCallback(async (batch: string) => {
    const withdrawals = await source.withdrawBatch(batch);
    setLedger((l) => ({ ...l, decisions: [...l.decisions, ...withdrawals] }));
    return withdrawals;
  }, [source]);

  return { queue, ledger, failed, decide, decideGroup, withdrawBatch };
}
