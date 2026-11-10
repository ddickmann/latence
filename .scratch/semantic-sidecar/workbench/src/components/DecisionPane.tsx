// The decision: four actions, one key each; what is recorded for this pair; and what the
// reviewer says about the group as a whole.
import { clock } from "../format";
import { useHost } from "../host";
import { KEYS, type KeyAction } from "../keys";
import type { Decision, GroupIntent, ReviewGroup, ReviewItem } from "../types";
import { VerdictMark } from "./QueuePane";

/** The four actions. A direct child of the detail view, so it can stay in reach on a phone. */
export function DecisionBar({ decision, onAct }: { decision: Decision | undefined; onAct: (action: KeyAction) => void }) {
  const { t, controls: { Button } } = useHost();
  return (
    <div className="rw-decide" role="group" aria-label={t("decide.title")}>
      <Button kind="primary" kbd={KEYS.same} onClick={() => onAct("same")} aria-pressed={decision?.verdict === "same"} testId="decide-same">{t("decide.same")}</Button>
      <Button kind="danger" kbd={KEYS.different} onClick={() => onAct("different")} aria-pressed={decision?.verdict === "different"} testId="decide-different">{t("decide.different")}</Button>
      <Button kind="secondary" kbd={KEYS.unsure} onClick={() => onAct("unsure")} aria-pressed={decision?.verdict === "unsure"} testId="decide-unsure">{t("decide.unsure")}</Button>
      <Button kind="quiet" kbd={KEYS.skip} onClick={() => onAct("skip")} testId="decide-skip">{t("decide.skip")}</Button>
    </div>
  );
}

export function DecisionPane(props: {
  item: ReviewItem;
  group: ReviewGroup | undefined;
  history: Decision[];
  intent: GroupIntent | undefined;
  onIntent: (intent: "merge" | "split") => void;
}) {
  const { t, locale, controls: { Button } } = useHost();
  const inForce = props.history[props.history.length - 1];
  return (
    <aside className="rw-aside" aria-label={t("decide.recordTitle")}>
      <section className="rw-section">
        <h3>{t("decide.recordTitle")}</h3>
        {inForce === undefined ? (
          <p className="rw-muted" data-testid="no-decision">{t("decide.none")}</p>
        ) : (
          <ol className="rw-history" data-testid="history">
            {[...props.history].reverse().map((d, i) => (
              <li key={d.at + String(i)} className={i === 0 ? "rw-history-now" : "rw-history-old"}>
                <VerdictMark verdict={d.verdict} />
                <span className="rw-muted">{d.actor.name} · {clock(d.at, locale)}{i > 0 ? ` · ${t("decide.superseded")}` : ""}</span>
                {d.reason !== undefined && <span>{t(`reason.${d.reason}`)}</span>}
                {d.note !== undefined && d.note !== "" && <span className="rw-quote">{d.note}</span>}
              </li>
            ))}
          </ol>
        )}
      </section>

      {props.group !== undefined && (
        <section className="rw-section">
          <h3>{t("intent.title")}</h3>
          <p className="rw-muted">{t("intent.help", { count: props.group.records.length })}</p>
          <div className="rw-intent">
            <Button kind="secondary" size="sm" onClick={() => props.onIntent("merge")} testId="intent-merge">{t("intent.merge")}</Button>
            <Button kind="secondary" size="sm" onClick={() => props.onIntent("split")} testId="intent-split">{t("intent.split")}</Button>
          </div>
          {props.intent !== undefined && (
            <p className="rw-note" data-testid="intent-recorded">
              {t(`intent.recorded.${props.intent.intent}`)} · {props.intent.actor.name} · {clock(props.intent.at, locale)}
              {props.intent.note !== undefined && props.intent.note !== "" && <span className="rw-quote">{props.intent.note}</span>}
            </p>
          )}
        </section>
      )}

      <section className="rw-section rw-keys">
        <h3>{t("keys.title")}</h3>
        <dl>
          {(Object.keys(KEYS) as KeyAction[]).map((k) => (
            <div key={k}><dt><kbd className="rw-kbd">{KEYS[k]}</kbd></dt><dd>{t(`keys.${k}`)}</dd></div>
          ))}
          <div><dt><kbd className="rw-kbd">⌘↵</kbd></dt><dd>{t("keys.confirm")}</dd></div>
        </dl>
      </section>
    </aside>
  );
}
