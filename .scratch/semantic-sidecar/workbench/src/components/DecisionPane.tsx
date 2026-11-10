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
  /** The latest decision about the whole group, and how many of its pair decisions still stand. */
  intent: GroupIntent | undefined;
  standing: number;
  onGroup: (intent: "merge" | "split") => void;
  onUndo: (batch: string) => void;
}) {
  const { t, locale, controls: { Button } } = useHost();
  const last = props.history[props.history.length - 1];
  const inForce = last?.withdraws === true ? undefined : last;
  return (
    <aside className="rw-aside" aria-label={t("decide.recordTitle")}>
      <section className="rw-section">
        <h3>{t("decide.recordTitle")}</h3>
        {inForce === undefined && <p className="rw-muted" data-testid="no-decision">{t("decide.none")}</p>}
        {props.history.length > 0 && (
          <ol className="rw-history" data-testid="history">
            {[...props.history].reverse().map((d, i) => (
              <li key={d.at + String(i)} className={i === 0 && d.withdraws !== true ? "rw-history-now" : "rw-history-old"}>
                <span className="rw-history-what">
                  <VerdictMark verdict={d.verdict} />
                  {d.withdraws === true && <span className="rw-tag">{t("decide.withdrawn")}</span>}
                  {d.batch !== undefined && d.withdraws !== true && <span className="rw-tag">{t("decide.byGroup")}</span>}
                </span>
                <span className="rw-muted">{d.actor.name} · {clock(d.at, locale)}{i > 0 && d.withdraws !== true ? ` · ${t("decide.superseded")}` : ""}</span>
                {d.reason !== undefined && <span>{t(`reason.${d.reason}`)}</span>}
                {d.note !== undefined && d.note !== "" && <span className="rw-quote">{d.note}</span>}
              </li>
            ))}
          </ol>
        )}
      </section>

      {props.group !== undefined && (
        <section className="rw-section">
          <h3>{t("group.title")}</h3>
          <p className="rw-muted">{t("group.help", { count: props.group.records.length })}</p>
          <div className="rw-intent">
            <Button kind="secondary" size="sm" kbd={KEYS.merge} onClick={() => props.onGroup("merge")} testId="group-merge">{t("group.merge")}</Button>
            <Button kind="secondary" size="sm" kbd={KEYS.split} onClick={() => props.onGroup("split")} testId="group-split">{t("group.split")}</Button>
          </div>
          {props.intent !== undefined && (
            <div className="rw-note" data-testid="group-recorded">
              <p>
                {t(`group.recorded.${props.intent.intent}`, { count: props.intent.decided })} · {props.intent.actor.name} · {clock(props.intent.at, locale)}
              </p>
              {props.intent.separates_accepted !== undefined && props.intent.separates_accepted.length > 0 && (
                <p>{t("group.recordedSeparates", { count: props.intent.separates_accepted.length })}</p>
              )}
              {props.intent.note !== undefined && props.intent.note !== "" && <span className="rw-quote">{props.intent.note}</span>}
              {props.standing > 0
                ? <Button kind="quiet" size="sm" onClick={() => props.onUndo(props.intent?.batch ?? "")} testId="group-undo">{t("group.undo", { count: props.standing })}</Button>
                : <p data-testid="group-undone">{t("group.undone")}</p>}
            </div>
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
