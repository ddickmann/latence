// The three sheets of the review screen: a decision that needs a reason, a statement about a
// whole group, and the read-only audit of what the machine accepted on its own.
import { useId, useState } from "react";

import { nameOf } from "../format";
import { useHost } from "../host";
import type { Decision, ReviewItem, Verdict } from "../types";
import { VerdictMark } from "./QueuePane";
import { Sheet } from "./Sheet";

export const REASONS = ["legal_entity", "location", "business", "related", "other"] as const;
export type Reason = (typeof REASONS)[number];

/** "Different" always carries a reason; changing a recorded decision always carries a note. */
export function DecisionSheet(props: {
  item: ReviewItem;
  verdict: Verdict;
  replacing: Decision | undefined;
  onConfirm: (extra: { reason?: string; note?: string }) => void;
  onCancel: () => void;
}) {
  const { t, controls: { TextArea } } = useHost();
  const name = useId();
  const [reason, setReason] = useState<Reason | "">("");
  const [note, setNote] = useState("");
  const needsReason = props.verdict === "different";
  const needsNote = props.replacing !== undefined || reason === "other";
  const may = (!needsReason || reason !== "") && (!needsNote || note.trim() !== "");
  const confirm = () =>
    props.onConfirm({ ...(reason === "" ? {} : { reason }), ...(note.trim() === "" ? {} : { note: note.trim() }) });
  return (
    <Sheet
      title={t(props.replacing === undefined ? `sheet.title.${props.verdict}` : "sheet.title.change")}
      confirm={t(`sheet.confirm.${props.verdict}`)}
      onConfirm={confirm}
      onCancel={props.onCancel}
      canConfirm={may}
      danger={props.verdict === "different"}
      testId="decision-sheet"
    >
      <p className="rw-sheet-pair">{nameOf(props.item.left)} · {nameOf(props.item.right)}</p>
      {props.replacing !== undefined && (
        <p className="rw-note">{t("sheet.replaces")} <VerdictMark verdict={props.replacing.verdict} /></p>
      )}
      {needsReason && (
        <fieldset className="rw-choices">
          <legend>{t("sheet.reason")}</legend>
          {REASONS.map((r) => (
            <label key={r} className="rw-choice">
              <input type="radio" name={name} value={r} checked={reason === r} onChange={() => setReason(r)} data-testid={`reason-${r}`} />
              <span>{t(`reason.${r}`)}</span>
            </label>
          ))}
        </fieldset>
      )}
      <TextArea id={`${name}-note`} label={t(needsNote ? "sheet.noteRequired" : "sheet.note")} value={note} onChange={setNote} rows={3} />
    </Sheet>
  );
}

export function IntentSheet(props: { intent: "merge" | "split"; records: number; onConfirm: (note: string) => void; onCancel: () => void }) {
  const { t, controls: { TextArea } } = useHost();
  const id = useId();
  const [note, setNote] = useState("");
  return (
    <Sheet title={t(`intent.sheet.${props.intent}`, { count: props.records })} confirm={t("intent.confirm")} onConfirm={() => props.onConfirm(note.trim())} onCancel={props.onCancel} testId="intent-sheet">
      <p className="rw-muted">{t("intent.sheetHelp")}</p>
      <TextArea id={`${id}-note`} label={t("sheet.note")} value={note} onChange={setNote} rows={3} />
    </Sheet>
  );
}

export function AuditSheet({ audit, onClose }: { audit: ReviewItem[]; onClose: () => void }) {
  const { t } = useHost();
  const known = audit.filter((a) => a.gold_same !== undefined);
  const right = known.filter((a) => a.gold_same === true).length;
  return (
    <Sheet title={t("audit.title", { count: audit.length })} onCancel={onClose} wide testId="audit-sheet">
      <p className="rw-muted">{t("audit.help")}</p>
      {known.length > 0 && <p className="rw-note" data-testid="audit-summary">{t("audit.simulated", { right, total: known.length })}</p>}
      <div className="rw-table-scroll" tabIndex={0} role="region" aria-label={t("audit.table")}>
        <table className="rw-table">
          <thead>
            <tr>
              <th scope="col">{t("evidence.recordA")}</th>
              <th scope="col">{t("evidence.recordB")}</th>
              <th scope="col" className="rw-right">{t("audit.probability")}</th>
              {known.length > 0 && <th scope="col">{t("audit.truth")}</th>}
            </tr>
          </thead>
          <tbody>
            {audit.map((a) => (
              <tr key={a.id}>
                <td>{nameOf(a.left)}</td>
                <td>{nameOf(a.right)}</td>
                <td className="rw-right rw-num">{a.probability.toFixed(2)}</td>
                {known.length > 0 && <td>{a.gold_same === undefined ? "" : <VerdictMark verdict={a.gold_same ? "same" : "different"} />}</td>}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Sheet>
  );
}
