// The sheets of the review screen: a decision that needs a reason, the two decisions about a
// whole group, and the read-only audit of what the machine accepted on its own.
import { useId, useState } from "react";

import { nameOf } from "../format";
import { count, planMerge, planSplit, suggestParts, type GroupPlan } from "../groupActions";
import { useHost } from "../host";
import type { Decision, ReviewGroup, ReviewItem, Verdict } from "../types";
import { ClusterGraph } from "./ClusterGraph";
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

function PlanNotes({ plan }: { plan: GroupPlan }) {
  const { t } = useHost();
  return (
    <>
      {plan.kept > 0 && <p className="rw-note" data-testid="plan-kept">{t("group.kept", { count: plan.kept })}</p>}
      {plan.disagree.length > 0 && <p className="rw-callout" data-testid="plan-disagree">{t("group.disagree", { count: plan.disagree.length })}</p>}
      {plan.separatesAccepted.length > 0 && <p className="rw-callout" data-testid="plan-separates">{t("group.separates", { count: plan.separatesAccepted.length })}</p>}
    </>
  );
}

/** "All one": every open pair of the group becomes the same. */
export function MergeSheet(props: {
  group: ReviewGroup;
  items: ReadonlyMap<string, ReviewItem>;
  decisions: ReadonlyMap<string, Decision>;
  onConfirm: (note: string) => void;
  onCancel: () => void;
}) {
  const { t, controls: { TextArea } } = useHost();
  const id = useId();
  const [note, setNote] = useState("");
  const plan = planMerge(props.group, props.items, props.decisions);
  const open = plan.decisions.length;
  return (
    <Sheet
      title={t("group.mergeTitle", { count: props.group.records.length })}
      confirm={t("group.mergeConfirm", { count: open })}
      onConfirm={() => props.onConfirm(note.trim())}
      onCancel={props.onCancel}
      canConfirm={open > 0}
      testId="merge-sheet"
    >
      <p data-testid="plan-outcome">{open > 0 ? t("group.mergeOutcome", { count: open }) : t("group.nothingOpen")}</p>
      <PlanNotes plan={plan} />
      <TextArea id={`${id}-note`} label={t("sheet.note")} value={note} onChange={setNote} rows={2} />
    </Sheet>
  );
}

const PART_NAMES = ["A", "B", "C", "D"];

/** A split: the reviewer says which records belong together; the pairs follow from that. */
export function SplitSheet(props: {
  group: ReviewGroup;
  items: ReadonlyMap<string, ReviewItem>;
  decisions: ReadonlyMap<string, Decision>;
  current: ReviewItem;
  onConfirm: (parts: string[][], note: string) => void;
  onCancel: () => void;
}) {
  const { t, controls: { TextArea } } = useHost();
  const id = useId();
  const { group } = props;
  const offered = PART_NAMES.slice(0, Math.min(PART_NAMES.length, group.records.length));
  // Start from what is already firm: records the machine linked, or the reviewer called the same.
  const [partOf, setPartOf] = useState<Record<string, number>>(() => {
    const start: Record<string, number> = {};
    suggestParts(group, props.decisions).forEach((part, index) => {
      for (const record of part) start[record] = Math.min(index, offered.length - 1);
    });
    return start;
  });
  const [note, setNote] = useState("");
  const parts = offered.map((_name, index) => group.records.filter((r) => (partOf[r] ?? 0) === index)).filter((p) => p.length > 0);
  const plan = planSplit(group, props.items, props.decisions, parts);
  const { same, different } = count(plan);
  const may = parts.length >= 2 && plan.decisions.length > 0;
  const letters = new Map(group.records.map((r) => [r, offered[partOf[r] ?? 0] ?? "A"] as const));
  return (
    <Sheet
      title={t("group.splitTitle", { count: group.records.length })}
      confirm={t("group.splitConfirm", { count: plan.decisions.length })}
      onConfirm={() => props.onConfirm(parts, note.trim())}
      onCancel={props.onCancel}
      canConfirm={may}
      wide
      testId="split-sheet"
    >
      <p className="rw-muted">{t("group.splitHelp")}</p>
      <div className="rw-split">
        <ClusterGraph group={group} current={props.current} decisions={props.decisions} parts={letters} bare />
        <ul className="rw-split-list">
          {group.records.map((record) => (
            <li key={record} className="rw-split-row">
              <span className="rw-split-name">{group.labels[record] ?? record}</span>
              <span className="rw-segments rw-segments--parts" role="group" aria-label={t("group.partOf", { name: group.labels[record] ?? record })}>
                {offered.map((name, index) => (
                  <button
                    key={name}
                    type="button"
                    className="rw-segment"
                    aria-pressed={(partOf[record] ?? 0) === index}
                    onClick={() => setPartOf((now) => ({ ...now, [record]: index }))}
                    data-testid={`part-${record}-${name}`}
                  >
                    {name}
                  </button>
                ))}
              </span>
            </li>
          ))}
        </ul>
      </div>
      <p data-testid="plan-outcome">
        {parts.length < 2 ? t("group.onePart") : plan.decisions.length === 0 ? t("group.nothingOpen") : t("group.splitOutcome", { parts: parts.length, same, different })}
      </p>
      <PlanNotes plan={plan} />
      <TextArea id={`${id}-note`} label={t("sheet.note")} value={note} onChange={setNote} rows={2} />
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
