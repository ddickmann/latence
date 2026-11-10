// The numbers a reviewer and their lead watch: what is left, what is decided, the pace, and a
// way into the audit of what the machine accepted on its own.
import { useHost } from "../host";
import type { ReviewMeta, Verdict } from "../types";

export function QaStrip(props: {
  meta: ReviewMeta;
  open: number;
  tally: Record<Verdict, number>;
  pace: number | null;
  onAudit: () => void;
}) {
  const { t, locale, controls: { Button } } = useHost();
  const n = (v: number) => new Intl.NumberFormat(locale).format(v);
  const decided = props.tally.same + props.tally.different + props.tally.unsure;
  return (
    <div className="rw-qa" data-testid="qa">
      <dl className="rw-qa-stats">
        <div><dt>{t("qa.open")}</dt><dd className="rw-num" data-testid="qa-open">{n(props.open)}</dd></div>
        <div><dt>{t("qa.decided")}</dt><dd className="rw-num" data-testid="qa-decided">{n(decided)}</dd></div>
        <div><dt>{t("qa.pace")}</dt><dd className="rw-num">{props.pace === null ? "–" : t("qa.perHour", { n: n(props.pace) })}</dd></div>
        <div className="rw-qa-wide"><dt>{t("qa.band")}</dt><dd>{t("qa.bandValue", { pairs: n(props.meta.counts.review), groups: n(props.meta.review_groups_total), loaded: n(props.meta.review_groups_exported) })}</dd></div>
      </dl>
      <Button kind="quiet" size="sm" onClick={props.onAudit} testId="open-audit">{t("qa.audit", { count: n(props.meta.counts.accept) })}</Button>
    </div>
  );
}
