// The evidence for one pair: the two records side by side, what the matcher weighed field by
// field, where the calibrated probability sits between the bands, and the group around the pair.
import { ArrowRight, Subtract } from "@carbon/icons-react";
import { useEffect, useState } from "react";
import { Link } from "react-router";

import type { GraphDataSource } from "../dataSource";
import { bridge, explain, lean, markShared, type Lean } from "../explain";
import { fieldLabel, nameOf, path, show, signed } from "../format";
import { useHost } from "../host";
import type { Decision, EvidenceRow, NodeRef, ReviewGroup, ReviewItem, ReviewMeta, Scalar } from "../types";
import { ClusterGraph } from "./ClusterGraph";

function agreement(a: Scalar | undefined, b: Scalar | undefined): "same" | "differ" | "missing" {
  if (a == null || b == null) return "missing";
  return String(a).trim().toLowerCase() === String(b).trim().toLowerCase() ? "same" : "differ";
}

function Value({ value, other, mark }: { value: Scalar | undefined; other: Scalar | undefined; mark: boolean }) {
  const { t } = useHost();
  if (value == null) return <span className="rw-none">{t("evidence.noValue")}</span>;
  if (!mark || other == null) return <>{show(value)}</>;
  return (
    <>
      {markShared(String(value), String(other)).map((w, i) => (
        <span key={i}>{i > 0 && " "}{w.shared ? w.word : <mark className="rw-diff">{w.word}</mark>}</span>
      ))}
    </>
  );
}

/** Where the pair's probability sits between "reject below" and "accept at". */
export function ProbabilityScale({ item, meta }: { item: ReviewItem; meta: ReviewMeta }) {
  const { t } = useHost();
  const lo = meta.bands.reject_below ?? 0;
  const hi = meta.bands.accept_at ?? 1;
  const pct = (v: number) => `${Math.min(100, Math.max(0, v * 100)).toFixed(1)}%`;
  // Thresholds to three places: at two, a pair just under the accept band would read as on it.
  const cut = (v: number) => String(Number(v.toFixed(3)));
  return (
    <figure className="rw-scale" aria-label={t("evidence.scaleLabel", { p: item.probability.toFixed(2), lo: cut(lo), hi: cut(hi) })}>
      <div className="rw-scale-track" aria-hidden="true">
        <span className="rw-scale-zone rw-scale-zone--reject" style={{ width: pct(lo) }} />
        <span className="rw-scale-zone rw-scale-zone--review" style={{ left: pct(lo), width: pct(hi - lo) }} />
        <span className="rw-scale-zone rw-scale-zone--accept" style={{ left: pct(hi) }} />
        <span className="rw-scale-pin" style={{ left: pct(item.probability) }} />
      </div>
      <figcaption className="rw-scale-legend">
        <span><b className="rw-num">{item.probability.toFixed(2)}</b> {t("evidence.probability")}</span>
        <span>{t("evidence.bands", { lo: cut(lo), hi: cut(hi) })}</span>
      </figcaption>
    </figure>
  );
}

const LEAN_ORDER: Lean[] = ["for", "against", "missing"];

function EvidenceBar({ row, scale }: { row: EvidenceRow; scale: number }) {
  const { t } = useHost();
  const l = lean(row);
  const { key, vars } = explain(row);
  const width = `${Math.min(50, (Math.abs(row.weight) / scale) * 50).toFixed(1)}%`;
  return (
    <li className={`rw-bar rw-bar--${l}`} data-testid="evidence-row">
      <div className="rw-bar-head">
        <span className="rw-bar-label">{row.label}</span>
        <span className="rw-bar-lean">{t(`lean.${l}`)}</span>
        <span className="rw-num rw-bar-weight">{signed(row.weight)}</span>
      </div>
      <div className="rw-bar-track" aria-hidden="true">
        <span className="rw-bar-fill" style={row.weight >= 0 ? { left: "50%", width } : { right: "50%", width }} />
      </div>
      <p className="rw-bar-why">{t(key, vars)}</p>
    </li>
  );
}

export function EvidencePane(props: {
  item: ReviewItem;
  group: ReviewGroup | undefined;
  meta: ReviewMeta;
  decisions: ReadonlyMap<string, Decision>;
  graph: GraphDataSource | undefined;
}) {
  const { t, base } = useHost();
  const { item, group, meta } = props;
  const [entities, setEntities] = useState<{ left?: NodeRef; right?: NodeRef }>({});

  useEffect(() => {
    let live = true;
    setEntities({});
    if (props.graph === undefined) return;
    void Promise.all([props.graph.entityOfRecord(item.left.unique_id), props.graph.entityOfRecord(item.right.unique_id)]).then(([left, right]) => {
      if (live) setEntities({ ...(left === undefined ? {} : { left }), ...(right === undefined ? {} : { right }) });
    });
    return () => {
      live = false;
    };
  }, [props.graph, item.left.unique_id, item.right.unique_id]);

  const fields = meta.record_fields.filter((f) => f !== "unique_id" && f !== "source_system");
  const scale = Math.max(1, ...item.evidence.map((e) => Math.abs(e.weight)));
  const fieldsSum = item.evidence.reduce((sum, e) => sum + e.weight, 0);
  const split = group === undefined ? null : bridge(group, { left: item.left.unique_id, right: item.right.unique_id }, props.decisions);

  const head = (side: "left" | "right") => {
    const record = item[side];
    const entity = entities[side];
    return (
      <th scope="col">
        <span className="rw-rec-side">{t(side === "left" ? "evidence.recordA" : "evidence.recordB")}</span>
        <span className="rw-rec-source">{show(record["source_system"])} · <span className="rw-mono">{record.unique_id}</span></span>
        {entity !== undefined && (
          <Link className="rw-link" to={path(base, "graph", entity.id)} data-testid={`open-graph-${side}`}>
            {t("evidence.openInGraph")} <ArrowRight size={16} aria-hidden />
          </Link>
        )}
      </th>
    );
  };

  return (
    <article className="rw-evidence" data-testid="evidence" data-item={item.id}>
      <header className="rw-ev-head">
        <p className="rw-eyebrow">{t("evidence.question")}</p>
        <h2 className="rw-ev-title">
          <span>{nameOf(item.left)}</span>
          <span className="rw-ev-and">{t("evidence.and")}</span>
          <span>{nameOf(item.right)}</span>
        </h2>
        <ProbabilityScale item={item} meta={meta} />
      </header>

      <section className="rw-section" aria-labelledby="rw-why">
        <h3 id="rw-why">{t("evidence.whyUnsure")}</h3>
        <ul className="rw-why">
          {LEAN_ORDER.map((l) => {
            const rows = item.evidence.filter((e) => lean(e) === l).sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
            if (rows.length === 0) return null;
            return (
              <li key={l} className={`rw-why-line rw-why-line--${l}`}>
                <span className="rw-why-key">{t(`why.${l}`)}</span>
                <span>{rows.map((e) => e.label).join(", ")}</span>
              </li>
            );
          })}
        </ul>
        {item.vetoed && <p className="rw-note">{t("evidence.vetoed")}</p>}
      </section>

      <section className="rw-section" aria-labelledby="rw-records">
        <h3 id="rw-records">{t("evidence.records")}</h3>
        <table className="rw-records">
          <thead>
            <tr>
              <th scope="col"><span className="rw-sr">{t("evidence.field")}</span></th>
              {head("left")}
              {head("right")}
            </tr>
          </thead>
          <tbody>
            {fields.map((f) => {
              const a = item.left[f];
              const b = item.right[f];
              const agrees = agreement(a, b);
              return (
                <tr key={f} className={`rw-rec-row rw-rec-row--${agrees}`}>
                  <th scope="row">
                    {fieldLabel(t, f)}
                    <span className="rw-rec-agree">{agrees === "missing" ? <Subtract size={16} aria-hidden /> : agrees === "same" ? "=" : "≠"}<span className="rw-sr"> {t(`agree.${agrees}`)}</span></span>
                  </th>
                  <td><Value value={a} other={b} mark={agrees === "differ"} /></td>
                  <td><Value value={b} other={a} mark={agrees === "differ"} /></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      <section className="rw-section" aria-labelledby="rw-weighed">
        <h3 id="rw-weighed">{t("evidence.weighed")}</h3>
        <p className="rw-lede">
          {t("evidence.sum", { prior: signed(item.prior_weight), fields: signed(fieldsSum), total: signed(item.match_weight) })}
        </p>
        <div className="rw-bar-axis" aria-hidden="true"><span>{t("lean.against")}</span><span>{t("lean.for")}</span></div>
        <ul className="rw-bars">
          {item.evidence.map((row) => <EvidenceBar key={row.comparison} row={row} scale={scale} />)}
        </ul>
      </section>

      {group !== undefined && (
        <section className="rw-section" aria-labelledby="rw-cluster">
          <h3 id="rw-cluster">{t("cluster.title", { count: group.records.length })}</h3>
          {split !== null && (
            <p className="rw-callout" data-testid="bridge">{t("cluster.bridge", { a: split[0].length, b: split[1].length })}</p>
          )}
          <ClusterGraph group={group} current={item} decisions={props.decisions} />
        </section>
      )}
    </article>
  );
}
