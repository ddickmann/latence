// The group around a pair, drawn: its records on a ring, the links the machine accepted, the
// links still in review, and what the reviewer has said so far. The list beside it says the same
// in words, so nothing is carried by the drawing alone.
import { useHost } from "../host";
import type { Decision, ReviewGroup, ReviewItem } from "../types";

const W = 320;
const H = 220;

export function ClusterGraph({ group, current, decisions, parts, bare }: {
  group: ReviewGroup;
  current: ReviewItem;
  decisions: ReadonlyMap<string, Decision>;
  /** While a split is being drawn up: each record's part. Nodes show the part, and a link between parts is drawn as cut. */
  parts?: ReadonlyMap<string, string>;
  /** The drawing alone, without the list and legend beside it. */
  bare?: boolean;
}) {
  const { t } = useHost();
  const n = group.records.length;
  const at = new Map(group.records.map((id, i) => {
    const angle = (2 * Math.PI * i) / n - Math.PI / 2;
    return [id, { x: W / 2 + 118 * Math.cos(angle), y: H / 2 + 82 * Math.sin(angle) }] as const;
  }));
  const isCurrent = (l: string, r: string) =>
    (l === current.left.unique_id && r === current.right.unique_id) || (l === current.right.unique_id && r === current.left.unique_id);
  const drawn = group.edges.filter((e) => e.band !== "reject");
  const state = (e: { left: string; right: string; band: string }) => {
    if (parts !== undefined && parts.get(e.left) !== parts.get(e.right)) return "different";
    return e.band === "accept" ? "accepted" : decisions.get(`${e.left}~${e.right}`)?.verdict ?? "open";
  };

  return (
    <div className={bare === true ? "rw-cluster rw-cluster--bare" : "rw-cluster"}>
      <svg className="rw-cluster-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={t("cluster.drawing", { count: n, links: drawn.length })}>
        {drawn.map((e) => {
          const a = at.get(e.left);
          const b = at.get(e.right);
          if (a === undefined || b === undefined) return null;
          return (
            <line
              key={`${e.left}~${e.right}`}
              x1={a.x} y1={a.y} x2={b.x} y2={b.y}
              className={`rw-edge rw-edge--${state(e)}${parts === undefined && isCurrent(e.left, e.right) ? " rw-edge--current" : ""}`}
            />
          );
        })}
        {group.records.map((id, i) => {
          const p = at.get(id);
          if (p === undefined) return null;
          const on = parts === undefined && (id === current.left.unique_id || id === current.right.unique_id);
          return (
            <g key={id} className={`rw-cnode${on ? " rw-cnode--on" : ""}`}>
              <circle cx={p.x} cy={p.y} r={13} />
              <text x={p.x} y={p.y} dy="0.35em" textAnchor="middle">{parts === undefined ? i + 1 : parts.get(id)}</text>
            </g>
          );
        })}
      </svg>
      {bare !== true && <div className="rw-cluster-side">
        <ol className="rw-cluster-list">
          {group.records.map((id) => (
            <li key={id} className={id === current.left.unique_id || id === current.right.unique_id ? "rw-on" : undefined}>
              {group.labels[id] ?? id}
            </li>
          ))}
        </ol>
        <ul className="rw-legend">
          <li><span className="rw-swatch rw-swatch--accepted" aria-hidden="true" />{t("cluster.accepted")}</li>
          <li><span className="rw-swatch rw-swatch--open" aria-hidden="true" />{t("cluster.open")}</li>
          <li><span className="rw-swatch rw-swatch--same" aria-hidden="true" />{t("cluster.same")}</li>
          <li><span className="rw-swatch rw-swatch--different" aria-hidden="true" />{t("cluster.different")}</li>
        </ul>
      </div>}
    </div>
  );
}
