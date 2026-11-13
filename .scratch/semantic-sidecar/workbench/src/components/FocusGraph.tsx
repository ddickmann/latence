// One node and what it touches: the node in the middle, its neighbours around it by relation.
// Every neighbour is a link to its own neighbourhood, so the graph is walked one step at a time
// and never drawn as a tangle. The lists under the drawing carry the same, complete.
import { useRef } from "react";
import { Link } from "react-router";

import { path } from "../format";
import { layoutFocus } from "../graphLayout";
import { useHost } from "../host";
import type { NodeDetail } from "../types";
import { TypeGlyph } from "./TypeGlyph";
import { useWidth } from "./useWidth";

export function FocusGraph({ detail }: { detail: NodeDetail }) {
  const { t, base } = useHost();
  const frame = useRef<HTMLDivElement>(null);
  const width = useWidth(frame, 720);
  const layout = layoutFocus(detail.groups, width);
  return (
    <div className={`rw-focus${layout.narrow ? " rw-focus--narrow" : ""}`} ref={frame} style={{ height: layout.height }} data-testid="focus-graph">
      <svg className="rw-focus-edges" width={width} height={layout.height} aria-hidden="true">
        {layout.hubs.map((h) => (
          <line
            key={h.group.relation}
            x1={layout.centre.x} y1={layout.centre.y} x2={h.at.x} y2={h.at.y}
            className={`rw-spoke rw-spoke--${h.group.status}`}
          />
        ))}
        {layout.spokes.map((s) => {
          const hub = layout.hubs.find((h) => h.group === s.group)?.at ?? layout.centre;
          return (
            <line
              key={`${s.group.relation}:${s.node.id}`}
              x1={hub.x} y1={hub.y} x2={s.at.x} y2={s.at.y}
              className={`rw-spoke rw-spoke--${s.group.status}`}
            />
          );
        })}
      </svg>
      {layout.hubs.map((h) => (
        <span key={h.group.relation} className={`rw-focus-rel rw-focus-rel--${h.group.status}`} style={{ left: h.at.x, top: h.at.y }} aria-hidden="true">
          {h.group.label}{h.hidden > 0 ? ` · ${t("graph.more", { n: h.hidden })}` : ""}
        </span>
      ))}
      <span className="rw-focus-centre" style={{ left: layout.centre.x, top: layout.centre.y }}>
        <TypeGlyph type={detail.node.type} size="lg" />
        <span className="rw-sr">{detail.node.label}</span>
      </span>
      {layout.spokes.map((s) => (
        <Link
          key={`${s.group.relation}:${s.node.id}`}
          to={path(base, "graph", s.node.id)}
          className="rw-focus-node"
          style={{ left: s.at.x, top: s.at.y }}
          title={`${s.group.label}: ${s.node.label}`}
          aria-label={`${s.group.label}: ${s.node.label}`}
          data-testid="focus-node"
        >
          <TypeGlyph type={s.node.type} />
          <span className="rw-focus-label">{s.node.label}</span>
        </Link>
      ))}
    </div>
  );
}
