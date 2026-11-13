// The schema level, drawn: each concept a box with its count, each relation an arrow that reads
// as a sentence ("Organisation — located in → City"). Boxes are buttons; the table of relations
// under the drawing says the same in words.
import { useRef } from "react";

import { layoutSchema } from "../graphLayout";
import { useHost } from "../host";
import type { Concept, NodeType, Relation } from "../types";
import { TypeGlyph } from "./TypeGlyph";
import { useWidth } from "./useWidth";

export function SchemaMap(props: {
  concepts: Concept[];
  relations: Relation[];
  selected: NodeType;
  onSelect: (concept: NodeType) => void;
}) {
  const { t, locale } = useHost();
  const frame = useRef<HTMLDivElement>(null);
  const width = useWidth(frame, 720);
  const layout = layoutSchema(props.concepts, props.relations, width);
  const n = (v: number) => new Intl.NumberFormat(locale).format(v);
  const offset = layout.horizontal ? Math.max(0, (width - layout.width) / 2) : 0;

  return (
    <div className={`rw-schema${layout.horizontal ? "" : " rw-schema--narrow"}`} ref={frame}>
      <div className="rw-schema-stage" style={{ height: layout.height, width: layout.horizontal ? layout.width : "100%", marginInlineStart: offset }}>
        <svg className="rw-schema-edges" width={layout.horizontal ? layout.width : width} height={layout.height} aria-hidden="true">
          <defs>
            <marker id="rw-arrow" viewBox="0 0 8 8" refX="1" refY="4" markerWidth="7" markerHeight="7" orient="auto">
              <path d="M0,0 L8,4 L0,8 z" className="rw-arrow" />
            </marker>
          </defs>
          {layout.edges.map((e) => <path key={e.relation.id} d={e.path} className="rw-schema-edge" markerEnd="url(#rw-arrow)" />)}
        </svg>
        {layout.edges.map((e) => (
          <span key={e.relation.id} className="rw-schema-rel" style={{ left: e.mid.x, top: e.mid.y }} aria-hidden="true">
            <span>{e.relation.label}</span>
            <span className="rw-num">{n(e.relation.count)}</span>
          </span>
        ))}
        {props.concepts.map((c) => {
          const at = layout.nodes.get(c.id);
          if (at === undefined) return null;
          return (
            <button
              key={c.id}
              type="button"
              className="rw-concept"
              style={{ left: at.x - layout.box.w / 2, top: at.y - layout.box.h / 2, width: layout.box.w, height: layout.box.h }}
              aria-pressed={props.selected === c.id}
              onClick={() => props.onSelect(c.id)}
              data-testid={`concept-${c.id}`}
            >
              <TypeGlyph type={c.id} />
              <span className="rw-concept-text">
                <span className="rw-concept-name">{c.label}</span>
                <span className="rw-num rw-concept-count">{n(c.count)}</span>
              </span>
            </button>
          );
        })}
      </div>
      <table className="rw-sr">
        <caption>{t("graph.relations")}</caption>
        <tbody>
          {props.relations.map((r) => (
            <tr key={r.id}>
              <td>{props.concepts.find((c) => c.id === r.from)?.label}</td>
              <td>{r.label}</td>
              <td>{props.concepts.find((c) => c.id === r.to)?.label}</td>
              <td>{n(r.count)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
