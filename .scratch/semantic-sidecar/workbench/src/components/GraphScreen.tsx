// The graph view. Its front page is the schema level with the headline numbers; from there a
// person walks to a node and from a node to its neighbours. What the review band has not
// decided is shown as open, with a way straight into the review of it.
import { ArrowRight, ChevronLeft } from "@carbon/icons-react";
import { useEffect, useState } from "react";
import { Link } from "react-router";

import type { GraphDataSource } from "../dataSource";
import { fieldLabel, path, show } from "../format";
import { useHost } from "../host";
import { inForce } from "../ledger";
import type { GraphSchema, NeighbourGroup, NodeDetail, NodeRef, NodeType } from "../types";
import type { ReviewState } from "../useReview";
import { FocusGraph } from "./FocusGraph";
import { SchemaMap } from "./SchemaMap";
import { TypeGlyph } from "./TypeGlyph";

function useNumber(): (v: number) => string {
  const { locale } = useHost();
  return (v) => new Intl.NumberFormat(locale).format(v);
}

function NodeRow({ node, children }: { node: NodeRef; children?: React.ReactNode }) {
  const { t, base } = useHost();
  const n = useNumber();
  return (
    <li className="rw-node-row">
      <Link to={path(base, "graph", node.id)} className="rw-node-link" data-testid="node-link">
        <TypeGlyph type={node.type} size="sm" />
        <span className="rw-node-name">{node.label}</span>
      </Link>
      <span className="rw-muted rw-node-size">
        {children ?? t(node.type === "organisation" || node.type === "source_system" ? "graph.size.records" : "graph.size.organisations", { count: node.size, n: n(node.size) })}
      </span>
    </li>
  );
}

function GraphHome({ source, review }: { source: GraphDataSource; review: ReviewState }) {
  const { t, base, chrome, controls: { TextField } } = useHost();
  const n = useNumber();
  const [schema, setSchema] = useState<GraphSchema | null>(null);
  const [selected, setSelected] = useState<NodeType>("organisation");
  const [instances, setInstances] = useState<{ nodes: NodeRef[]; total: number }>({ nodes: [], total: 0 });
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<NodeRef[]>([]);

  chrome.useCrumbs([{ label: t("tabs.graph") }]);
  chrome.useListKeys({});

  useEffect(() => {
    let live = true;
    void source.loadSchema().then((s) => live && setSchema(s));
    return () => {
      live = false;
    };
  }, [source]);
  useEffect(() => {
    let live = true;
    void source.listInstances(selected, 12).then((r) => live && setInstances(r));
    return () => {
      live = false;
    };
  }, [source, selected]);
  useEffect(() => {
    let live = true;
    void source.search(query, 8).then((r) => live && setHits(r));
    return () => {
      live = false;
    };
  }, [source, query]);

  if (schema === null) return <p className="rw-empty" aria-busy="true">{t("graph.loading")}</p>;
  const { meta } = schema;
  const systems = schema.concepts.find((c) => c.id === "source_system")?.count ?? 0;
  const concept = schema.concepts.find((c) => c.id === selected);
  const attributes = schema.attributes.filter((a) => a.concept === selected);
  const openPairs = review.queue === null ? null : review.queue.items.length - inForce(review.ledger.decisions).size;
  const share = meta.organisations === 0 ? 0 : Math.round((meta.multi_record_organisations / meta.organisations) * 100);

  return (
    <div className="rw-graph" data-testid="graph-home">
      <header className="rw-graph-head">
        <h2 className="rw-graph-title">{t("graph.headline", { records: n(meta.records), systems: n(systems), organisations: n(meta.organisations) })}</h2>
        <dl className="rw-tiles">
          <div className="rw-tile"><dt>{t("graph.tile.records")}</dt><dd className="rw-num">{n(meta.records)}</dd></div>
          <div className="rw-tile"><dt>{t("graph.tile.organisations")}</dt><dd className="rw-num">{n(meta.organisations)}</dd></div>
          <div className="rw-tile"><dt>{t("graph.tile.multi")}</dt><dd className="rw-num">{share}%</dd></div>
          {openPairs !== null && (
            <div className="rw-tile">
              <dt>{t("graph.tile.open")}</dt>
              <dd>
                <span className="rw-num">{n(openPairs)}</span>
                <Link className="rw-link" to={path(base, "review")}>{t("graph.toReview")} <ArrowRight size={16} aria-hidden /></Link>
              </dd>
            </div>
          )}
        </dl>
      </header>

      <section className="rw-card" aria-labelledby="rw-schema-title">
        <div className="rw-card-head">
          <h3 id="rw-schema-title">{t("graph.schemaTitle")}</h3>
          <p className="rw-muted">{t(meta.schema_origin === "declared" ? "graph.schemaDeclared" : "graph.schemaInduced")}</p>
        </div>
        <SchemaMap concepts={schema.concepts} relations={schema.relations} selected={selected} onSelect={setSelected} />
      </section>

      <div className="rw-graph-cols">
        {concept !== undefined && (
          <section className="rw-card" aria-labelledby="rw-concept-title" data-testid="concept-detail">
            <div className="rw-card-head">
              <h3 id="rw-concept-title"><TypeGlyph type={concept.id} size="sm" /> {concept.label}</h3>
              <p className="rw-muted">{concept.definition}</p>
            </div>
            {attributes.length > 0 && (
              <ul className="rw-fills" aria-label={t("graph.attributes")}>
                {attributes.map((a) => (
                  <li key={a.id}>
                    <span>{a.label}</span>
                    <span className="rw-fill-track" aria-hidden="true"><span className="rw-fill" style={{ width: `${Math.round(a.filled * 100)}%` }} /></span>
                    <span className="rw-num">{t("graph.filled", { pct: Math.round(a.filled * 100) })}</span>
                  </li>
                ))}
              </ul>
            )}
            <h4>{t("graph.largest")}</h4>
            <ul className="rw-node-list">
              {instances.nodes.map((node) => <NodeRow key={node.id} node={node} />)}
            </ul>
            <p className="rw-muted rw-small">
              {selected === "organisation"
                ? t("graph.sampleNote", { shown: n(instances.nodes.length), sample: n(meta.organisations_in_sample), total: n(meta.organisations) })
                : t("graph.shownOf", { shown: n(instances.nodes.length), total: n(instances.total) })}
            </p>
          </section>
        )}

        <section className="rw-card" aria-labelledby="rw-search-title">
          <div className="rw-card-head"><h3 id="rw-search-title">{t("graph.find")}</h3></div>
          <TextField id="rw-graph-search" label={t("graph.searchLabel")} value={query} onChange={setQuery} placeholder={t("graph.searchHint")} />
          {query.trim() !== "" && (
            hits.length === 0
              ? <p className="rw-muted" data-testid="search-empty">{t("graph.noHits")}</p>
              : <ul className="rw-node-list" data-testid="search-hits">{hits.map((node) => <NodeRow key={node.id} node={node} />)}</ul>
          )}
        </section>
      </div>
    </div>
  );
}

function OpenLinks({ group, review }: { group: NeighbourGroup; review: ReviewState }) {
  const { t, base } = useHost();
  const decisions = inForce(review.ledger.decisions);
  return (
    <ul className="rw-node-list">
      {group.nodes.map((node) => {
        const items = node.items ?? [];
        const decided = items.filter((id) => decisions.has(id)).length;
        const next = items.find((id) => !decisions.has(id)) ?? items[0];
        return (
          <NodeRow key={node.id} node={node}>
            <span className="rw-open-link">
              <span className="rw-num">{(node.probability ?? 0).toFixed(2)}</span>
              {items.length > 0 && <span>{t("graph.pairsDecided", { decided, total: items.length })}</span>}
              {next !== undefined && <Link className="rw-link" to={path(base, "review", next)} data-testid="review-link">{t("graph.review")} <ArrowRight size={16} aria-hidden /></Link>}
            </span>
          </NodeRow>
        );
      })}
    </ul>
  );
}

function GraphNode({ source, review, id }: { source: GraphDataSource; review: ReviewState; id: string }) {
  const { t, base, chrome } = useHost();
  const n = useNumber();
  const [detail, setDetail] = useState<NodeDetail | null | undefined>(null);

  useEffect(() => {
    let live = true;
    setDetail(null);
    void source.loadNode(id).then((d) => live && setDetail(d));
    return () => {
      live = false;
    };
  }, [source, id]);

  chrome.useCrumbs(detail == null ? [{ label: t("tabs.graph"), to: path(base, "graph") }] : [{ label: t("tabs.graph"), to: path(base, "graph") }, { label: detail.node.label }]);
  chrome.useListKeys({});

  const back = <Link className="rw-link" to={path(base, "graph")} data-testid="back-to-graph"><ChevronLeft size={16} aria-hidden /> {t("graph.back")}</Link>;
  if (detail === null) return <p className="rw-empty" aria-busy="true">{t("graph.loading")}</p>;
  if (detail === undefined) return <div className="rw-graph"><nav>{back}</nav><p className="rw-empty" data-testid="node-unknown">{t("graph.unknown")}</p></div>;

  const { node } = detail;
  const systems = new Set((detail.records ?? []).map((r) => r["source_system"])).size;
  return (
    <div className="rw-graph" data-testid="graph-node">
      <nav className="rw-detail-nav" aria-label={t("graph.position")}>{back}</nav>
      <header className="rw-node-head">
        <TypeGlyph type={node.type} size="lg" />
        <div>
          <p className="rw-eyebrow">{t(`type.${node.type}`)}</p>
          <h2 className="rw-graph-title">{node.label}</h2>
          <p className="rw-muted">
            {detail.records !== undefined
              ? t("graph.orgSummary", { count: detail.records.length, n: n(detail.records.length), systems })
              : node.type === "source_system"
                ? t("graph.systemSummary", { n: n(node.size), listed: n(detail.listed ?? 0) })
                : t("graph.refSummary", { n: n(detail.total ?? node.size), listed: n(detail.listed ?? 0) })}
          </p>
        </div>
      </header>

      <section className="rw-card" aria-label={t("graph.neighbourhood")}>
        <FocusGraph detail={detail} />
      </section>

      <div className="rw-graph-cols">
        {detail.values !== undefined && (
          <section className="rw-card" aria-labelledby="rw-values-title" data-testid="node-values">
            <div className="rw-card-head">
              <h3 id="rw-values-title">{t("graph.valuesTitle")}</h3>
              <p className="rw-muted">{t("graph.valuesHelp")}</p>
            </div>
            <dl className="rw-values">
              {Object.entries(detail.values).map(([field, values]) => (
                <div key={field} className={values.length > 1 ? "rw-values-row rw-values-row--varies" : "rw-values-row"}>
                  <dt>{fieldLabel(t, field)}</dt>
                  <dd>
                    {values.length === 0 && <span className="rw-none">{t("evidence.noValue")}</span>}
                    {values.map((v) => (
                      <span key={v.value} className="rw-value">
                        <span>{v.value}</span>
                        <span className="rw-muted rw-num">{t("graph.ofRecords", { n: v.records, total: detail.records?.length ?? 0 })}</span>
                      </span>
                    ))}
                  </dd>
                </div>
              ))}
            </dl>
            <details className="rw-details">
              <summary>{t("graph.showRecords", { count: detail.records?.length ?? 0 })}</summary>
              <ul className="rw-record-list">
                {(detail.records ?? []).map((r) => (
                  <li key={r.unique_id}><span className="rw-mono">{r.unique_id}</span> · {show(r["source_system"])} · {show(r["name"])}</li>
                ))}
              </ul>
            </details>
          </section>
        )}

        <div className="rw-graph-groups">
          {detail.groups.map((group) => (
            <section key={group.relation} className={`rw-card rw-card--${group.status}`} aria-label={group.label} data-testid={`group-${group.relation}`}>
              <div className="rw-card-head">
                <h3>{group.label.charAt(0).toUpperCase() + group.label.slice(1)}</h3>
                {group.status === "open" && <p className="rw-muted">{t("graph.openHelp")}</p>}
              </div>
              {group.status === "open"
                ? <OpenLinks group={group} review={review} />
                : <ul className="rw-node-list">{group.nodes.slice(0, 20).map((nd) => <NodeRow key={nd.id} node={nd} />)}</ul>}
              {group.total > Math.min(20, group.nodes.length) && (
                <p className="rw-muted rw-small">{t("graph.shownOf", { shown: n(Math.min(20, group.nodes.length)), total: n(group.total) })}</p>
              )}
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}

export function GraphScreen(props: { source: GraphDataSource; review: ReviewState; node: string | undefined }) {
  return props.node === undefined
    ? <GraphHome source={props.source} review={props.review} />
    : <GraphNode source={props.source} review={props.review} id={props.node} />;
}
