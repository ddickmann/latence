// The queue: review pairs grouped by the cluster they belong to, the group that settles the
// most first. A row is a link, so the open pair is in the URL.
import { CheckmarkFilled, ErrorFilled, Help } from "@carbon/icons-react";
import { useEffect, useRef } from "react";
import { Link } from "react-router";

import { nameOf, path } from "../format";
import { useHost } from "../host";
import type { QueueFilter, QueueGroup } from "../queue";
import type { Decision, Verdict } from "../types";

const VERDICT_ICON = { same: CheckmarkFilled, different: ErrorFilled, unsure: Help } as const;

export function VerdictMark({ verdict }: { verdict: Verdict }) {
  const { t } = useHost();
  const Icon = VERDICT_ICON[verdict];
  return (
    <span className={`rw-verdict rw-verdict--${verdict}`}>
      <Icon size={16} aria-hidden />
      <span>{t(`verdict.${verdict}`)}</span>
    </span>
  );
}

export function QueuePane(props: {
  groups: QueueGroup[];
  current: string | undefined;
  decisions: ReadonlyMap<string, Decision>;
  filter: QueueFilter;
  onFilter: (filter: QueueFilter) => void;
  systems: string[];
  counts: { open: number; decided: number; all: number };
}) {
  const { t, base, controls: { Select, Checkbox } } = useHost();
  const list = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const row = list.current?.querySelector<HTMLElement>('[aria-current="true"]');
    if (row !== null && row !== undefined && typeof row.scrollIntoView === "function") row.scrollIntoView({ block: "nearest" });
  }, [props.current]);

  const statuses: QueueFilter["status"][] = ["open", "decided", "all"];
  return (
    <section className="rw-queue" aria-label={t("queue.title")}>
      <div className="rw-queue-filters">
        <div className="rw-segments" role="group" aria-label={t("queue.show")}>
          {statuses.map((s) => (
            <button
              key={s}
              type="button"
              className="rw-segment"
              aria-pressed={props.filter.status === s}
              onClick={() => props.onFilter({ ...props.filter, status: s })}
              data-testid={`filter-${s}`}
            >
              {t(`queue.status.${s}`)} <span className="rw-num">{props.counts[s]}</span>
            </button>
          ))}
        </div>
        <div className="rw-queue-narrow">
          <Select
            id="rw-filter-system"
            label={t("queue.system")}
            value={props.filter.system}
            onChange={(system) => props.onFilter({ ...props.filter, system })}
            options={[{ value: "all", text: t("queue.anySystem") }, ...props.systems.map((s) => ({ value: s, text: s }))]}
          />
          <Checkbox
            id="rw-filter-conflicts"
            label={t("queue.conflictsOnly")}
            checked={props.filter.conflictsOnly}
            onChange={(conflictsOnly) => props.onFilter({ ...props.filter, conflictsOnly })}
          />
        </div>
      </div>
      <div className="rw-queue-list" ref={list}>
        {props.groups.length === 0 && <p className="rw-empty" data-testid="queue-empty">{t(props.filter.status === "open" ? "queue.emptyOpen" : "queue.empty")}</p>}
        {props.groups.map(({ group, items, decided }) => (
          <section key={group.id} className="rw-group" aria-label={t("queue.groupLabel", { name: group.labels[group.id] ?? group.id })}>
            <header className="rw-group-head">
              <span className="rw-group-name">{group.labels[group.id] ?? group.id}</span>
              <span className="rw-group-meta">
                {t("queue.groupMeta", { count: group.records.length })} · {t("queue.groupProgress", { decided, total: group.items.length })}
              </span>
            </header>
            <ul className="rw-rows">
              {items.map((item) => {
                const decision = props.decisions.get(item.id);
                return (
                  <li key={item.id}>
                    <Link
                      to={path(base, "review", item.id)}
                      className="rw-row"
                      aria-current={props.current === item.id ? "true" : undefined}
                      data-testid="queue-row"
                    >
                      <span className="rw-row-names">
                        <span>{nameOf(item.left)}</span>
                        <span>{nameOf(item.right)}</span>
                      </span>
                      <span className="rw-row-side">
                        {decision === undefined
                          ? <span className="rw-num" title={t("evidence.probability")}>{item.probability.toFixed(2)}</span>
                          : <VerdictMark verdict={decision.verdict} />}
                      </span>
                    </Link>
                  </li>
                );
              })}
            </ul>
          </section>
        ))}
      </div>
    </section>
  );
}
