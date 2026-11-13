// The review screen: queue, evidence, decision. The open pair is in the URL; every action has
// one key; a decision moves on to the next undecided pair.
import { ChevronLeft } from "@carbon/icons-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router";

import type { GraphDataSource, ReviewDataSource } from "../dataSource";
import { nameOf, path } from "../format";
import { suggested } from "../groupActions";
import { useHost } from "../host";
import { actionFor, isTyping, onControl, type KeyAction } from "../keys";
import { agreement, historyOf, inForce, latestIntent, pace, standing, tally } from "../ledger";
import { DEFAULT_FILTER, nextOpen, order, step, systemsOf, visibleGroups, type QueueFilter } from "../queue";
import type { Decision, ReviewItem, Verdict } from "../types";
import type { ReviewState } from "../useReview";
import { DecisionBar, DecisionPane } from "./DecisionPane";
import { EvidencePane } from "./EvidencePane";
import { QaStrip } from "./QaStrip";
import { QueuePane } from "./QueuePane";
import { AuditSheet, DecisionSheet, MergeSheet, SplitSheet } from "./Sheets";

type SheetState =
  | { kind: "decide"; verdict: Verdict; replacing: Decision | undefined }
  | { kind: "group"; intent: "merge" | "split" }
  | { kind: "audit"; items: ReviewItem[] }
  | null;

export function ReviewScreen(props: { state: ReviewState; source: ReviewDataSource; graph: GraphDataSource | undefined; item: string | undefined }) {
  const { t, chrome, base } = useHost();
  const navigate = useNavigate();
  const { queue, ledger } = props.state;
  const [filter, setFilter] = useState<QueueFilter>(DEFAULT_FILTER);
  const [sheet, setSheet] = useState<SheetState>(null);

  const items = useMemo(() => new Map((queue?.items ?? []).map((i) => [i.id, i] as const)), [queue]);
  const decisions = useMemo(() => inForce(ledger.decisions), [ledger.decisions]);
  const groups = useMemo(
    () => (queue === null ? [] : visibleGroups(queue.groups, items, decisions, filter, props.item)),
    [queue, items, decisions, filter, props.item],
  );
  const ids = useMemo(() => order(groups), [groups]);
  const item = props.item === undefined ? undefined : items.get(props.item);
  const group = item === undefined ? undefined : queue?.groups.find((g) => g.id === item.group);
  const counts = useMemo(() => {
    const all = items.size;
    const decided = [...items.keys()].filter((id) => decisions.has(id)).length;
    return { all, decided, open: all - decided };
  }, [items, decisions]);

  const open = useCallback(
    (id: string | undefined) => navigate(id === undefined ? path(base, "review") : path(base, "review", id)),
    [navigate, base],
  );

  const record = async (target: ReviewItem, verdict: Verdict, extra: { reason?: string; note?: string } = {}) => {
    try {
      const decision = await props.state.decide({ item: target.id, verdict, ...extra, ...suggested(target) });
      chrome.notify({ tone: "positive", title: t(`notice.${verdict}`), detail: `${nameOf(target.left)} · ${nameOf(target.right)}` });
      chrome.recount();
      const after = new Map(decisions).set(target.id, decision);
      const next = nextOpen(ids, target.id, after);
      if (next !== undefined) open(next);
    } catch {
      chrome.notify({ tone: "critical", title: t("notice.failed") });
    }
  };

  const act = (action: KeyAction) => {
    if (action === "next" || action === "previous") {
      if (sheet === null) open(step(ids, props.item, action === "next" ? 1 : -1));
      return;
    }
    if (action === "close") {
      if (sheet !== null) setSheet(null);
      else if (props.item !== undefined) open(undefined);
      return;
    }
    if (sheet !== null) return;
    if (action === "skip") {
      open(nextOpen(ids, props.item, decisions) ?? step(ids, props.item, 1));
      return;
    }
    if (item === undefined) return;
    if (action === "merge" || action === "split") {
      if (group !== undefined) setSheet({ kind: "group", intent: action });
      return;
    }
    const existing = decisions.get(item.id);
    if (action === "take") {
      // The suggestion as it stands: one key, and "different" needs no reason of its own.
      const verdict = item.suggestion?.verdict;
      if (verdict === undefined || existing?.verdict === verdict) return;
      if (existing !== undefined) setSheet({ kind: "decide", verdict, replacing: existing });
      else void record(item, verdict, verdict === "different" ? { reason: "as_suggested" } : {});
      return;
    }
    if (existing?.verdict === action) return;
    if (action === "different" || existing !== undefined) setSheet({ kind: "decide", verdict: action, replacing: existing });
    else void record(item, action);
  };

  const undo = async (batch: string) => {
    try {
      const withdrawn = await props.state.withdrawBatch(batch);
      chrome.notify({ tone: "info", title: t("notice.undone", { count: withdrawn.length }) });
      chrome.recount();
    } catch {
      chrome.notify({ tone: "critical", title: t("notice.failed") });
    }
  };

  const decideGroup = async (intent: "merge" | "split", note: string, parts?: string[][]) => {
    if (group === undefined || item === undefined) return;
    try {
      const done = await props.state.decideGroup({ group: group.id, intent, ...(parts === undefined ? {} : { parts }), ...(note === "" ? {} : { note }) });
      const batch = done.intent.batch;
      chrome.notify({
        tone: "positive",
        title: t(`group.recorded.${intent}`, { count: done.decisions.length }),
        action: { label: t("group.undoShort"), run: () => void undo(batch) },
      });
      chrome.recount();
      const after = new Map(decisions);
      for (const d of done.decisions) after.set(d.item, d);
      const next = nextOpen(ids, item.id, after);
      if (next !== undefined && after.has(item.id)) open(next);
    } catch {
      chrome.notify({ tone: "critical", title: t("notice.failed") });
    }
  };

  // The latest `act`, for listeners registered once.
  const acting = useRef(act);
  acting.current = act;

  chrome.useCrumbs(item === undefined
    ? [{ label: t("tabs.review") }]
    : [{ label: t("tabs.review"), to: path(base, "review") }, { label: `${nameOf(item.left)} · ${nameOf(item.right)}` }]);
  chrome.useListKeys({
    next: () => acting.current("next"),
    previous: () => acting.current("previous"),
    close: () => acting.current("close"),
  });

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;
      const action = actionFor({ key: event.key, metaKey: event.metaKey, ctrlKey: event.ctrlKey, altKey: event.altKey, typing: isTyping(event.target) });
      if (action === null || action === "next" || action === "previous" || action === "close") return;
      if (action === "skip" && onControl(event.target)) return;
      event.preventDefault();
      acting.current(action);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  if (props.state.failed) return <p className="rw-empty" role="alert">{t("review.failed")}</p>;
  if (queue === null) return <p className="rw-empty" aria-busy="true">{t("review.loading")}</p>;

  const position = props.item === undefined ? -1 : ids.indexOf(props.item);
  const groupIntent = group === undefined ? undefined : latestIntent(ledger.intents, group.id);
  return (
    <div className={`rw-review${props.item === undefined ? "" : " rw-review--detail"}`}>
      <QaStrip
        meta={queue.meta}
        open={counts.open}
        tally={tally(ledger.decisions)}
        pace={pace(ledger.decisions, new Date())}
        agreement={queue.meta.suggester?.available === true ? agreement(ledger.decisions) : null}
        onAudit={() => void props.source.loadAudit().then((audit) => setSheet({ kind: "audit", items: audit }))}
      />
      <div className="rw-panes">
        <QueuePane
          groups={groups}
          current={props.item}
          decisions={decisions}
          filter={filter}
          onFilter={setFilter}
          systems={systemsOf(items.values())}
          counts={counts}
        />
        {item === undefined ? (
          <div className="rw-detail rw-detail--none">
            <p className="rw-empty">{t(props.item === undefined ? "review.pick" : "review.unknown")}</p>
          </div>
        ) : (
          <div className="rw-detail" key={item.id}>
            <nav className="rw-detail-nav" aria-label={t("review.position")}>
              <Link className="rw-link" to={path(base, "review")} data-testid="back-to-queue"><ChevronLeft size={16} aria-hidden /> {t("review.back")}</Link>
              {position >= 0 && <span className="rw-muted rw-num">{t("review.of", { n: position + 1, total: ids.length })}</span>}
            </nav>
            <EvidencePane item={item} group={group} meta={queue.meta} decisions={decisions} graph={props.graph} onTake={() => act("take")} />
            <DecisionPane
              item={item}
              group={group}
              history={historyOf(ledger.decisions, item.id)}
              intent={groupIntent}
              standing={groupIntent === undefined ? 0 : standing(ledger.decisions, groupIntent.batch).length}
              onGroup={(intent) => setSheet({ kind: "group", intent })}
              onUndo={(batch) => void undo(batch)}
            />
            <DecisionBar decision={decisions.get(item.id)} onAct={act} />
          </div>
        )}
      </div>

      {sheet?.kind === "decide" && item !== undefined && (
        <DecisionSheet
          item={item}
          verdict={sheet.verdict}
          replacing={sheet.replacing}
          onCancel={() => setSheet(null)}
          onConfirm={(extra) => {
            const verdict = sheet.verdict;
            setSheet(null);
            void record(item, verdict, extra);
          }}
        />
      )}
      {sheet?.kind === "group" && sheet.intent === "merge" && group !== undefined && (
        <MergeSheet
          group={group}
          items={items}
          decisions={decisions}
          onCancel={() => setSheet(null)}
          onConfirm={(note) => {
            setSheet(null);
            void decideGroup("merge", note);
          }}
        />
      )}
      {sheet?.kind === "group" && sheet.intent === "split" && group !== undefined && item !== undefined && (
        <SplitSheet
          group={group}
          items={items}
          decisions={decisions}
          current={item}
          onCancel={() => setSheet(null)}
          onConfirm={(parts, note) => {
            setSheet(null);
            void decideGroup("split", note, parts);
          }}
        />
      )}
      {sheet?.kind === "audit" && <AuditSheet audit={sheet.items} onClose={() => setSheet(null)} />}
    </div>
  );
}
