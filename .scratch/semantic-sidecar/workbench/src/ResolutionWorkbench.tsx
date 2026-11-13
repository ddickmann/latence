// The component a host mounts. It reads where it is from the router it is mounted in, so the
// open pair or node is part of the URL and the host's own links and back button work.
//
//   <Route path="/resolution/*" element={<ResolutionWorkbench reviewSource={...} />} />
import "./styles/workbench.scss";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useLocation, useParams } from "react-router";

import { useNoCrumbs, useWindowListKeys } from "./chrome";
import { GraphScreen } from "./components/GraphScreen";
import { ReviewScreen } from "./components/ReviewScreen";
import type { GraphDataSource, ReviewDataSource } from "./dataSource";
import { path } from "./format";
import { HostContext, type Controls, type Host, type HostChrome, type Notice, type Translate } from "./host";
import { translator } from "./i18n";
import { inForce } from "./ledger";
import { carbonControls } from "./ui/carbon";
import { useReview } from "./useReview";

export interface ResolutionWorkbenchProps {
  /** Where review items come from and decisions go. */
  reviewSource: ReviewDataSource;
  /** The resolved graph. Without it the workbench is the review screen alone. */
  graphSource?: GraphDataSource;
  /** The host's breadcrumb, list keys, counters and notices. Missing parts fall back to the workbench's own. */
  chrome?: Partial<HostChrome>;
  /** The host's own Button, TextField, TextArea, Select and Checkbox. Missing ones are Carbon's. */
  controls?: Partial<Controls>;
  /** The host's translate function, once it has merged messages/en.json under a prefix of its choice. */
  translate?: Translate;
  /** Locale for numbers and times. */
  locale?: string;
}

const builtinTranslate = translator();

function decode(segment: string): string {
  try {
    return decodeURIComponent(segment);
  } catch {
    return segment;
  }
}

export function ResolutionWorkbench(props: ResolutionWorkbenchProps) {
  const { pathname } = useLocation();
  const splat = useParams()["*"] ?? "";
  const depth = splat.split("/").filter(Boolean).length;
  const segments = pathname.split("/").filter(Boolean);
  const mount = segments.slice(0, segments.length - depth).join("/");
  const base = mount === "" ? "" : `/${mount}`;
  const rest = segments.slice(segments.length - depth).map(decode);
  const view = rest[0] === "graph" && props.graphSource !== undefined ? "graph" : "review";
  const target = rest[0] === "graph" || rest[0] === "review" ? rest.slice(1).join("/") || undefined : undefined;

  const [notices, setNotices] = useState<(Notice & { key: number })[]>([]);
  const counter = useRef(0);
  const notify = useCallback((notice: Notice) => {
    counter.current += 1;
    const key = counter.current;
    setNotices((all) => [...all.slice(-2), { ...notice, key }]);
    // A notice that offers an undo stays long enough to be used.
    window.setTimeout(() => setNotices((all) => all.filter((n) => n.key !== key)), notice.action === undefined ? 5000 : 10000);
  }, []);

  const t = props.translate ?? builtinTranslate;
  const host = useMemo<Host>(() => ({
    chrome: { useCrumbs: useNoCrumbs, useListKeys: useWindowListKeys, recount: () => undefined, notify, ...props.chrome },
    controls: { ...carbonControls, ...props.controls },
    t,
    base,
    locale: props.locale ?? "en",
  }), [props.chrome, props.controls, t, base, props.locale, notify]);

  const review = useReview(props.reviewSource);
  const open = review.queue === null ? null : review.queue.items.length - inForce(review.ledger.decisions).size;

  // A new screen starts at its top: on a phone the queue and the pair are separate screens.
  const root = useRef<HTMLDivElement>(null);
  const first = useRef(true);
  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    const el = root.current;
    if (el !== null && typeof el.scrollIntoView === "function" && el.getBoundingClientRect().top < 0) el.scrollIntoView({ block: "start" });
  }, [view, target]);

  return (
    <HostContext.Provider value={host}>
      <div className="rw-root" ref={root} data-testid="resolution-workbench">
        <div className="rw-top">
          {props.graphSource !== undefined && (
            <nav className="rw-tabs" aria-label={t("workbench.label")}>
              <Link to={path(base, "review")} className="rw-tab" aria-current={view === "review" ? "page" : undefined} data-testid="tab-review">
                {t("tabs.review")}{open !== null && <span className="rw-num rw-tab-count">{open}</span>}
              </Link>
              <Link to={path(base, "graph")} className="rw-tab" aria-current={view === "graph" ? "page" : undefined} data-testid="tab-graph">
                {t("tabs.graph")}
              </Link>
            </nav>
          )}
          {review.queue?.meta.synthetic === true && <span className="rw-tag">{t("tabs.synthetic")}</span>}
        </div>
        <div className="rw-body">
          {view === "graph" && props.graphSource !== undefined
            ? <GraphScreen source={props.graphSource} review={review} node={target} />
            : <ReviewScreen state={review} source={props.reviewSource} graph={props.graphSource} item={target} />}
        </div>
        {props.chrome?.notify === undefined && (
          <div className="rw-notices" role="status" aria-live="polite">
            {notices.map((n) => (
              <div key={n.key} className={`rw-notice rw-notice--${n.tone}`} data-testid="notice">
                <b>{n.title}</b>
                {n.detail !== undefined && <span>{n.detail}</span>}
                {n.action !== undefined && (
                  <button
                    type="button"
                    className="rw-notice-action"
                    data-testid="notice-action"
                    onClick={() => {
                      n.action?.run();
                      setNotices((all) => all.filter((x) => x.key !== n.key));
                    }}
                  >
                    {n.action.label}
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </HostContext.Provider>
  );
}
