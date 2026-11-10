// The standalone app: the workbench in a minimal shell of its own, over the synthetic sample.
// It exercises the same seams a host uses: the chrome (breadcrumb), the router, the data sources.
import { StrictMode, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { HashRouter, MemoryRouter, Navigate, Route, Routes, useLocation } from "react-router";

import { ResolutionWorkbench, StaticGraphSource, StaticReviewSource, browserStore, useWindowListKeys, type Crumb, type GraphExport, type ReviewExport } from "../index";
import graphSample from "./sample/graph-export.json";
import reviewSample from "./sample/review-export.json";
import "./standalone.scss";

declare const __RW_ROUTER__: "hash" | "memory";

type Theme = "light" | "dark";

function storedTheme(): Theme | null {
  try {
    const v = window.localStorage.getItem("rw.theme");
    return v === "light" || v === "dark" ? v : null;
  } catch {
    return null;
  }
}

// The shell's breadcrumb, fed by the workbench through the chrome seam.
let showCrumbs: (crumbs: Crumb[]) => void = () => undefined;
function useShellCrumbs(crumbs: Crumb[]): void {
  const key = JSON.stringify(crumbs);
  useEffect(() => {
    showCrumbs(crumbs);
    // the crumbs' content, not their identity
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
}

function App() {
  const [theme, setTheme] = useState<Theme | null>(storedTheme);
  const [crumbs, setCrumbs] = useState<Crumb[]>([]);
  showCrumbs = setCrumbs;

  useEffect(() => {
    if (theme === null) return;
    document.documentElement.dataset["theme"] = theme;
    try {
      window.localStorage.setItem("rw.theme", theme);
    } catch {
      // the choice lasts for this visit
    }
  }, [theme]);

  const reviewSource = useMemo(
    () => new StaticReviewSource(reviewSample as unknown as ReviewExport, { actor: { id: "reviewer-1", name: "Demo reviewer" }, store: browserStore() }),
    [],
  );
  const graphSource = useMemo(() => new StaticGraphSource(graphSample as unknown as GraphExport), []);
  const chrome = useMemo(() => ({ useCrumbs: useShellCrumbs, useListKeys: useWindowListKeys }), []);
  const dark = theme === "dark" || (theme === null && typeof window.matchMedia === "function" && window.matchMedia("(prefers-color-scheme: dark)").matches);

  return (
    <div className="sa-shell">
      <header className="sa-header">
        <span className="sa-name">Resolution workbench</span>
        <span className="sa-crumbs">{crumbs.map((c) => c.label).join("  /  ")}</span>
        <button type="button" className="sa-theme" onClick={() => setTheme(dark ? "light" : "dark")} data-testid="theme-toggle">
          {dark ? "Light" : "Dark"}
        </button>
      </header>
      <Routes>
        <Route path="/resolution/*" element={<ResolutionWorkbench reviewSource={reviewSource} graphSource={graphSource} chrome={chrome} />} />
        <Route path="*" element={<Navigate to="/resolution/review" replace />} />
      </Routes>
    </div>
  );
}

// A page that may not use the address bar keeps its place in memory, and remembers it for a reload.
const PLACE = "rw.place";
function storedPlace(): string {
  try {
    const v = window.localStorage.getItem(PLACE);
    return v !== null && v.startsWith("/resolution") ? v : "/resolution/review";
  } catch {
    return "/resolution/review";
  }
}
function RememberPlace() {
  const { pathname } = useLocation();
  useEffect(() => {
    try {
      window.localStorage.setItem(PLACE, pathname);
    } catch {
      // the place lasts for this visit
    }
  }, [pathname]);
  return null;
}

const mount = document.getElementById("root");
if (mount !== null) {
  createRoot(mount).render(
    <StrictMode>
      {__RW_ROUTER__ === "memory" ? (
        <MemoryRouter initialEntries={[storedPlace()]}>
          <RememberPlace />
          <App />
        </MemoryRouter>
      ) : (
        <HashRouter>
          <App />
        </HashRouter>
      )}
    </StrictMode>,
  );
}
