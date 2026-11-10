// What the workbench does for itself when no host does it: its own J/K/Esc, and its own notices.
import { useEffect, useRef } from "react";

import type { Crumb, ListKeys } from "./host";
import { actionFor, isTyping } from "./keys";

/** J, K and Esc on the window. A host with its own key handling passes its hook instead. */
export function useWindowListKeys(keys: ListKeys): void {
  const latest = useRef(keys);
  latest.current = keys;
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;
      const action = actionFor({ key: event.key, metaKey: event.metaKey, ctrlKey: event.ctrlKey, altKey: event.altKey, typing: isTyping(event.target) });
      const run = action === "next" ? latest.current.next : action === "previous" ? latest.current.previous : action === "close" ? latest.current.close : undefined;
      if (run !== undefined) {
        event.preventDefault();
        run();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}

export function useNoCrumbs(_crumbs: Crumb[]): void {
  // no breadcrumb of its own: the tabs say where a person is
}
