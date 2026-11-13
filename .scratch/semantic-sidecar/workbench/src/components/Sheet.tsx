// A dialog for the actions that need a reason: it takes focus, keeps it, cancels with Esc,
// confirms with Ctrl+Enter (⌘↵) once it may, and gives focus back to what opened it.
import { useEffect, useId, useRef, type ReactNode } from "react";

import { useHost } from "../host";

const FOCUSABLE = 'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

export function Sheet(props: {
  title: string;
  children: ReactNode;
  /** Label of the confirming button; omit for a sheet that only closes. */
  confirm?: string;
  onConfirm?: () => void;
  onCancel: () => void;
  canConfirm?: boolean;
  danger?: boolean;
  wide?: boolean;
  testId?: string;
}) {
  const { t, controls: { Button } } = useHost();
  const titleId = useId();
  const box = useRef<HTMLDivElement>(null);
  const latest = useRef(props);
  latest.current = props;
  const may = props.onConfirm !== undefined && props.canConfirm !== false;

  useEffect(() => {
    const opener = document.activeElement;
    const first = box.current?.querySelector<HTMLElement>("input, textarea, select") ?? box.current?.querySelector<HTMLElement>(FOCUSABLE);
    first?.focus();
    return () => {
      if (opener instanceof HTMLElement) opener.focus();
    };
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const now = latest.current;
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        now.onCancel();
      } else if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
        event.preventDefault();
        event.stopPropagation();
        if (now.onConfirm !== undefined && now.canConfirm !== false) now.onConfirm();
      } else if (event.key === "Tab" && box.current !== null) {
        const stops = [...box.current.querySelectorAll<HTMLElement>(FOCUSABLE)];
        const first = stops[0];
        const last = stops[stops.length - 1];
        if (first === undefined || last === undefined) return;
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
      }
    };
    // Capture, so a host's own Esc and letter keys never see what the sheet has taken.
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, []);

  return (
    <div className="rw-sheet-veil" onMouseDown={(e) => e.target === e.currentTarget && props.onCancel()}>
      <div ref={box} className={`rw-sheet${props.wide === true ? " rw-sheet--wide" : ""}`} role="dialog" aria-modal="true" aria-labelledby={titleId} data-testid={props.testId}>
        <h2 id={titleId} className="rw-sheet-title">{props.title}</h2>
        <div className="rw-sheet-body">{props.children}</div>
        <div className="rw-sheet-actions">
          <Button kind="quiet" onClick={props.onCancel} kbd="Esc" testId="sheet-cancel">{props.onConfirm === undefined ? t("sheet.close") : t("sheet.cancel")}</Button>
          {props.onConfirm !== undefined && (
            <Button kind={props.danger === true ? "danger" : "primary"} disabled={!may} onClick={props.onConfirm} kbd="⌘↵" testId="sheet-confirm">{props.confirm}</Button>
          )}
        </div>
      </div>
    </div>
  );
}
