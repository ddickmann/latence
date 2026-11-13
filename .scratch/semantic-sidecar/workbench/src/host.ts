// The seams to the application around the workbench. Each has a default, so the workbench runs
// alone; a host replaces the ones it has its own answer for. Nothing here imports a host.
import { createContext, useContext, type ComponentType, type ReactNode } from "react";

export interface Crumb {
  label: string;
  /** An absolute path in the host's router. */
  to?: string;
}

export interface ListKeys {
  next?: () => void;
  previous?: () => void;
  close?: () => void;
}

export interface Notice {
  tone: "positive" | "caution" | "critical" | "info";
  title: string;
  detail?: string;
  /** At most one thing a person can do about it, such as undo. */
  action?: { label: string; run: () => void };
}

/**
 * What the workbench tells the application around it. `useCrumbs` and `useListKeys` are called
 * as hooks on every render, so a host passes its own hooks straight through; they must be the
 * same functions for the life of the mounted workbench.
 */
export interface HostChrome {
  /** The breadcrumb of the screen being shown. */
  useCrumbs(crumbs: Crumb[]): void;
  /** What J, K and Esc mean on this screen. The host owns the keys and calls these. */
  useListKeys(keys: ListKeys): void;
  /** Something the host's navigation counts has changed (a decision was recorded). */
  recount(): void;
  /** A short confirmation of a finished action. */
  notify(notice: Notice): void;
}

// ---------- The five controls the workbench takes from the host's design system ----------

export interface ButtonProps {
  kind?: "primary" | "secondary" | "quiet" | "danger";
  size?: "sm" | "md" | "lg";
  /** A keyboard hint shown on the button. */
  kbd?: string;
  children?: ReactNode;
  disabled?: boolean;
  onClick?: () => void;
  testId?: string;
  "aria-pressed"?: boolean;
  "aria-label"?: string;
}

export interface TextFieldProps {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}

export interface TextAreaProps {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  rows?: number;
}

export interface SelectProps {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: { value: string; text: string }[];
}

export interface CheckboxProps {
  id: string;
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}

export interface Controls {
  Button: ComponentType<ButtonProps>;
  TextField: ComponentType<TextFieldProps>;
  TextArea: ComponentType<TextAreaProps>;
  Select: ComponentType<SelectProps>;
  Checkbox: ComponentType<CheckboxProps>;
}

export type Translate = (key: string, vars?: Record<string, string | number>) => string;

export interface Host {
  chrome: HostChrome;
  controls: Controls;
  t: Translate;
  /** The absolute path the workbench is mounted at, without a trailing slash. */
  base: string;
  /** Locale for numbers and times. */
  locale: string;
}

export const HostContext = createContext<Host | null>(null);

export function useHost(): Host {
  const host = useContext(HostContext);
  if (host === null) throw new Error("useHost outside <ResolutionWorkbench>");
  return host;
}
