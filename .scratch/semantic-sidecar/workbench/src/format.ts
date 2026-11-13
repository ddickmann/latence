// Small shared formatters.
import type { Translate } from "./host";
import type { Scalar, SourceRecord } from "./types";

export function nameOf(record: SourceRecord): string {
  const name = record["name"];
  if (typeof name === "string" && name !== "") return name;
  const person = [record["first_name"], record["surname"]].filter((v) => typeof v === "string" && v !== "").join(" ");
  return person !== "" ? person : record.unique_id;
}

export function show(value: Scalar | undefined): string {
  return value === null || value === undefined ? "" : String(value);
}

/** A field's label: the message when there is one, otherwise the field name made readable. */
export function fieldLabel(t: Translate, field: string): string {
  const key = `field.${field}`;
  const text = t(key);
  if (text !== key) return text;
  const words = field.replace(/_/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

export function signed(weight: number): string {
  const abs = Math.abs(weight).toFixed(1);
  if (abs === "0.0") return "0.0";
  return `${weight > 0 ? "+" : "−"}${abs}`;
}

export function clock(iso: string, locale: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return new Intl.DateTimeFormat(locale, { dateStyle: "medium", timeStyle: "short" }).format(d);
}

export function path(base: string, ...parts: string[]): string {
  return [base, ...parts.map(encodeURIComponent)].join("/") || "/";
}
