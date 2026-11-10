// One key per action. Pure, so the map is tested without a DOM and a host can show it.
export type KeyAction = "same" | "different" | "unsure" | "take" | "skip" | "merge" | "split" | "next" | "previous" | "close";

export interface KeyLike {
  key: string;
  metaKey?: boolean;
  ctrlKey?: boolean;
  altKey?: boolean;
  /** True when the event came from a field a person types in. */
  typing?: boolean;
}

export const KEYS: Record<KeyAction, string> = {
  same: "A",
  different: "R",
  unsure: "U",
  take: "Y",
  skip: "Space",
  merge: "M",
  split: "S",
  next: "J",
  previous: "K",
  close: "Esc",
};

/** The action a key press means, or null. Keys with a modifier, and keys typed in a field, mean nothing. */
export function actionFor(event: KeyLike): KeyAction | null {
  if (event.metaKey || event.ctrlKey || event.altKey) return null;
  if (event.key === "Escape") return "close";
  if (event.typing) return null;
  switch (event.key.toLowerCase()) {
    case "a": return "same";
    case "r": return "different";
    case "u": return "unsure";
    case "y": return "take";
    case " ": return "skip";
    case "m": return "merge";
    case "s": return "split";
    case "j": return "next";
    case "k": return "previous";
    default: return null;
  }
}

export function isTyping(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
}

/** True when the press belongs to a button or link the person is on: Space there is a click. */
export function onControl(target: EventTarget | null): boolean {
  return target instanceof HTMLElement && ["BUTTON", "A", "SUMMARY"].includes(target.tagName);
}
