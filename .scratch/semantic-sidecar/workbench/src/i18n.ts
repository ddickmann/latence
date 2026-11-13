// English strings, in i18next's `{{name}}` form so a host can merge messages/en.json into its
// own catalogue and pass its translate function instead of this one.
import en from "./messages/en.json";
import type { Translate } from "./host";

type Tree = { [key: string]: string | Tree };

function lookup(tree: Tree, key: string): string | undefined {
  let at: string | Tree | undefined = tree;
  for (const part of key.split(".")) {
    if (typeof at !== "object") return undefined;
    at = at[part];
  }
  return typeof at === "string" ? at : undefined;
}

export function translator(messages: Tree = en as Tree): Translate {
  return (key, vars = {}) => {
    const count = vars["count"];
    const suffix = typeof count === "number" ? (count === 1 ? "_one" : "_other") : "";
    const text = (suffix === "" ? undefined : lookup(messages, key + suffix)) ?? lookup(messages, key) ?? key;
    return text.replace(/\{\{(\w+)\}\}/g, (_m, name: string) => String(vars[name] ?? ""));
  };
}

export const messages = en;
