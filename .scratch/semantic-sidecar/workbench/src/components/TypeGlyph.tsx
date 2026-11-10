// A node type's mark: a shape with an icon, so the type never rests on colour alone.
import { DataBase, Earth, Enterprise, Industry, Location } from "@carbon/icons-react";

import type { NodeType } from "../types";

const ICON = { organisation: Enterprise, city: Location, country: Earth, industry: Industry, source_system: DataBase } as const;

export function TypeGlyph({ type, size = "md" }: { type: NodeType; size?: "sm" | "md" | "lg" }) {
  const Icon = ICON[type];
  return (
    <span className={`rw-glyph rw-glyph--${type} rw-glyph--${size}`} aria-hidden="true">
      <Icon size={size === "lg" ? 24 : 16} />
    </span>
  );
}
