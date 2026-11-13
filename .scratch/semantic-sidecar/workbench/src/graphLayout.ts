// Where things go in the two drawings. Pure geometry, so it is tested without a browser.
import type { Concept, NeighbourGroup, NodeRef, NodeType, Relation } from "./types";

export interface Point {
  x: number;
  y: number;
}

// ---------- The schema map: concepts in layers, relations flowing one way ----------

export interface SchemaLayout {
  width: number;
  height: number;
  horizontal: boolean;
  box: { w: number; h: number };
  nodes: Map<NodeType, Point>;
  edges: { relation: Relation; path: string; mid: Point }[];
}

/** A concept's layer: the longest chain of relations that leads to it. */
export function layers(concepts: readonly Concept[], relations: readonly Relation[]): Map<NodeType, number> {
  const out = new Map<NodeType, number>(concepts.map((c) => [c.id, 0]));
  for (let pass = 0; pass < concepts.length; pass += 1) {
    for (const r of relations) {
      const from = out.get(r.from);
      const to = out.get(r.to);
      if (from !== undefined && to !== undefined && to < from + 1) out.set(r.to, from + 1);
    }
  }
  return out;
}

export function layoutSchema(concepts: readonly Concept[], relations: readonly Relation[], width: number): SchemaLayout {
  const horizontal = width >= 640;
  const layer = layers(concepts, relations);
  const depth = Math.max(0, ...layer.values()) + 1;
  const rows: Concept[][] = Array.from({ length: depth }, () => []);
  for (const c of concepts) rows[layer.get(c.id) ?? 0]?.push(c);
  const widest = Math.max(1, ...rows.map((r) => r.length));

  // `across` runs along a layer, `along` from layer to layer.
  const box = horizontal ? { w: 168, h: 64 } : { w: Math.min(136, Math.floor((width - 8 * (widest - 1)) / widest)), h: 92 };
  const acrossSize = horizontal ? box.h : box.w;
  const alongSize = horizontal ? box.w : box.h;
  const alongGap = horizontal ? Math.max(96, (width - depth * box.w) / Math.max(1, depth - 1)) : 84;
  const acrossGap = horizontal ? 28 : 8;
  const slot = acrossSize + acrossGap;
  const acrossSpan = horizontal ? widest * slot - acrossGap : width;

  const across = new Map<NodeType, number>();
  rows.forEach((row, i) => {
    row.forEach((c, j) => {
      const parents = relations.filter((r) => r.to === c.id && (layer.get(r.from) ?? 0) < i).map((r) => across.get(r.from)).filter((v): v is number => v !== undefined);
      const even = (acrossSpan - (row.length * slot - acrossGap)) / 2 + acrossSize / 2 + slot * j;
      // A layer narrower than the widest hangs its nodes under what they come from.
      across.set(c.id, row.length < widest && parents.length > 0 ? parents.reduce((a, b) => a + b, 0) / parents.length : even);
    });
    // Keep neighbours in a layer from overlapping after the pull towards their parents.
    const sorted = [...row].sort((a, b) => (across.get(a.id) ?? 0) - (across.get(b.id) ?? 0));
    for (let j = 1; j < sorted.length; j += 1) {
      const prev = across.get((sorted[j - 1] as Concept).id) ?? 0;
      const here = across.get((sorted[j] as Concept).id) ?? 0;
      if (here - prev < slot) across.set((sorted[j] as Concept).id, prev + slot);
    }
  });

  const nodes = new Map<NodeType, Point>();
  for (const c of concepts) {
    const a = across.get(c.id) ?? 0;
    const l = (layer.get(c.id) ?? 0) * (alongSize + alongGap) + alongSize / 2;
    nodes.set(c.id, horizontal ? { x: l, y: a } : { x: a, y: l });
  }

  const edges = relations.flatMap((relation) => {
    const from = nodes.get(relation.from);
    const to = nodes.get(relation.to);
    if (from === undefined || to === undefined) return [];
    const a = horizontal ? { x: from.x + box.w / 2, y: from.y } : { x: from.x, y: from.y + box.h / 2 };
    const b = horizontal ? { x: to.x - box.w / 2 - 6, y: to.y } : { x: to.x, y: to.y - box.h / 2 - 6 };
    const c1 = horizontal ? { x: (a.x + b.x) / 2, y: a.y } : { x: a.x, y: (a.y + b.y) / 2 };
    const c2 = horizontal ? { x: (a.x + b.x) / 2, y: b.y } : { x: b.x, y: (a.y + b.y) / 2 };
    // The label sits nearer the target, where the edges of one fan have spread apart.
    const s = 0.62;
    const mid = {
      x: (1 - s) ** 3 * a.x + 3 * (1 - s) ** 2 * s * c1.x + 3 * (1 - s) * s ** 2 * c2.x + s ** 3 * b.x,
      y: (1 - s) ** 3 * a.y + 3 * (1 - s) ** 2 * s * c1.y + 3 * (1 - s) * s ** 2 * c2.y + s ** 3 * b.y,
    };
    return [{ relation, path: `M${a.x},${a.y} C${c1.x},${c1.y} ${c2.x},${c2.y} ${b.x},${b.y}`, mid }];
  });

  const alongTotal = depth * alongSize + (depth - 1) * alongGap;
  return { width: horizontal ? alongTotal : width, height: horizontal ? acrossSpan : alongTotal, horizontal, box, nodes, edges };
}

// ---------- The neighbourhood: one node in the middle, its neighbours around it ----------

export interface Spoke {
  group: NeighbourGroup;
  node: NodeRef & { items?: string[]; probability?: number; pairs?: number };
  at: Point;
}

export interface FocusLayout {
  width: number;
  height: number;
  narrow: boolean;
  centre: Point;
  spokes: Spoke[];
  /**
   * One per group: the point its neighbours branch from, which is where the relation's label
   * sits, and how many of its neighbours are not drawn.
   */
  hubs: { group: NeighbourGroup; at: Point; hidden: number }[];
}

export function layoutFocus(groups: readonly NeighbourGroup[], width: number): FocusLayout {
  const narrow = width < 560;
  const perGroup = narrow ? 3 : 7;
  const budget = narrow ? 8 : 16;
  // A phone is narrow and tall, so the ring is too: neighbours spread down the screen.
  const height = narrow ? 440 : 400;
  const centre = { x: width / 2, y: height / 2 };
  const rx = Math.min(340, Math.max(60, width / 2 - (narrow ? 56 : 96)));
  const ry = height / 2 - (narrow ? 60 : 56);

  // Every group gets at least one node; the rest of the budget goes round in order.
  const take = groups.map(() => 0);
  let left = budget;
  for (let round = 0; round < perGroup && left > 0; round += 1) {
    groups.forEach((g, i) => {
      if (left > 0 && (take[i] ?? 0) < Math.min(perGroup, g.nodes.length) && (take[i] ?? 0) === round) {
        take[i] = round + 1;
        left -= 1;
      }
    });
  }
  const shown = groups.map((g, i) => g.nodes.slice(0, take[i] ?? 0));
  const count = shown.reduce((n, s) => n + s.length, 0);
  const gaps = groups.length > 1 ? groups.length : 0;
  const slots = count + gaps * 0.6;
  const start = count <= 2 ? 0 : -Math.PI / 2;

  // How far out a group's hub sits: further on a phone, to clear the node in the middle.
  const reach = narrow ? 0.62 : 0.5;
  const spokes: Spoke[] = [];
  const hubs: FocusLayout["hubs"] = [];
  let slot = 0;
  groups.forEach((group, i) => {
    const nodes = shown[i] ?? [];
    const angles: number[] = [];
    for (const node of nodes) {
      const angle = start + (2 * Math.PI * (slot + 0.5)) / Math.max(1, slots);
      angles.push(angle);
      spokes.push({ group, node, at: { x: centre.x + rx * Math.cos(angle), y: centre.y + ry * Math.sin(angle) } });
      slot += 1;
    }
    slot += gaps > 0 ? 0.6 : 0;
    const mid = angles.length === 0 ? start : (angles[0] as number) + ((angles[angles.length - 1] as number) - (angles[0] as number)) / 2;
    hubs.push({
      group,
      at: { x: centre.x + rx * reach * Math.cos(mid), y: centre.y + ry * reach * Math.sin(mid) },
      hidden: group.total - nodes.length,
    });
  });
  return { width, height, narrow, centre, spokes, hubs };
}
