// The seam to wherever review items and decisions live. The workbench only ever talks to these
// two interfaces; the static implementations below serve an export file and keep decisions in
// the browser, which is all a prototype needs. A deployment puts its own service behind them.
import type {
  Actor, Decision, DecisionInput, GraphEntity, GraphExport, GraphReferenceNode, GraphSchema, GroupIntent,
  Ledger, NeighbourGroup, NodeDetail, NodeRef, NodeType, ReviewExport, ReviewGroup, ReviewItem, ReviewMeta,
} from "./types";

export interface Queue {
  meta: ReviewMeta;
  groups: ReviewGroup[];
  items: ReviewItem[];
}

export interface ReviewDataSource {
  /** The review queue: groups in priority order and their items. */
  loadQueue(): Promise<Queue>;
  /** Every decision and group intent recorded so far, oldest first. */
  loadLedger(): Promise<Ledger>;
  /** Append a decision. Deciding an item again supersedes; nothing is ever edited. */
  decide(input: DecisionInput): Promise<Decision>;
  /** Record what a reviewer says about a whole group. */
  proposeGroup(input: { group: string; intent: "merge" | "split"; note?: string }): Promise<GroupIntent>;
  /** A sample of pairs the machine accepted on its own, for a reviewer to audit. */
  loadAudit(): Promise<ReviewItem[]>;
}

export interface GraphDataSource {
  loadSchema(): Promise<GraphSchema>;
  /** A node with its neighbours grouped by relation, or undefined when it is not known. */
  loadNode(id: string): Promise<NodeDetail | undefined>;
  /** Nodes whose name contains the query, best first. */
  search(query: string, limit: number): Promise<NodeRef[]>;
  /** The instances of a concept, largest first. */
  listInstances(concept: NodeType, limit: number): Promise<{ nodes: NodeRef[]; total: number }>;
  /** The organisation a source record supports, if this source carries it. */
  entityOfRecord(recordId: string): Promise<NodeRef | undefined>;
}

/** The part of Storage the static source uses; `window.localStorage` fits. */
export interface KeyValueStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export function memoryStore(): KeyValueStore {
  const map = new Map<string, string>();
  return { getItem: (k) => map.get(k) ?? null, setItem: (k, v) => void map.set(k, v) };
}

/** localStorage when the browser allows it, otherwise memory: private windows may refuse. */
export function browserStore(): KeyValueStore {
  const fallback = memoryStore();
  return {
    getItem(key) {
      try {
        return window.localStorage.getItem(key) ?? fallback.getItem(key);
      } catch {
        return fallback.getItem(key);
      }
    },
    setItem(key, value) {
      fallback.setItem(key, value);
      try {
        window.localStorage.setItem(key, value);
      } catch {
        // kept in memory for this visit
      }
    },
  };
}

export interface StaticOptions {
  actor: Actor;
  store?: KeyValueStore;
  now?: () => Date;
  /** Storage key; change it to keep two exports' decisions apart. */
  key?: string;
}

/** An export file as the queue, with the decision log kept in a key-value store. */
export class StaticReviewSource implements ReviewDataSource {
  private readonly store: KeyValueStore;
  private readonly now: () => Date;
  private readonly key: string;

  constructor(private readonly data: ReviewExport, private readonly options: StaticOptions) {
    this.store = options.store ?? memoryStore();
    this.now = options.now ?? (() => new Date());
    this.key = options.key ?? "rw.ledger.v1";
  }

  private read(): Ledger {
    try {
      const parsed = JSON.parse(this.store.getItem(this.key) ?? "null") as Partial<Ledger> | null;
      const known = new Set(this.data.items.map((i) => i.id));
      return {
        // A log written against another export says nothing about this one.
        decisions: (parsed?.decisions ?? []).filter((d) => known.has(d.item)),
        intents: (parsed?.intents ?? []).filter((i) => this.data.groups.some((g) => g.id === i.group)),
      };
    } catch {
      return { decisions: [], intents: [] };
    }
  }

  private write(ledger: Ledger): void {
    this.store.setItem(this.key, JSON.stringify(ledger));
  }

  loadQueue(): Promise<Queue> {
    return Promise.resolve({ meta: this.data.meta, groups: this.data.groups, items: this.data.items });
  }

  loadLedger(): Promise<Ledger> {
    return Promise.resolve(this.read());
  }

  decide(input: DecisionInput): Promise<Decision> {
    if (!this.data.items.some((i) => i.id === input.item)) return Promise.reject(new Error(`unknown item ${input.item}`));
    const decision: Decision = { ...input, actor: this.options.actor, at: this.now().toISOString() };
    const ledger = this.read();
    this.write({ ...ledger, decisions: [...ledger.decisions, decision] });
    return Promise.resolve(decision);
  }

  proposeGroup(input: { group: string; intent: "merge" | "split"; note?: string }): Promise<GroupIntent> {
    const intent: GroupIntent = { ...input, actor: this.options.actor, at: this.now().toISOString() };
    const ledger = this.read();
    this.write({ ...ledger, intents: [...ledger.intents, intent] });
    return Promise.resolve(intent);
  }

  loadAudit(): Promise<ReviewItem[]> {
    return Promise.resolve(this.data.audit);
  }
}

const SEARCH_ORDER: NodeType[] = ["organisation", "city", "industry", "country", "source_system"];

/** A graph export served from memory. */
export class StaticGraphSource implements GraphDataSource {
  private readonly entities = new Map<string, GraphEntity>();
  private readonly references = new Map<string, GraphReferenceNode>();
  private readonly ofRecord = new Map<string, string>();

  constructor(private readonly data: GraphExport) {
    for (const e of data.entities) {
      this.entities.set(e.id, e);
      for (const r of e.records) this.ofRecord.set(r.unique_id, e.id);
    }
    for (const n of data.nodes) this.references.set(n.id, n);
  }

  private entityRef(e: GraphEntity): NodeRef {
    return { id: e.id, type: "organisation", label: e.label, size: e.records.length };
  }

  private referenceRef(n: GraphReferenceNode): NodeRef {
    return { id: n.id, type: n.type, label: n.label, size: n.organisations ?? n.records ?? 0 };
  }

  private ref(id: string): NodeRef | undefined {
    const e = this.entities.get(id);
    if (e !== undefined) return this.entityRef(e);
    const n = this.references.get(id);
    return n === undefined ? undefined : this.referenceRef(n);
  }

  private refs(ids: (string | null | undefined)[]): NodeRef[] {
    return ids.map((id) => (id == null ? undefined : this.ref(id))).filter((r): r is NodeRef => r !== undefined);
  }

  loadSchema(): Promise<GraphSchema> {
    const { meta, concepts, relations, attributes } = this.data;
    return Promise.resolve({ meta, concepts, relations, attributes });
  }

  loadNode(id: string): Promise<NodeDetail | undefined> {
    const e = this.entities.get(id);
    if (e !== undefined) return Promise.resolve(this.entityDetail(e));
    const n = this.references.get(id);
    return Promise.resolve(n === undefined ? undefined : this.referenceDetail(n));
  }

  private entityDetail(e: GraphEntity): NodeDetail {
    const groups: NeighbourGroup[] = [];
    const one = (relation: string, label: string, type: NodeType, ids: (string | null)[]) => {
      const nodes = this.refs(ids);
      if (nodes.length > 0) groups.push({ relation, label, type, status: "asserted", nodes, total: nodes.length });
    };
    one("located_in", "located in", "city", [e.city === null ? null : `city:${e.city}`]);
    one("operates_in", "operates in", "industry", [e.industry === null ? null : `industry:${e.industry}`]);
    const systems = e.systems.map((s) => ({
      id: `system:${s}`, type: "source_system" as const, label: s,
      size: e.records.filter((r) => r["source_system"] === s).length,
    }));
    if (systems.length > 0) groups.push({ relation: "recorded_in", label: "recorded in", type: "source_system", status: "asserted", nodes: systems, total: systems.length });
    const maybe = e.possibly_same.flatMap((link) => {
      const other = this.entities.get(link.entity);
      return other === undefined ? [] : [{ ...this.entityRef(other), items: link.items, probability: link.probability, pairs: link.pairs }];
    });
    if (maybe.length > 0) groups.push({ relation: "possibly_same", label: "might be the same as", type: "organisation", status: "open", nodes: maybe, total: maybe.length });
    return { node: this.entityRef(e), groups, records: e.records, values: e.values };
  }

  private referenceDetail(n: GraphReferenceNode): NodeDetail {
    const groups: NeighbourGroup[] = [];
    const value = n.type === "industry" ? n.code ?? n.label : n.label;
    const holds = (e: GraphEntity): boolean =>
      n.type === "city" ? e.city === value
      : n.type === "country" ? e.country === value
      : n.type === "industry" ? e.industry === value
      : e.systems.includes(value);
    if (n.type === "city" && n.country != null) {
      const up = this.refs([`country:${n.country}`]);
      if (up.length > 0) groups.push({ relation: "in_country", label: "lies in", type: "country", status: "asserted", nodes: up, total: 1 });
    }
    if (n.type === "country") {
      const cities = this.refs((n.cities ?? []).map((c) => `city:${c}`)).sort((a, b) => b.size - a.size);
      if (cities.length > 0) groups.push({ relation: "has_city", label: "has cities", type: "city", status: "asserted", nodes: cities, total: cities.length });
    }
    const orgs = this.data.entities.filter(holds).map((e) => this.entityRef(e)).sort((a, b) => b.size - a.size || a.label.localeCompare(b.label));
    const total = n.organisations ?? orgs.length;
    if (orgs.length > 0) {
      groups.push({
        relation: "has_organisation",
        label: n.type === "source_system" ? "holds records of" : n.type === "industry" ? "is the industry of" : "is the location of",
        type: "organisation", status: "asserted", nodes: orgs, total: n.type === "source_system" ? orgs.length : total,
      });
    }
    return { node: this.referenceRef(n), groups, total, listed: orgs.length };
  }

  search(query: string, limit: number): Promise<NodeRef[]> {
    const q = query.trim().toLowerCase();
    if (q === "") return Promise.resolve([]);
    const all = [...this.data.entities.map((e) => this.entityRef(e)), ...this.data.nodes.map((n) => this.referenceRef(n))];
    const hits = all
      .filter((n) => n.label.toLowerCase().includes(q))
      .sort((a, b) =>
        Number(b.label.toLowerCase().startsWith(q)) - Number(a.label.toLowerCase().startsWith(q))
        || SEARCH_ORDER.indexOf(a.type) - SEARCH_ORDER.indexOf(b.type)
        || b.size - a.size
        || a.label.localeCompare(b.label));
    return Promise.resolve(hits.slice(0, limit));
  }

  listInstances(concept: NodeType, limit: number): Promise<{ nodes: NodeRef[]; total: number }> {
    const total = this.data.concepts.find((c) => c.id === concept)?.count ?? 0;
    const nodes = concept === "organisation"
      ? this.data.entities.map((e) => this.entityRef(e))
      : this.data.nodes.filter((n) => n.type === concept).map((n) => this.referenceRef(n));
    nodes.sort((a, b) => b.size - a.size || a.label.localeCompare(b.label));
    return Promise.resolve({ nodes: nodes.slice(0, limit), total });
  }

  entityOfRecord(recordId: string): Promise<NodeRef | undefined> {
    const id = this.ofRecord.get(recordId);
    return Promise.resolve(id === undefined ? undefined : this.ref(id));
  }
}
