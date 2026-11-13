"""Induce the schema level of an estate: concepts, relations, attributes and measure bindings.

Signals vote, strongest first (ADR-0073). What a metric view or a verified join states is
recorded as ``declared``. What is only mined from the data is ``proposed``, with its support
counted on the data: how many rows hold the rule and how many break it. A mined rule under the
support floor is dropped. Nothing here is approved: every element is a proposal for a person.

Tier 0, metadata: metric-view sources, joins, dimensions and measures; verified joins handed
in by a host; names and comments. Tier 1, aggregates computed inside the warehouse: uniqueness,
inclusion, functional dependency, distinct counts. A column marked sensitive is never queried.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any

from .codes import say
from .metric_views import join_columns
from .reader import CatalogObject, MetricView, Reader

SUPPORT_FLOOR = 0.9  # a mined inclusion that holds for fewer rows than this is dropped
REFERENCE_MAX_VALUES = 400  # a shared column with more distinct values is not a value set
REFERENCE_MIN_TABLES = 3


@dataclass
class Element:
    id: str
    label: str
    status: str  # "declared" | "proposed"
    signals: list[dict[str, Any]] = field(default_factory=list)
    body: dict[str, Any] = field(default_factory=dict)


# Plurals the rules below get wrong, and words that end in "s" and are one thing.
SINGULAR = {"statuses": "status", "buses": "bus", "analyses": "analysis", "series": "series", "data": "data", "news": "news"}


def singular(word: str) -> str:
    """One of what a table name names. English table names only: "cases" is a case, "addresses"
    an address. A word the rules get wrong is given a label in the estate file."""
    if word in SINGULAR:
        return SINGULAR[word]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("zzes"):
        return word[:-3]
    if re.search(r"(ss|x|ch|sh)es$", word):
        return word[:-2]
    if word.endswith(("ss", "us", "is")) or not word.endswith("s"):
        return word
    return word[:-1]


def label_of(name: str, labels: dict[str, str] | None = None) -> str:
    """A readable label from a catalog name; ``labels`` holds the ones the estate states."""
    if labels and name in labels:
        return labels[name]
    words = name.split("_")
    words[-1] = singular(words[-1])
    text = " ".join(words)
    return text[:1].upper() + text[1:]


class Profiler:
    """Aggregates the induction needs, each computed once inside the warehouse."""

    def __init__(self, reader: Reader, objects: list[CatalogObject]) -> None:
        self.reader = reader
        self.by_name = {o.full_name: o for o in objects}
        self._distinct: dict[tuple[str, tuple[str, ...]], int] = {}
        self._nulls: dict[tuple[str, str], int] = {}
        self.queries = 0

    def _run(self, sql: str) -> list[tuple[Any, ...]]:
        self.queries += 1
        return self.reader.query(sql)

    def usable(self, obj: CatalogObject, cols: tuple[str, ...]) -> bool:
        return all((c := obj.column(n)) is not None and not c.sensitive for n in cols)

    def distinct(self, obj: CatalogObject, cols: tuple[str, ...]) -> int:
        key = (obj.full_name, cols)
        if key not in self._distinct:
            inner = ", ".join(self.reader.ident(c) for c in cols)
            where = " AND ".join(f"{self.reader.ident(c)} IS NOT NULL" for c in cols)
            sql = f"SELECT count(*) FROM (SELECT DISTINCT {inner} FROM {self.reader.quote(obj)} WHERE {where}) t"
            self._distinct[key] = int(self._run(sql)[0][0])
        return self._distinct[key]

    def nulls(self, obj: CatalogObject, col: str) -> int:
        key = (obj.full_name, col)
        if key not in self._nulls:
            sql = f"SELECT count(*) FROM {self.reader.quote(obj)} WHERE {self.reader.ident(col)} IS NULL"
            self._nulls[key] = int(self._run(sql)[0][0])
        return self._nulls[key]

    def unique(self, obj: CatalogObject, cols: tuple[str, ...]) -> bool:
        if not obj.rows or not self.usable(obj, cols):
            return False
        return all(self.nulls(obj, c) == 0 for c in cols) and self.distinct(obj, cols) == obj.rows

    def inclusion(self, child: CatalogObject, c_cols: tuple[str, ...], parent: CatalogObject, p_cols: tuple[str, ...]) -> dict[str, int]:
        """Rows of the child whose value is found in the parent (holds), is not (breaks), or is
        missing (empty)."""
        on = " AND ".join(f"{self.reader.as_text('c.' + self.reader.ident(a))} = {self.reader.as_text('p.' + self.reader.ident(b))}" for a, b in zip(c_cols, p_cols))
        some_null = " OR ".join(f"c.{self.reader.ident(a)} IS NULL" for a in c_cols)
        keys = ", ".join(self.reader.ident(b) for b in p_cols)
        sql = (
            f"SELECT count(*) FILTER (WHERE NOT ({some_null}) AND p.{self.reader.ident(p_cols[0])} IS NOT NULL), "
            f"count(*) FILTER (WHERE NOT ({some_null}) AND p.{self.reader.ident(p_cols[0])} IS NULL), "
            f"count(*) FILTER (WHERE {some_null}) "
            f"FROM {self.reader.quote(child)} c LEFT JOIN (SELECT DISTINCT {keys} FROM {self.reader.quote(parent)}) p ON {on}"
        )
        holds, breaks, empty = self._run(sql)[0]
        return {"holds": int(holds), "breaks": int(breaks), "empty": int(empty)}

    def determines(self, obj: CatalogObject, a: str, b: str) -> bool:
        """Does every value of ``a`` go with exactly one value of ``b`` in this object?"""
        return self.distinct(obj, (a,)) == self.distinct(obj, (a, b))


def _owners_of(col: str, objects: list[CatalogObject]) -> set[str]:
    """The objects a column is named after: ``owner_id`` belongs to ``owners``. Where two layers
    hold a table of that name, it is named after both."""
    out = set()
    for o in objects:
        stem = singular(o.name)
        if col in (f"{stem}_id", f"{stem}_ref", f"{stem}_key", f"{stem}_number", f"{stem}_no"):
            out.add(o.full_name)
    return out


def _key_of(obj: CatalogObject, prof: Profiler, join_targets: set[str], objects: list[CatalogObject],
            ignore: frozenset[str]) -> tuple[tuple[str, ...], str]:
    """The object's key and how it was found. In order: a column a metric-view join points at;
    a column named after the object itself; any other single unique column that is not named
    after another object (``owner_id`` in ``vehicles`` refers to owners, it does not identify
    the vehicle); the smallest unique combination of up to three columns. Columns in ``ignore``
    (load and lineage columns such as a source line number) are never keys."""
    usable = [c.name for c in obj.columns if not c.sensitive and c.name not in ignore]
    for col in usable:
        if col in join_targets and prof.unique(obj, (col,)):
            return (col,), "join_target"
    for col in usable:
        if obj.full_name in _owners_of(col, objects) and prof.unique(obj, (col,)):
            return (col,), "own_name"
    free = [c for c in usable if not (owners := _owners_of(c, objects)) or obj.full_name in owners]
    for col in free:
        if prof.unique(obj, (col,)):
            return (col,), "unique"
    if obj.rows and len(usable) <= 16:
        # A combination is a key only if no single column of it already is one.
        small = [c for c in usable if prof.nulls(obj, c) == 0 and not prof.unique(obj, (c,))][:8]
        for size in (2, 3):
            for combo in itertools.combinations(small, size):
                if prof.unique(obj, combo):
                    return combo, "combination"
    return (), ""


def _instances_of(o: CatalogObject, personal_key: bool, shown: tuple[str, ...]) -> str:
    """Whether rows of an object may be shown. One with a personal column gives none, unless
    the estate names it as an object about something else (an organisation with a contact
    person): then its rows are shown without those columns. A personal key always withholds."""
    if not any(c.sensitive for c in o.columns):
        return "available"
    return "without_personal_columns" if o.full_name in shown and not personal_key else "withheld_personal"


def induce(
    reader: Reader,
    verified_joins: list[tuple[str, str]] | None = None,
    ignore_columns: tuple[str, ...] = (),
    derived_schemas: tuple[str, ...] = (),
    without_personal_columns: tuple[str, ...] = (),
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """The induced schema level as plain data: ``concepts``, ``relations``, ``measures`` and
    ``meta``. ``verified_joins`` are (child object, parent object) pairs a host has verified.

    ``ignore_columns`` are load and lineage columns that carry no meaning. ``derived_schemas``
    hold objects computed from the others (aggregates): they are shown as derived, and no
    relation is mined from or into them, because a column they share with their source says
    nothing new. Where the catalog records lineage, that is where this comes from; a mirror
    without lineage has to be told."""
    ignore = frozenset(ignore_columns)
    derived = frozenset(derived_schemas)
    objects = sorted(reader.objects(), key=lambda o: o.full_name)
    views = sorted(reader.metric_views(), key=lambda v: v.name)
    prof = Profiler(reader, objects)
    by_name = prof.by_name
    view_of: dict[str, list[MetricView]] = {}
    for v in views:
        view_of.setdefault(v.source, []).append(v)

    join_targets: dict[str, set[str]] = {}
    for v in views:
        for j in v.joins:
            cols = join_columns(j.on)
            if cols and j.source in by_name:
                join_targets.setdefault(j.source, set()).add(cols[1])
    declared_pk = {k.object: k.columns for k in reader.declared_keys() if k.kind == "primary"}

    # ---- concepts: one per object, keyed where a key can be shown ---------------------------
    keys: dict[str, tuple[str, ...]] = {}
    concepts: dict[str, Element] = {}
    for o in objects:
        sig: list[dict[str, Any]] = []
        if o.full_name in declared_pk:
            keys[o.full_name] = declared_pk[o.full_name]
            sig.append(say("declared_key", o.id, columns=list(declared_pk[o.full_name])))
        else:
            key, how = _key_of(o, prof, join_targets.get(o.full_name, set()), objects, ignore)
            if key:
                keys[o.full_name] = key
                sig.append(say(f"unique_key.{how}", o.id, columns=list(key), rows=o.rows))
        for v in view_of.get(o.full_name, []):
            sig.append(say("metric_view_source", v.name, view=v.name))
        comment = o.comment or next((v.comment for v in view_of.get(o.full_name, []) if v.comment), "")
        personal_key = any((c := o.column(k)) is not None and c.sensitive for k in keys.get(o.full_name, ()))
        concepts[o.full_name] = Element(
            id=f"concept:{o.full_name}", label=(labels or {}).get(o.full_name) or label_of(o.name, labels),
            status="declared" if view_of.get(o.full_name) else "proposed", signals=sig,
            body={"object": o.id, "object_name": o.full_name, "rows": o.rows, "definition": comment,
                  "key": list(keys.get(o.full_name, ())), "kind": "derived" if o.schema in derived else "entity",
                  "instances": _instances_of(o, personal_key, without_personal_columns)},
        )

    # ---- relations: declared by metric-view joins, then mined by name and inclusion ---------
    relations: dict[tuple[str, tuple[str, ...], str], Element] = {}

    def add_relation(child: CatalogObject, c_cols: tuple[str, ...], parent: CatalogObject, p_cols: tuple[str, ...],
                     status: str, signal: dict[str, Any], partial_key: bool = False) -> None:
        rid = (child.full_name, c_cols, parent.full_name)
        if rid in relations:
            relations[rid].signals.append(signal)
            if status == "declared":
                relations[rid].status = "declared"
            return
        support: dict[str, Any] | None = None
        if prof.usable(child, c_cols) and prof.usable(parent, p_cols):
            support = prof.inclusion(child, c_cols, parent, p_cols)
            filled = support["holds"] + support["breaks"]
            support["share"] = round(support["holds"] / filled, 4) if filled else None
            if status == "proposed" and (not filled or support["holds"] / filled < SUPPORT_FLOOR):
                dropped.append({"from": child.full_name, "columns": list(c_cols), "to": parent.full_name, **support})
                return
        relations[rid] = Element(
            id=f"relation:{child.full_name}.{'+'.join(c_cols)}->{parent.full_name}",
            label=f"{label_of(child.name)} → {label_of(parent.name)}", status=status, signals=[signal],
            body={"from": f"concept:{child.full_name}", "to": f"concept:{parent.full_name}", "columns": list(c_cols),
                  "to_columns": list(p_cols), "cardinality": "many_to_many" if partial_key else "many_to_one", "support": support,
                  **({"partial_key": True} if partial_key else {})},
        )

    dropped: list[dict[str, Any]] = []
    for v in views:
        child = by_name.get(v.source)
        for j in v.joins:
            cols = join_columns(j.on)
            parent = by_name.get(j.source)
            if child is None or parent is None or cols is None:
                continue
            add_relation(child, (cols[0],), parent, (cols[1],), "declared",
                         say("metric_view_join", v.name, view=v.name, on=j.on.strip()))

    for parent_name, key in sorted(keys.items()):
        parent = by_name[parent_name]
        if not prof.usable(parent, key) or parent.schema in derived:
            continue
        stem = singular(parent.name)
        for child in objects:
            if child.full_name == parent_name or child.schema in derived:
                continue
            if len(key) == 1:
                k = key[0]
                names = [c.name for c in child.columns if not c.sensitive and c.name not in ignore and (
                    c.name == k or c.name.endswith("_" + k) or c.name in (f"{stem}_id", f"{stem}_ref", f"{stem}_key"))]
                for c in names:
                    if (child.full_name, (c,), parent_name) in relations:
                        continue
                    add_relation(child, (c,), parent, key, "proposed",
                                 say("name_and_inclusion.column", child.id, child=child.name, column=c, parent=parent.name, key=k))
            elif all(child.column(k) is not None for k in key) and prof.usable(child, key):
                add_relation(child, key, parent, key, "proposed",
                             say("name_and_inclusion.key", child.id, child=child.name, columns=list(key), parent=parent.name))
            else:
                # The child carries one column of the parent's key, and that column is named
                # after the parent (campaign_id in a table of completions per campaign). It
                # refers to the parent's rows that share the value, not to one row.
                for k in key:
                    if parent_name in _owners_of(k, objects) and k not in ignore and prof.usable(child, (k,)):
                        add_relation(child, (k,), parent, (k,), "proposed",
                                     say("name_and_inclusion.part_of_key", child.id, child=child.name, column=k, parent=parent.name, key=list(key)),
                                     partial_key=True)

    stated_only = []
    for child_name, parent_name in verified_joins or []:
        hit = [r for (c, _, p), r in relations.items() if c == child_name and p == parent_name]
        for r in hit:
            r.status = "declared"
            r.signals.append(say("verified_join", f"{child_name}->{parent_name}"))
        if not hit:
            stated_only.append({"from": child_name, "to": parent_name})

    referenced = {r.body["to"] for r in relations.values()}
    for name, c in concepts.items():
        has_measures = any(v.measures for v in view_of.get(name, []))
        points_out = any(r.body["from"] == c.id for r in relations.values())
        if c.id in referenced:
            c.signals.append(say("referenced", relations=sum(r.body["to"] == c.id for r in relations.values())))
        elif has_measures and points_out and c.body["kind"] != "derived":
            c.body["kind"] = "fact"
        if c.id in referenced and by_name[name].schema in derived:
            c.body["kind"] = "derived"

    # ---- value sets: a column shared by several objects, with few distinct values ----------
    taken = {(c, col) for (c, cols, _p) in relations for col in cols} | {(n, k) for n, ks in keys.items() if len(ks) == 1 for k in ks}
    shared: dict[str, list[CatalogObject]] = {}
    for o in objects:
        for col in o.columns:
            if (not col.sensitive and col.name not in ignore and o.schema not in derived
                    and (o.full_name, col.name) not in taken and re.search(r"char|text|string", col.type, re.I)):
                shared.setdefault(col.name, []).append(o)
    display: dict[str, tuple[str, str, tuple[str, ...]]] = {}
    for v in views:
        for d in v.dimensions:
            m = re.fullmatch(r"source\.(\w+)", d.expr.strip())
            if m and m.group(1) not in display and d.display_name:
                display[m.group(1)] = (d.display_name, d.comment, d.synonyms)
    value_sets: dict[str, Element] = {}
    for col, objs in sorted(shared.items()):
        if len(objs) < REFERENCE_MIN_TABLES:
            continue
        counts = {o.full_name: prof.distinct(o, (col,)) for o in objs}
        if max(counts.values()) > REFERENCE_MAX_VALUES or max(counts.values()) < 2:
            continue
        if not any(o.rows and counts[o.full_name] * 5 <= o.rows for o in objs):
            continue  # nowhere do its values repeat: free text or a title, not a value set
        label, definition, synonyms = display.get(col, (label_of(col), "", ()))
        value_sets[col] = Element(
            id=f"concept:value:{col}", label=label, status="proposed",
            signals=[say("shared_column.value_set", column=col, objects=len(objs), values=max(counts.values()))]
            + ([say("metric_view_dimension", label=label)] if col in display else []),
            body={"kind": "reference", "column": col, "definition": definition, "synonyms": list(synonyms),
                  "values": max(counts.values()), "objects": sorted(counts), "instances": "available"},
        )
        for o in objs:
            relations[(o.full_name, (col,), f"value:{col}")] = Element(
                id=f"relation:{o.full_name}.{col}->value:{col}", label=f"{label_of(o.name)} → {label}", status="proposed",
                signals=[say("shared_column.holds", o.id, object=o.name, column=col, values=counts[o.full_name])],
                body={"from": f"concept:{o.full_name}", "to": f"concept:value:{col}", "columns": [col], "to_columns": [col],
                      "cardinality": "many_to_one", "support": None},
            )

    # ---- hierarchy between value sets: one value of a always goes with one value of b -------
    for a, b in itertools.permutations(sorted(value_sets), 2):
        both = [o for o in shared[a] if o in shared[b]]
        if len(both) < 2 or value_sets[a].body["values"] <= value_sets[b].body["values"]:
            continue
        holds = [o.full_name for o in both if prof.determines(o, a, b)]
        if len(holds) == len(both):
            relations[(f"value:{a}", (a,), f"value:{b}")] = Element(
                id=f"relation:value:{a}->value:{b}", label=f"{value_sets[a].label} → {value_sets[b].label}", status="proposed",
                signals=[say("functional_dependency.all", objects=len(both), a=a, b=b)],
                body={"from": f"concept:value:{a}", "to": f"concept:value:{b}", "columns": [a], "to_columns": [b],
                      "cardinality": "many_to_one", "support": {"objects_checked": len(both), "objects_holding": len(holds)}},
            )

    # A hierarchy that follows from two others (a -> b and b -> c give a -> c) says nothing new.
    steps = {(k[0], k[2]) for k in relations if k[0].startswith("value:") and k[2].startswith("value:")}
    for a_, c_ in sorted(steps):
        if any((a_, b_) in steps and (b_, c_) in steps for b_ in {x for x, _ in steps} if b_ not in (a_, c_)):
            relations.pop((a_, (a_.split(":", 1)[1],), c_), None)

    # ---- attributes and measures -----------------------------------------------------------
    for name, c in concepts.items():
        o = by_name[name]
        used = {col for (cn, cols, _p) in relations if cn == name for col in cols} | set(keys.get(name, ()))
        attrs = []
        for col in o.columns:
            if col.name in used:
                continue
            shown = display.get(col.name, ("", "", ()))
            attrs.append({"column": col.name, "label": shown[0] or label_of(col.name), "type": col.type,
                          "definition": col.comment or shown[1], "sensitive": col.sensitive})
        c.body["attributes"] = attrs

    measures = []
    for v in views:
        if v.source not in concepts:
            continue
        for m in v.measures:
            measures.append({"id": f"measure:{v.name}.{m.name}", "label": m.display_name or label_of(m.name), "name": m.name,
                             "concept": concepts[v.source].id, "view": v.name, "expr": m.expr, "definition": m.comment,
                             "status": "declared", "signals": [say("metric_view_measure", v.name, view=v.name)]})

    def out(e: Element) -> dict[str, Any]:
        return {"id": e.id, "label": e.label, "status": e.status, "signals": list(e.signals), **e.body}

    # Two concepts never carry one label: where they would (the same table name in two layers),
    # each says its layer. A relation is labelled by the two concepts it joins.
    seen: dict[str, int] = {}
    for c in [*concepts.values(), *value_sets.values()]:
        seen[c.label] = seen.get(c.label, 0) + 1
    for name, c in concepts.items():
        if seen[c.label] > 1:
            c.label = f"{c.label} ({by_name[name].schema})"
    shown_as = {c.id: c.label for c in [*concepts.values(), *value_sets.values()]}
    for r in relations.values():
        r.label = f"{shown_as[r.body['from']]} → {shown_as[r.body['to']]}"

    all_concepts = [out(c) for _, c in sorted(concepts.items())] + [out(c) for _, c in sorted(value_sets.items())]
    return {
        "concepts": all_concepts,
        "relations": sorted((out(r) for r in relations.values()), key=lambda r: r["id"]),
        "measures": sorted(measures, key=lambda m: m["id"]),
        "meta": {
            "objects": len(objects), "metric_views": len(views), "queries": prof.queries,
            "support_floor": SUPPORT_FLOOR, "dropped_below_floor": sorted(dropped, key=lambda d: (d["from"], d["columns"], d["to"])),
            "verified_joins_without_columns": stated_only,
        },
    }
