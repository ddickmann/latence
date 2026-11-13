"""What the engine says about an element, as a code and its parameters.

The export carries no sentence the engine wrote. A signal, a line of evidence and a reason on
a review item are each ``{kind, code, params, ref}``; a host renders the code from its own
messages, in whatever languages it has. ``render`` gives the English, for the Turtle file and
the command line, and ``schemas/codes.json`` is this table for a host to check its messages
against.

A parameter is a name from the catalog, a number, or a list of names. A code never carries a
value of a personal column: the callers have none to give.
"""

from __future__ import annotations

import json
from typing import Any

# The top classes of the class tree (owner, 2026-11-11: our own six, no outside vocabulary).
# The definitions are what a model is shown too, so they live here once.
TOP: dict[str, str] = {
    "organisation": "A company, a public body or a unit of one that acts under a name.",
    "person": "A human being. Records of persons are never shown.",
    "product": "A thing that is made, sold, owned or repaired, or a kind of such things.",
    "event": "Something that happened at a time and involves other things: a claim, a visit, a recall.",
    "place": "A location or an area: a town, a region, a market.",
    "document": "A text someone wrote and filed: a bulletin, a report, a letter.",
}

# code -> (parameter names, English). The part before the first dot is the signal's kind.
CODES: dict[str, tuple[tuple[str, ...], str]] = {
    # --- concepts and their keys ---------------------------------------------------------
    "declared_key": (("columns",), "primary key ({columns})"),
    "unique_key.join_target": (("columns", "rows"), "({columns}): the target of a metric-view join, unique in the data, over {rows} rows"),
    "unique_key.own_name": (("columns", "rows"), "({columns}): named after the object, unique in the data, over {rows} rows"),
    "unique_key.unique": (("columns", "rows"), "({columns}): unique in the data, over {rows} rows"),
    "unique_key.combination": (("columns", "rows"), "({columns}): a combination unique in the data, over {rows} rows"),
    "metric_view_source": (("view",), "source of the metric view {view}"),
    "referenced": (("relations",), "{relations} relations point here"),
    # --- relations -------------------------------------------------------------------------
    "metric_view_join": (("view", "on"), "the metric view {view} joins on {on}"),
    "name_and_inclusion.column": (("child", "column", "parent", "key"), "{child}.{column} is named like the key {parent}.{key} and its values are found there"),
    "name_and_inclusion.key": (("child", "columns", "parent"), "{child} carries ({columns}), the key of {parent}, and its values are found there"),
    "name_and_inclusion.part_of_key": (("child", "column", "parent", "key"),
                                       "{child}.{column} is named after {parent} and is one column of its key ({key}); its values are found there"),
    "verified_join": ((), "verified by the host's layer job"),
    # --- value sets ------------------------------------------------------------------------
    "shared_column.value_set": (("column", "objects", "values"), "the column {column} appears in {objects} objects with at most {values} distinct values"),
    "shared_column.holds": (("object", "column", "values"), "{object}.{column} holds {values} of the shared values"),
    "metric_view_dimension": (("label",), "named “{label}” as a dimension of a metric view"),
    "metric_view_measure": (("view",), "a measure of the metric view {view}"),
    # --- structure the data proves ---------------------------------------------------------
    "functional_dependency.all": (("objects", "a", "b"), "in all {objects} objects that carry both, each {a} goes with exactly one {b}"),
    "functional_dependency.most": (("objects", "holding", "checked", "a", "b"),
                                   "over {objects} objects, {holding} of {checked} values of {a} go with exactly one {b}"),
    "same_grouping": (("objects", "a", "b"), "in all {objects} objects that carry both, each {a} goes with exactly one {b} and each {b} with exactly one {a}"),
    "refers_to_a_level.all": (("concept", "level"), "each row of {concept} carries one {level}, which is a level of a hierarchy"),
    "refers_to_a_level.some": (("filled", "rows", "concept", "level"),
                               "{filled} of {rows} rows of {concept} carry a {level}, which is a level of a hierarchy; the others sit under none"),
    "key_contains_key": (("concept", "key", "whole", "columns"), "the key of {concept} ({key}) contains the key of {whole} ({columns})"),
    "key_contains_part_of_key": (("concept", "key", "whole", "columns"), "the key of {concept} ({key}) contains {columns}, one column of the key of {whole}"),
    "same_key": (("base", "key"), "a derived object with the key of {base} ({key})"),
    "names_an_entity": (("value_set", "entity", "keys", "values"),
                        "{value_set} has {values} values over the {keys} rows of {entity}: it names a single {entity}, it does not group them"),
    # --- the class tree --------------------------------------------------------------------
    "language_model.class": (("model", "parent"), "proposed by {model} as a class between concepts and {parent}"),
    "language_model.placed": (("model",), "placed by {model}"),
    "language_model.declined": (("model",), "{model} found none of the classes fitting"),
    "attribute_pattern.place": (("column",), "the column is called {column}"),
    "attribute_pattern.person": (("columns",), "it carries personal columns that name a human ({columns})"),
    "attribute_pattern.organisation": (("column",), "it carries a name that is not personal ({column}) with a place or a contact"),
    "attribute_pattern.event": (("column", "targets"), "it is dated ({column}) and points at {targets}"),
    "attribute_pattern.document": (("columns",), "it carries text columns ({columns}) and nothing refers to it"),
    "attribute_pattern.product": (("sources",), "other concepts refer to it ({sources}) and its key names a thing"),
    # --- what people ask --------------------------------------------------------------------
    "used_in_answers": (("answers", "sources"), "used in {answers} answered questions ({sources})"),
    # --- two names that may be the same ----------------------------------------------------
    "same_start": (("word",), "both names begin with “{word}”"),
    "shared_rare_word": (("word",), "both names contain “{word}”"),
    "same_parent": (("level", "parent"), "both belong to the {level} {parent}"),
    "different_parents": (("level", "a", "b"), "they belong to different values of {level} ({a}, {b})"),
    # --- what a pair of records rests on ---------------------------------------------------
    "name.same": ((), "the same name"),
    "name.same_core": ((), "the same name apart from the legal form"),
    "name.near": ((), "nearly the same name"),
    "name.token_share": (("percent",), "the names share {percent}% of their distinctive words"),
    "town.same": ((), "the same town"),
    "town.different": ((), "different towns"),
    "town.missing": ((), "one record has no town"),
    "postcode.same": ((), "the same postcode"),
    "postcode.different": ((), "different postcodes"),
    "legal_form.different": ((), "different legal forms"),
    "source.same": (("source",), "both from {source}"),
    "source.two": (("a", "b"), "{a} and {b}"),
}
for _name, _definition in TOP.items():
    CODES[f"top_class.{_name}.label"] = ((), _name.capitalize())
    CODES[f"top_class.{_name}.definition"] = ((), _definition)


def say(code: str, ref: str = "", **params: Any) -> dict[str, Any]:
    """One statement of the engine, checked against the table: an unknown code or a parameter
    the code does not have is a defect here, not something a host should find out."""
    return check({"kind": code.split(".", 1)[0], "code": code, "params": params, "ref": ref})


def check(item: dict[str, Any]) -> dict[str, Any]:
    """A statement that came from elsewhere (the matcher's evidence) held to the same table."""
    code = item.get("code")
    if code not in CODES:
        raise KeyError(f"unknown code {code!r}")
    if set(item.get("params") or {}) != set(CODES[code][0]):
        raise ValueError(f"{code} takes {sorted(CODES[code][0])}, got {sorted(item.get('params') or {})}")
    if item.get("kind") != code.split(".", 1)[0]:
        raise ValueError(f"{code} is of kind {code.split('.', 1)[0]!r}, got {item.get('kind')!r}")
    return item


def render(item: dict[str, Any]) -> str:
    """The English for one statement. Free text that came with it (a model's reason) follows
    the statement."""
    shown = {k: ", ".join(str(x) for x in v) if isinstance(v, list) else v for k, v in item["params"].items()}
    line = CODES[item["code"]][1].format(**shown)
    text = (item.get("text") or {}).get("text")
    return f"{line}: {text}" if text else line


def text(value: str | None, origin: str, lang: str = "en", code: str | None = None) -> list[dict[str, str]]:
    """Free text as the export carries it: who wrote it and in what language. ``origin`` is
    ``catalog``, ``engine`` (then ``code`` names the message a host can translate) or
    ``model:<name>``. No text gives an empty list."""
    if not value:
        return []
    return [{"text": value, "lang": lang, "origin": origin, **({"code": code} if code else {})}]


def table() -> str:
    """``schemas/codes.json``: every code with its parameters and its English."""
    return json.dumps({"codes": {c: {"params": list(p), "en": en} for c, (p, en) in sorted(CODES.items())}}, indent=1, ensure_ascii=False) + "\n"
