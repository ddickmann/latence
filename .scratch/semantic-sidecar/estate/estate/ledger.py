"""The ledger: what the engine asserted and what people decided, both append-only.

An Assertion is one induced element with the signals it rests on and the actor that made it.
A Decision is a person's verdict on an element, keyed on the element's id, which is built from
catalog object names and so stays the same across rebuilds. The layer is the replay of both: a
rebuild appends new assertions and never touches a decision, so curation survives it
(ADR-0069).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ENGINE = "engine:induce/1"
VERDICTS = ("approved", "rejected")


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


# What an element states, per kind: the fields a decision is about. Support counts, signals and
# labels are not among them, so a rebuild on changed data keeps the claim, and a proposal that
# now says something else (another parent, other columns) gets a new one.
CLAIM_FIELDS: dict[str, tuple[str, ...]] = {
    "concept": ("id", "kind", "object_name", "column", "key"),
    "relation": ("id", "from", "to", "columns", "to_columns"),
    "measure": ("id", "concept", "view", "expr"),
    "roll_up": ("from", "to", "kind"),
    "equivalent": ("concepts",),
    "part_of": ("from", "to", "kind"),
    "describes": ("from", "to"),
    "class": ("id", "parent"),
    "subclass": ("concept", "parent"),
    "possibly_same": ("a", "b"),
}
UNORDERED = {("equivalent", "concepts")}


def claim_of(kind: str, el: dict[str, Any]) -> str:
    """A short digest of what the element states. A decision is made on a claim, not on an id."""
    if kind == "possibly_same":
        stated: list[Any] = sorted((el["a"], el["b"]))
    else:
        stated = [sorted(el.get(f) or []) if (kind, f) in UNORDERED else el.get(f) for f in CLAIM_FIELDS[kind]]
    return hashlib.sha256(_canon([kind, stated]).encode()).hexdigest()[:16]


# Kinds whose id says all that the claim says (both ends are in it). A decision on one of them
# may come without a claim, as a line written by hand does. For the others the id stays the
# same while the statement changes (a concept placed under another class), so a decision
# without a claim does not say what it decides, and is not applied.
ID_STATES_THE_CLAIM = {"concept", "relation", "measure", "roll_up", "equivalent", "describes", "possibly_same"}
STRUCTURE_KINDS = {"roll_ups": "roll_up", "equivalent": "equivalent", "part_of": "part_of", "describes": "describes"}
CLASS_KINDS = {"classes": "class", "subclass_of": "subclass"}


class Decisions:
    """The decisions in force, asked element by element while the layer is built.

    ``verdict`` answers for one element and remembers what it answered, so that at the end the
    build can say which decisions were applied, which were made on a statement that has since
    changed (stale), which do not say what they decide, and which name nothing in this build."""

    def __init__(self, decisions: dict[str, dict[str, Any]] | None = None) -> None:
        self.by = decisions or {}
        self._applied: dict[str, str] = {}
        self._stale: dict[str, dict[str, str]] = {}
        self._bare: set[str] = set()

    def verdict(self, kind: str, el: dict[str, Any]) -> dict[str, Any] | None:
        d = self.by.get(el["id"])
        if d is None:
            return None
        now = claim_of(kind, el)
        if "claim" not in d:
            if kind not in ID_STATES_THE_CLAIM:
                self._bare.add(el["id"])
                return None
        elif d["claim"] != now:
            if el["id"] not in self._applied:
                self._stale[el["id"]] = {"subject": el["id"], "decided_on": d["claim"], "now": now}
            return None
        self._stale.pop(el["id"], None)
        self._applied[el["id"]] = d["verdict"]
        return {k: d[k] for k in ("verdict", "by", "reason") if k in d}

    def rejected(self, kind: str, el: dict[str, Any]) -> bool:
        v = self.verdict(kind, el)
        return v is not None and v["verdict"] == "rejected"

    def marked(self, kind: str, el: dict[str, Any]) -> dict[str, Any]:
        """The element with the decision in force on it, if there is one."""
        v = self.verdict(kind, el)
        return {**el, "decision": v} if v is not None else el

    def summary(self) -> dict[str, Any]:
        seen = set(self._applied) | set(self._stale) | self._bare
        return {"approved": sum(v == "approved" for v in self._applied.values()), "rejected": sum(v == "rejected" for v in self._applied.values()),
                "unknown_subjects": sorted(set(self.by) - seen), "stale": [self._stale[k] for k in sorted(self._stale)], "without_claim": sorted(self._bare)}


def assertions_of(model: dict[str, Any], structure: dict[str, Any] | None = None, classes: dict[str, Any] | None = None,
                  actor: str = ENGINE) -> list[dict[str, Any]]:
    """One assertion per element of the model, of the structure and of the class tree."""
    groups = [(kind[:-1], model[kind]) for kind in ("concepts", "relations", "measures")]
    groups += [(kind, (structure or {}).get(key, [])) for key, kind in STRUCTURE_KINDS.items()]
    groups += [(kind, (classes or {}).get(key, [])) for key, kind in CLASS_KINDS.items()]
    out = []
    for kind, elements in groups:
        for el in elements:
            body = {k: v for k, v in el.items() if k not in ("signals", "decision")}
            out.append({
                "id": hashlib.sha256(_canon([kind, body]).encode()).hexdigest()[:20],
                "kind": kind, "subject": el["id"], "actor": actor, "signals": el.get("signals", []), "payload": body,
            })
    return out


def append(path: Path, rows: list[dict[str, Any]]) -> int:
    """Append the rows whose id the file does not hold yet. Returns how many were new."""
    seen = {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()} if path.exists() else set()
    new = [r for r in rows if r["id"] not in seen]
    with open(path, "a", encoding="utf-8") as fh:
        for r in new:
            fh.write(_canon(r) + "\n")
    return len(new)


def read_decisions(path: Path) -> dict[str, dict[str, Any]]:
    """The decision in force per element: the last one written for it."""
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("verdict") not in VERDICTS or not d.get("subject") or not d.get("by"):
            raise ValueError(f"{path}:{n}: a decision needs a subject, a verdict of {VERDICTS} and who made it")
        out[d["subject"]] = d
    return out


def replay(model: dict[str, Any], decisions: Decisions | dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The layer: every element with the decision in force on it. A rejected element stays in
    the ledger and leaves the layer; what pointed at a rejected concept leaves with it."""
    asked = decisions if isinstance(decisions, Decisions) else Decisions(decisions)
    out = dict(model)
    out["concepts"] = [asked.marked("concept", c) for c in model["concepts"] if not asked.rejected("concept", c)]
    kept = {c["id"] for c in out["concepts"]}
    out["relations"] = [asked.marked("relation", r) for r in model["relations"] if not asked.rejected("relation", r) and r["from"] in kept and r["to"] in kept]
    out["measures"] = [asked.marked("measure", m) for m in model["measures"] if not asked.rejected("measure", m) and m["concept"] in kept]
    out["meta"] = {**model["meta"], "decisions": asked.summary()}
    return out
