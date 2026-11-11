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


def assertions_of(model: dict[str, Any], actor: str = ENGINE) -> list[dict[str, Any]]:
    out = []
    for kind in ("concepts", "relations", "measures"):
        for el in model[kind]:
            body = {k: v for k, v in el.items() if k not in ("signals",)}
            out.append({
                "id": hashlib.sha256(_canon([kind, body]).encode()).hexdigest()[:20],
                "kind": kind[:-1], "subject": el["id"], "actor": actor, "signals": el["signals"], "payload": body,
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


def replay(model: dict[str, Any], decisions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The layer: every element with the decision in force on it. A rejected element stays in
    the ledger and leaves the layer; what pointed at a rejected concept leaves with it."""
    gone = {s for s, d in decisions.items() if d["verdict"] == "rejected"}
    out = dict(model)
    out["concepts"] = [c for c in model["concepts"] if c["id"] not in gone]
    kept = {c["id"] for c in out["concepts"]}
    out["relations"] = [r for r in model["relations"] if r["id"] not in gone and r["from"] in kept and r["to"] in kept]
    out["measures"] = [m for m in model["measures"] if m["id"] not in gone and m["concept"] in kept]
    for kind in ("concepts", "relations", "measures"):
        out[kind] = [{**el, **({"decision": {k: decisions[el["id"]][k] for k in ("verdict", "by", "reason") if k in decisions[el["id"]]}}
                              if el["id"] in decisions else {})} for el in out[kind]]
    out["meta"] = {**model["meta"], "decisions": {"approved": sum(d["verdict"] == "approved" for d in decisions.values()),
                                                  "rejected": len(gone), "unknown_subjects": sorted(
                                                      set(decisions) - {el["id"] for k in ("concepts", "relations", "measures") for el in model[k]})}}
    return out
