"""The five states of an element and who moves it between them (ticket 51; the owner, 14 Nov).

- **draft**: the engine has it below what is shown (an element dropped below the support floor);
- **proposed**: shown, nobody decided it;
- **approved**: a named reviewer approved it in the decisions file, or the engine did, for a
  kind that a recorded calibration proves safe (and that kind only);
- **certified**: the data owner the estate file names (``lifecycle.data_owner``) certified an
  element whose approval is in force, on the statement it makes now
  (``certifications.jsonl`` in the build folder: ``subject``, ``by``, ``claim``, ``at``);
- **retired**: a reviewer withdrew it (rejected); it leaves the layer and stays on the record.

The engine approves nothing by itself unless the estate file names a kind under
``lifecycle.auto_approve`` with a calibration record (``kind``, ``precision_lower_bound``,
``target``, ``labels``, ``by``, ``at``) whose lower bound reaches its target. Its approvals say
so (``by: engine`` and the calibration), and a person's decision outranks them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STATES = ("draft", "proposed", "approved", "certified", "retired")
AUTO_KINDS = {"concept": "concepts", "relation": "relations", "measure": "measures"}  # what the engine may approve under a calibration


def calibrated(rules: list[dict[str, Any]]) -> tuple[dict[str, str], list[dict[str, str]]]:
    """The kinds a calibration proves safe (kind -> the calibration's reference), and the
    rules that prove nothing, with the reason."""
    kinds: dict[str, str] = {}
    refused: list[dict[str, str]] = []
    for rule in rules:
        kind, ref = str(rule.get("kind")), str(rule.get("calibration"))
        if kind not in AUTO_KINDS:
            refused.append({"kind": kind, "why": "kind_not_supported"})
            continue
        path = Path(ref)
        if not path.exists():
            refused.append({"kind": kind, "why": "no_calibration_record"})
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("kind") != kind:
            refused.append({"kind": kind, "why": "another_kind"})
        elif float(record.get("precision_lower_bound", 0)) < float(record.get("target", 1)) or not record.get("labels"):
            refused.append({"kind": kind, "why": "below_its_target"})
        else:
            kinds[kind] = ref
    return kinds, refused


def engine_approvals(model: dict[str, Any], kinds: dict[str, str], decided: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The engine's approvals of the proposed elements of a calibrated kind that no person has
    decided. Their ids state their claim, so a decision without one applies (ledger)."""
    out = {}
    for kind, ref in kinds.items():
        for el in model[AUTO_KINDS[kind]]:
            if el.get("status") == "proposed" and el["id"] not in decided:
                out[el["id"]] = {"subject": el["id"], "verdict": "approved", "by": "engine", "calibration": ref}
    return out


def read_certifications(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        c = json.loads(line)
        if not c.get("subject") or not c.get("by") or not c.get("claim") or not c.get("at"):
            raise ValueError(f"{path}:{n}: a certification names the element, who certified it, the claim and the day")
        out.append(c)
    return out


def _groups(export: dict[str, Any]) -> list[list[dict[str, Any]]]:
    structure = [v for k, v in sorted(export["structure"].items()) if k != "not_proposed"]
    return [export["concepts"], export["relations"], export["measures"], export["review"], *structure,
            export["classes"]["classes"], export["classes"]["subclass_of"]]


def with_states(export: dict[str, Any], certifications: list[dict[str, Any]], data_owner: str | None,
                retired: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The export with every element's state, and what the lifecycle did and refused."""
    elements = {el["id"]: el for part in _groups(export) for el in part}
    certified: dict[str, dict[str, str]] = {}
    refused: list[dict[str, str]] = []
    for c in certifications:  # the last certification of an element counts
        el = elements.get(c["subject"])
        approved = el is not None and (el.get("decision") or {}).get("verdict") == "approved"
        if data_owner is None or c["by"] != data_owner:
            refused.append({"subject": c["subject"], "why": "not_the_data_owner"})
        elif el is not None and str(el.get("claim")) != str(c["claim"]):
            refused.append({"subject": c["subject"], "why": "stale"})
        elif not approved:
            refused.append({"subject": c["subject"], "why": "not_approved"})
        else:
            certified[c["subject"]] = {"by": str(c["by"]), "at": str(c["at"])[:10]}

    def stated(el: dict[str, Any]) -> dict[str, Any]:
        verdict = (el.get("decision") or {}).get("verdict")
        if el["id"] in certified:
            return {**el, "state": "certified", "certification": certified[el["id"]]}
        return {**el, "state": "approved" if verdict == "approved" else "retired" if verdict == "rejected" else "proposed"}

    out = {**export, "concepts": [stated(e) for e in export["concepts"]], "relations": [stated(e) for e in export["relations"]],
           "measures": [stated(e) for e in export["measures"]], "review": [stated(e) for e in export["review"]],
           "structure": {k: v if k == "not_proposed" else [stated(e) for e in v] for k, v in export["structure"].items()},
           "classes": {**export["classes"], "classes": [stated(e) for e in export["classes"]["classes"]],
                       "subclass_of": [stated(e) for e in export["classes"]["subclass_of"]]}}
    counts = dict.fromkeys(STATES, 0)
    for part in _groups(out):
        for el in part:
            counts[el["state"]] += 1
    counts["draft"] = len(export.get("dropped") or [])
    counts["retired"] += len(retired)  # withdrawn elements left the layer; they are counted, not shown
    by_engine: dict[str, int] = {}
    for part in _groups(out):
        for el in part:
            if (el.get("decision") or {}).get("by") == "engine":
                kind = next(k for k, v in AUTO_KINDS.items() if part is out[v]) if any(part is out[v] for v in AUTO_KINDS.values()) else "other"
                by_engine[kind] = by_engine.get(kind, 0) + 1
    report = {"states": counts, "retired": sorted(retired), "refused": sorted(refused, key=lambda r: r["subject"]),
              "approved_by_the_engine": dict(sorted(by_engine.items()))}
    return out, report
