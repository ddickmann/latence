# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""Calibration mode: Hyperedge γ judgments for the isotonic-calibration sample (ticket 09).

The ``latence calibrate`` harness (a post-M5 ticket) fits isotonic regression from ~200
hand-judged Hyperedges per domain. This module is the labeling side: it iterates the rows
of a ``graph-hyperedges.parquet`` (the S9 KG Export's first-class Hyperedge sidecar,
ADR-0057), shows ``tau_text`` + entities/relations, and records one judgment per Hyperedge.

Judgment JSONL format (one object per line, sorted keys)::

    {"hyperedge_id": ..., "judgment": "correct" | "incorrect" | "partial", "note": ...}

``note`` is optional context for the calibration fit reviewer; it is omitted when empty.
Like the gold set the file is append-only and resumable — the last judgment for a
Hyperedge wins, so a correction is simply appended.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

#: The closed judgment vocabulary (ticket 09: correct / incorrect + partial for edge cases).
JUDGMENTS = ("correct", "incorrect", "partial")


@dataclass(frozen=True)
class HyperedgeRow:
    """One Hyperedge as read from ``graph-hyperedges.parquet`` (display fields only)."""

    hyperedge_id: str
    tau_text: str
    entity_node_ids: tuple[str, ...]
    #: The relations list exactly as the export JSON-encoded it (shown verbatim in the UI).
    relations_json: str
    gamma: float | None
    support: int | None
    arity: int | None

    def to_json(self) -> dict[str, Any]:
        return {
            "hyperedge_id": self.hyperedge_id,
            "tau_text": self.tau_text,
            "entity_node_ids": list(self.entity_node_ids),
            "relations_json": self.relations_json,
            "gamma": self.gamma,
            "support": self.support,
            "arity": self.arity,
        }


@dataclass(frozen=True)
class HyperedgeJudgment:
    """One recorded judgment."""

    hyperedge_id: str
    judgment: str
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        obj: dict[str, Any] = {"hyperedge_id": self.hyperedge_id, "judgment": self.judgment}
        if self.note:
            obj["note"] = self.note
        return obj


def load_hyperedges(path: Path) -> list[HyperedgeRow]:
    """Read a ``graph-hyperedges.parquet`` into display rows, in file order."""
    import pyarrow.parquet as pq  # latence-core dependency; imported lazily.

    table = pq.read_table(path)  # type: ignore[no-untyped-call]
    rows: list[HyperedgeRow] = []
    for row in table.to_pylist():
        entity_node_ids: tuple[str, ...] = ()
        raw_entities = row.get("entity_node_ids")
        if isinstance(raw_entities, str):
            try:
                decoded = json.loads(raw_entities)
            except json.JSONDecodeError:
                decoded = []
            if isinstance(decoded, list):
                entity_node_ids = tuple(str(e) for e in decoded)
        gamma = row.get("gamma")
        support = row.get("support")
        arity = row.get("arity")
        rows.append(
            HyperedgeRow(
                hyperedge_id=str(row.get("hyperedge_id") or ""),
                tau_text=str(row.get("tau_text") or ""),
                entity_node_ids=entity_node_ids,
                relations_json=str(row.get("relations") or "[]"),
                gamma=float(gamma) if isinstance(gamma, (int, float)) else None,
                support=int(support) if isinstance(support, int) else None,
                arity=int(arity) if isinstance(arity, int) else None,
            )
        )
    return [r for r in rows if r.hyperedge_id]


def _judgment_from_json(obj: dict[str, Any]) -> HyperedgeJudgment:
    judgment = HyperedgeJudgment(
        hyperedge_id=str(obj["hyperedge_id"]),
        judgment=str(obj["judgment"]),
        note=str(obj.get("note", "")),
    )
    if judgment.judgment not in JUDGMENTS:
        msg = f"judgment must be one of {JUDGMENTS}, got {judgment.judgment!r}"
        raise ValueError(msg)
    if not judgment.hyperedge_id:
        raise ValueError("hyperedge_id must be non-empty")
    return judgment


def load_judgments(path: Path) -> dict[str, HyperedgeJudgment]:
    """Load the judgment file for resume — last judgment per Hyperedge wins."""
    judgments: dict[str, HyperedgeJudgment] = {}
    if not path.exists():
        return judgments
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                judgment = _judgment_from_json(obj)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                msg = f"{path}:{lineno}: malformed judgment line ({exc})"
                raise ValueError(msg) from exc
            judgments.pop(judgment.hyperedge_id, None)
            judgments[judgment.hyperedge_id] = judgment
    return judgments


def append_judgment(
    path: Path,
    known_ids: set[str],
    hyperedge_id: str,
    judgment: str,
    note: str = "",
) -> HyperedgeJudgment:
    """Validate + append one judgment (re-judging an id appends; the loader keeps the last).

    :raises ValueError: unknown Hyperedge id or a judgment outside :data:`JUDGMENTS`.
    """
    if judgment not in JUDGMENTS:
        msg = f"judgment must be one of {JUDGMENTS}, got {judgment!r}"
        raise ValueError(msg)
    if hyperedge_id not in known_ids:
        msg = f"unknown hyperedge_id {hyperedge_id!r} (not in the loaded parquet)"
        raise ValueError(msg)
    record = HyperedgeJudgment(hyperedge_id=hyperedge_id, judgment=judgment, note=note.strip())
    line = json.dumps(record.to_json(), sort_keys=True, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return record


def validate_calibration(path: Path, known_ids: set[str] | None = None) -> list[str]:
    """Every problem in the judgment file, as human-readable strings. Empty list = valid."""
    errors: list[str] = []
    if not path.exists():
        return [f"{path}: judgment file does not exist"]
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                judgment = _judgment_from_json(obj)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                errors.append(f"{path}:{lineno}: malformed line ({exc})")
                continue
            if known_ids is not None and judgment.hyperedge_id not in known_ids:
                errors.append(
                    f"{path}:{lineno}: unknown hyperedge_id {judgment.hyperedge_id!r} "
                    "(not in the parquet)"
                )
    return errors
