"""Stage the ticket-03 churn corpus: deterministic base / add / update / delete batches.

Splits the real multihop_rag markdown corpus (609 docs) into:

* **base** — the first ~90% in sorted-filename order (548 docs), copied into the working input
  dir before the base build;
* **add** — the remaining 61 docs, copied in before the add delta;
* **update** — 20 base docs (every 25th in sorted base order, skipping delete collisions),
  edited in place before the update delta by appending a real, entity-bearing correction
  paragraph (a content-addressed edit: new sha256 id, old id retired);
* **delete** — 20 other base docs (every 27th, disjoint from update), removed before the
  removal delta.

The selection is a pure function of the sorted filename list — no RNG — and is written to
``batches.json`` beside the working dir so every later phase (and the report) reads the same
batches. Mutation of the working dir between phases is done by ``run_proof.py`` calling
``apply_phase``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
SOURCE_MD = REPO / "benchmark/sota-corpus/multihop_rag/markdown"

UPDATE_APPENDIX = (
    "\n\n## Editorial update\n\n"
    "This article was revised by the Reuters editorial desk in partnership with the Associated "
    "Press on 12 August 2026; the correction was reviewed in New York by editor Jane Alvarez.\n"
)


def batches() -> dict[str, list[str]]:
    names = sorted(p.name for p in SOURCE_MD.glob("*.md"))
    if len(names) != 609:
        msg = f"expected 609 multihop_rag docs, found {len(names)}"
        raise SystemExit(msg)
    split = int(len(names) * 0.9)  # 548 base / 61 add
    base, add = names[:split], names[split:]
    delete = [base[i] for i in range(0, len(base), 27)][:20]
    delete_set = set(delete)
    update = [n for i, n in enumerate(base) if i % 25 == 0 and n not in delete_set][:20]
    return {"base": base, "add": add, "update": update, "delete": delete}


def apply_phase(root: Path, phase: str) -> None:
    """Bring the working input dir into the state the named phase's run expects."""
    plan = json.loads((root / "batches.json").read_text(encoding="utf-8"))
    workdir = root / "input" / "markdown"
    if phase == "base":
        workdir.mkdir(parents=True, exist_ok=True)
        for name in plan["base"]:
            shutil.copy2(SOURCE_MD / name, workdir / name)
    elif phase == "add":
        for name in plan["add"]:
            shutil.copy2(SOURCE_MD / name, workdir / name)
    elif phase == "update":
        for name in plan["update"]:
            path = workdir / name
            path.write_text(
                path.read_text(encoding="utf-8") + UPDATE_APPENDIX, encoding="utf-8"
            )
    elif phase == "delete":
        for name in plan["delete"]:
            (workdir / name).unlink()
    else:  # pragma: no cover - driver passes known phases
        msg = f"unknown phase {phase!r}"
        raise SystemExit(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof-root", required=True)
    parser.add_argument("--phase", choices=["init", "base", "add", "update", "delete"])
    args = parser.parse_args()
    root = Path(args.proof_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "batches.json"
    if not plan_path.exists():
        plan_path.write_text(json.dumps(batches(), indent=1), encoding="utf-8")
        print(f"wrote {plan_path}")
    if args.phase and args.phase != "init":
        apply_phase(root, args.phase)
        print(f"applied phase {args.phase}")


if __name__ == "__main__":
    main()
