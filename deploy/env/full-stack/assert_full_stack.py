"""Assert the full-stack Blackwell/CUDA-13 run really produced what it claims (issue #199).

The last gate of ``verify-full-stack.sh``, and the reason the run record in
``deploy/env/compatibility-matrix.yaml`` can say "corpus in, Quality Report + Exports out" as a
FACT. It reads the artifacts the run wrote and refuses every shape of near-miss the framework's own
history has produced:

* **no Quality Report / no exports** — the run did not complete;
* **a report with ``document_count: 0``** — the corpus never got in (a Source extension mismatch
  reads exactly like a green run otherwise);
* **``contracts_complete: false``** — records shipped without Provenance/Classification;
* **a learned Stage with ``skipped: true``** — the skip-with-flag posture (ADR-0036 G1) is CORRECT
  behaviour on a CPU host and a FALSE GREEN for a GPU verification: the pipeline finishes, the
  report renders, and the GPU stack was never exercised;
* **a learned Stage with ``device_selected != "cuda"``** — the #191 signature: configured for cuda,
  resolved to cpu, output still plausible;
* **an empty ``corpus.jsonl``** — a pipeline that exported a header and no records;
* **a missing KG export** — Graph Assembly ran but nothing was materialised.

``check_run`` is a pure function over a storage root so it is unit-tested offline (see
``packages/latence-core/tests/test_compat.py``): the assertion that guards the one run nobody can
re-run cheaply cannot itself be untested.

    python assert_full_stack.py <storage-root-dir> [run-id]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# The Stages whose whole point is that they ran ON THE GPU. Kept in step with stack.yaml by
# `test_assert_full_stack_requires_every_cuda_stage_of_the_stack`, which reads the stack file.
CUDA_STAGES: tuple[str, ...] = (
    "extract",
    "redact_pii",
    "redact_financial",
    "disambiguate",
)

# Written by the served-OCR boundary leg; a run that skipped it produces no document text at all.
PARSE_STAGE = "parse"


def _report_path(root: Path, run_id: str) -> Path:
    return root / "_latence" / "runs" / run_id / "quality-report.json"


def _export_dir(root: Path, run_id: str) -> Path:
    return root / "_latence" / "runs" / run_id / "export"


def check_run(root: Path, run_id: str = "full-stack-0001") -> list[str]:
    """Every way this run failed to be the verification it claims to be. Empty means green."""
    failures: list[str] = []
    report_path = _report_path(root, run_id)
    if not report_path.is_file():
        return [
            f"no Quality Report at {report_path} — the run did not complete "
            "(there is no partial credit here: the report IS the run's output contract)."
        ]
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"the Quality Report at {report_path} is not valid JSON: {exc}"]

    rendered = report_path.with_suffix(".md")
    if not rendered.is_file():
        failures.append(
            f"no human-readable Quality Report at {rendered} (S10: the report is both "
            "machine- and human-readable)."
        )

    if int(report.get("document_count", 0)) < 1:
        failures.append(
            "the Quality Report records document_count=0 — no document reached the pipeline. "
            "A corpus whose extensions the Source does not accept produces exactly this."
        )
    if not report.get("contracts_complete", False):
        failures.append(
            "contracts_complete is false — exported records are missing Provenance/Classification "
            "or carry misaligned offsets."
        )

    stages = {
        str(stage.get("name")): stage
        for stage in report.get("stages", [])
        if isinstance(stage, dict)
    }
    for name in (*CUDA_STAGES, PARSE_STAGE):
        stage = stages.get(name)
        if stage is None:
            failures.append(f"stage '{name}' is absent from the Quality Report — it never ran.")
            continue
        if stage.get("skipped"):
            failures.append(
                f"stage '{name}' was SKIPPED ({stage.get('skip_reason')}). Skip-with-flag is the "
                "correct posture on a CPU host and a false green here: this run exists to prove "
                "the GPU/served path, so a skipped Stage fails it."
            )
        if int(stage.get("records_out", 0)) < 1 and name == PARSE_STAGE:
            failures.append(
                "the parse Stage emitted no records — nothing came back from the served OCR "
                "endpoint, so no leg downstream of it means anything."
            )

    for name in CUDA_STAGES:
        stage = stages.get(name)
        if stage is None:
            continue
        selected = stage.get("device_selected")
        if selected != "cuda":
            failures.append(
                f"stage '{name}' resolved device_selected={selected!r}, not 'cuda'. That is the "
                "#191 signature: configured for the GPU, executed on the CPU, output still "
                "plausible."
            )

    exports = _export_dir(root, run_id)
    corpus = exports / "corpus.jsonl"
    if not corpus.is_file():
        failures.append(f"no RAG corpus export at {corpus}.")
    elif not [line for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()]:
        failures.append(f"the RAG corpus export {corpus} is empty — zero records were exported.")

    for artifact in ("graph-nodes.parquet", "graph-edges.parquet"):
        if not (exports / artifact).is_file():
            failures.append(f"no knowledge-graph export at {exports / artifact}.")

    return failures


def main(argv: list[str]) -> int:
    if not 2 <= len(argv) <= 3:
        print(__doc__)
        return 2
    root = Path(argv[1])
    run_id = argv[2] if len(argv) == 3 else "full-stack-0001"
    if not root.is_dir():
        print(f"FAIL: storage root {root} does not exist — the run produced nothing.")
        return 1
    failures = check_run(root, run_id)
    if failures:
        print(f"FAIL: the full-stack verification produced {len(failures)} problem(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("OK: served OCR -> GPU learned Providers -> text stack -> Quality Report + Exports")
    print(f"    report:  {_report_path(root, run_id)}")
    print(f"    exports: {_export_dir(root, run_id)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
