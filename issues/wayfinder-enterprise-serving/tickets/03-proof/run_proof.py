"""Ticket-03 equivalence-proof supervisor: drive churn + rebuild with done-probes.

Runs the whole protocol unattended (ADR-0061 §8), sequentially, with per-phase logs, a durable
state file (``phases.json``) and hard churn-witness probes — a phase whose Quality-Report churn
does not match the protocol's expectation FAILS the proof loudly rather than letting a silently
degenerate run (e.g. reuse not engaging) masquerade as evidence.

Phases (all through the real CLI, ``PYTHONHASHSEED=0``, CPU):

1. ``base``    — stage 548/609 docs, ``latence delta`` (bootstrap, no dense embedder)
2. ``add``     — stage the remaining 61 docs, delta (expect 548 reused / 61 extracted)
3. ``update``  — edit 20 docs in place, delta (expect 589 reused / 20 extracted)
4. ``delete``  — remove 20 docs, delta WITH the dense embedder (expect 589 reused / 0 extracted;
                 the final incremental state, embeddings included)
5. ``rebuild`` — full batch run of the same final corpus into a fresh store, embedder on

Resumable: completed phases are skipped on re-invocation (state file), so a crash resumes where
it stopped. Embeddings are attached only in the two compared runs (see ``make_pipeline.py``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
VENV = REPO / ".venv/bin"

# phase -> (corpus mutation, pipeline yaml, CLI verb, run id, expected churn or None)
PHASES: list[tuple[str, str | None, str, str, str, dict[str, int] | None]] = [
    ("base", "base", "incr-noembed.yaml", "delta", "t03-v1", None),
    (
        "add",
        "add",
        "incr-noembed.yaml",
        "delta",
        "t03-v2",
        {"documents_reused": 548, "documents_extracted": 61, "documents_added": 61},
    ),
    (
        "update",
        "update",
        "incr-noembed.yaml",
        "delta",
        "t03-v3",
        # A content-addressed edit is add(new id) + delete(old id): 20 edited, 589 unchanged.
        {"documents_reused": 589, "documents_extracted": 20, "documents_deleted": 20},
    ),
    (
        "delete",
        "delete",
        "incr-embed.yaml",
        "delta",
        "t03-v4",
        {"documents_reused": 589, "documents_extracted": 0, "documents_deleted": 20},
    ),
    ("rebuild", None, "rebuild.yaml", "run", "t03-rebuild", None),
]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _store_for(yaml_name: str) -> Path:
    import yaml as _yaml

    data = _yaml.safe_load((HERE / "pipelines" / yaml_name).read_text(encoding="utf-8"))
    return Path(str(data["storage_uri"]).removeprefix("file://"))


def _probe(yaml_name: str, run_id: str, expect: dict[str, int] | None) -> None:
    store = _store_for(yaml_name)
    report_path = store / "_latence/runs" / run_id / "quality-report.json"
    if not report_path.exists():
        msg = f"done-probe failed: {report_path} missing"
        raise SystemExit(msg)
    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    if expect is None:
        return
    delta = report.get("delta") or {}
    mismatches = {k: (delta.get(k), v) for k, v in expect.items() if delta.get(k) != v}
    if mismatches:
        msg = f"churn-witness probe failed for {run_id}: got/expected {mismatches}"
        raise SystemExit(msg)


def main(proof_root: str) -> None:
    root = Path(proof_root).resolve()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    state_path = root / "phases.json"
    state: dict[str, str] = (
        json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    )
    sys.path.insert(0, str(HERE))
    from prepare_corpus import apply_phase  # noqa: PLC0415 - sibling-script import

    for phase, mutation, yaml_name, verb, run_id, expect in PHASES:
        if state.get(phase) == "done":
            _log(f"phase {phase}: already done, skipping")
            continue
        _log(f"phase {phase}: start")
        if mutation is not None:
            apply_phase(root, mutation)
        cmd = [
            str(VENV / "latence"),
            verb,
            str(HERE / "pipelines" / yaml_name),
            "--run-id",
            run_id,
        ]
        started = time.time()
        with (logs / f"{phase}.log").open("ab") as log:
            result = subprocess.run(  # noqa: S603 - fixed local command
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONHASHSEED": "0"},
                check=False,
            )
        if result.returncode != 0:
            state[phase] = "failed"
            state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")
            msg = f"phase {phase} exited {result.returncode}; see {logs / f'{phase}.log'}"
            raise SystemExit(msg)
        _probe(yaml_name, run_id, expect)
        state[phase] = "done"
        state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")
        _log(f"phase {phase}: done in {(time.time() - started) / 60:.1f} min")
    _log("all phases done")


if __name__ == "__main__":
    main(sys.argv[1])
