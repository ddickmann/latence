"""Byte-identical output across INTERPRETERS, not merely across calls in one process.

Two runs in the same process share a string-hash seed, so an in-process equality check cannot see
a dict/set-iteration-order dependency. This test runs the profiler in two subprocesses with
DIFFERENT ``PYTHONHASHSEED`` values and compares the bytes. That is the only version of the
determinism claim that is worth anything: it is exactly the failure the repo-root conftest guard
exists to stop the suite from hiding.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")

_SCRIPT = """
import json, sys
sys.path.insert(0, %r)
from latence_label_profiler import CorpusProfiler, ProfilerConfig

corpus = []
for i in range(40):
    corpus.append(
        "The commodity code %%03d maps to the tariff heading of the customs schedule." %% i
    )
    corpus.append("The export licence %%03d was issued by the customs authority." %% i)
    corpus.append("Contact officer%%03d@example.org about the binding ruling." %% i)

profile = CorpusProfiler(ProfilerConfig(seed=11, sample_size=60, min_mention_mass=1)).profile(
    corpus
)
sys.stdout.write(profile.to_json())
""" % _SRC


def _run(hash_seed: str) -> str:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_output_is_byte_identical_under_two_different_hash_seeds() -> None:
    first = _run("0")
    second = _run("12345")
    assert first == second
    assert first.strip()
