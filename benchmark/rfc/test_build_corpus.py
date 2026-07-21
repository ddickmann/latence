"""Regression: the §1 build_corpus command must yield a corpus deep enough to pass its own gate.

The reported dogfood finding (§1): with the tutorial's hub seeds (RFC 9110-9114) and ``--max-docs
80``, a plain breadth-first fetch fills the entire budget with depth-1 documents — every RFC a hub
directly references — so ``docs>=2hops_from_seed`` is **0**, failing the tutorial's own gate
(needs ≥15). This proves the fix (a per-document reference-fanout cap that lets the frontier advance
in depth) over a synthetic hub graph, with **no network and no fake dependency**: the real
``_parse_refs`` + BFS + ``_connectivity`` stats run against an injected in-memory fetch.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_MODULE_PATH = Path(__file__).with_name("build_corpus.py")


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_corpus", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bc = _load_module()

# --- a synthetic RFC "web" that reproduces the hub-star pathology -----------------------------
# Seed 1000 is a hub that references 100 documents (1001..1100) — like RFC 9110's 67 references.
# Each of those references a child one "layer" deeper (n -> n+100), forming chains 1001->1101->1201…
# and a couple of same-layer siblings to keep out-degree up. A breadth-first fetch capped at
# max_docs never reaches the 11xx layer (it is queued behind all 100 hub links); a fanout-capped
# fetch advances into it, so depth-2 documents actually enter the corpus.
_SEED = 1000
_HUB_WIDTH = 100
_LAYERS = 6  # bands 1..6 == 1001..1100, 1101..1200, … 1501..1600


def _band(n: int) -> int:
    """1 for 1001..1100, 2 for 1101..1200, … (the hub seed 1000 is band 0)."""
    return (n - 1001) // 100 + 1


def _in_web(n: int) -> bool:
    if n == _SEED:
        return True
    return 1 <= _band(n) <= _LAYERS and 1 <= ((n - 1001) % 100 + 1) <= _HUB_WIDTH


def _children(n: int) -> list[int]:
    if n == _SEED:
        return list(range(1001, 1001 + _HUB_WIDTH))
    band = _band(n)
    base = 1000 + (band - 1) * 100  # first-node-minus-one of this band
    pos = (n - 1001) % 100 + 1  # 1..100 within the band
    refs: list[int] = []
    if band < _LAYERS:
        refs.append(n + 100)  # chain pointer one band deeper
    # two same-band siblings (wrap within the band) to keep median out-degree ≥ 3
    refs.append(base + (pos % _HUB_WIDTH) + 1)
    refs.append(base + ((pos + 1) % _HUB_WIDTH) + 1)
    return sorted(set(refs) - {n})


def _fetch_synthetic(n: int) -> str | None:
    if not _in_web(n):
        return None  # outside the synthetic web
    body = " ".join(f"RFC {m}" for m in _children(n))
    return f"RFC {n} synthetic node.\n\n{body}\n"


def test_hub_seed_is_shallow_without_fanout_cap(tmp_path: Path) -> None:
    """Pre-fix behaviour: uncapped BFS from a hub seed produces 0 documents ≥2 hops deep."""
    stats = bc.build(
        tmp_path, seeds=[_SEED], expand={"references"}, max_docs=80, ref_fanout=None, fetch=_fetch_synthetic
    )
    assert stats.n_docs == 80  # budget filled …
    assert stats.deep == 0  # … entirely with depth-1 documents (the bug)


def test_fanout_cap_reaches_multi_hop_depth(tmp_path: Path) -> None:
    """Post-fix behaviour: the default reference-fanout cap makes the corpus pass the §1 gate."""
    stats = bc.build(tmp_path, seeds=[_SEED], expand={"references"}, max_docs=80, fetch=_fetch_synthetic)
    assert stats.n_docs == 80
    assert stats.median_out_degree >= 3  # still wide
    assert stats.deep >= 15  # and now genuinely deep — the tutorial's gate is satisfiable


def test_refs_sidecar_records_full_edges_regardless_of_fanout(tmp_path: Path) -> None:
    """The fanout cap changes which docs are fetched, never the edges recorded for a fetched doc."""
    import json

    bc.build(tmp_path, seeds=[_SEED], expand={"references"}, max_docs=80, fetch=_fetch_synthetic)
    rows = [json.loads(line) for line in (tmp_path / "_refs.jsonl").read_text().splitlines()]
    hub = next(r for r in rows if r["rfc"] == _SEED)
    # The hub's recorded references are its *full* list (100), not the capped frontier slice (8).
    assert hub["references"] == list(range(1001, 1001 + _HUB_WIDTH))
