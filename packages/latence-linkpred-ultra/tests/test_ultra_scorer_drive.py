"""Anti-false-green anchor, guard #2 — the faithful-fake DRIVE (torch-gated).

``UltraScorer`` is driven end-to-end against a FAITHFUL fake whose ``Ultra(cfg, cfg)``,
``model(data, batch) -> [B, num_nodes]`` and ``tasks.all_negative(data, batch) -> (t_batch
[B, num_nodes, 3], h_batch)`` reproduce the REAL ULTRA shapes. A provider that called a
non-existent method, mis-built the doubled-relation ``Data``, or mis-parsed the output shape
raises here.

This needs real torch tensors, so it is ``pytest.importorskip("torch")``-gated and runs on the
pod / any torch host — NOT in the CPU merge gate. The merge gate is covered by guard #1
(``test_ultra_scorer_faithful.py``), which AST-parses the vendored ULTRA source torch-free and
fails RED on any vendored-API drift. The two guards together are the issue's non-negotiable false-
green trap: the torch-free surface anchor always runs, the shape-faithful drive runs wherever torch
is installed.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from latence_linkpred_ultra import scorer as scorer_mod  # noqa: E402
from latence_linkpred_ultra.scorer import BuiltKG, UltraScorer  # noqa: E402


class _FakeData:
    """A minimal stand-in for torch_geometric.data.Data used by the faithful fake."""

    def __init__(self, edge_index: Any, edge_type: Any, num_nodes: int, num_relations: int) -> None:
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.relation_graph: Any = None


class _FakeTasks:
    """Faithful ``tasks`` helpers: the REAL shapes (an un-faithful provider mis-uses these)."""

    @staticmethod
    def build_relation_graph(graph: _FakeData) -> _FakeData:
        # Real ULTRA sets graph.relation_graph and returns graph.
        graph.relation_graph = object()
        return graph

    @staticmethod
    def all_negative(data: _FakeData, batch: Any) -> tuple[Any, Any]:
        # REAL: t_batch is [B, num_nodes, 3] (all-tail negatives), h_batch symmetric.
        b = batch.shape[0]
        n = data.num_nodes
        all_index = torch.arange(n)
        pos_h = batch[:, 0]
        pos_r = batch[:, 2]
        r_index = pos_r.unsqueeze(-1).expand(-1, n)
        h_index, t_index = torch.meshgrid(pos_h, all_index, indexing="ij")
        t_batch = torch.stack([h_index, t_index, r_index], dim=-1)  # [B, N, 3]
        assert t_batch.shape == (b, n, 3)
        return t_batch, t_batch


class _FakeUltra:
    """A faithful fake model: ``model(data, batch) -> [B, num_nodes]`` scores over tails.

    The score is a deterministic function of the tail index so a specific tail can be made #1 — the
    faithful stub returns REAL-shaped per-entity scores; a provider that mis-parses [B, N] fails.
    """

    def __init__(self, rel_model_cfg: Any, entity_model_cfg: Any) -> None:
        # Faithful: the two configs each carry a popped "class".
        self.rel_class = rel_model_cfg["class"]
        self.entity_class = entity_model_cfg["class"]
        self._preferred_tail: int | None = None

    def load_state_dict(self, state: Any) -> None:
        self._preferred_tail = int(state.get("preferred_tail", 0))

    def to(self, device: str) -> _FakeUltra:
        return self

    def eval(self) -> _FakeUltra:
        return self

    def __call__(self, data: _FakeData, batch: Any) -> Any:
        # batch is [B, N, 3]; return [B, N] with the preferred tail scored highest.
        b, n, _ = batch.shape
        scores = torch.zeros(b, n)
        if self._preferred_tail is not None and self._preferred_tail < n:
            scores[:, self._preferred_tail] = 10.0
        return scores


def _install_faithful_fake(monkeypatch: pytest.MonkeyPatch, preferred_tail: int) -> None:
    """Monkeypatch _import_ultra to hand the scorer the faithful fake (torch present, PyG faked)."""

    def ultra_factory(cfg: dict[str, Any]) -> _FakeUltra:
        rel = {"class": "RelNBFNet", **cfg}
        ent = {"class": "EntityNBFNet", **cfg}
        return _FakeUltra(rel, ent)

    def load_state(checkpoint: str, device: str) -> dict[str, Any]:
        return {"model": {"preferred_tail": preferred_tail}}

    def fake_import() -> tuple[Any, Any, Any, Any]:
        return torch, _FakeTasks(), ultra_factory, load_state

    monkeypatch.setattr(scorer_mod, "_import_ultra", fake_import)
    # The scorer imports torch_geometric.data.Data inside prepare; hand it the fake Data.
    import sys
    import types

    fake_pyg = types.ModuleType("torch_geometric")
    fake_pyg_data = types.ModuleType("torch_geometric.data")
    fake_pyg_data.Data = _FakeData  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch_geometric", fake_pyg)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", fake_pyg_data)


def test_ultrascorer_drives_the_real_api_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    """UltraScorer builds the doubled-relation Data, calls model(data, all_negative), parses [B,N]."""
    _install_faithful_fake(monkeypatch, preferred_tail=2)
    kg = BuiltKG(num_entities=4, num_relations=1, edges=[(0, 0, 1), (1, 0, 2)])
    s = UltraScorer(checkpoint="dummy", device="cpu")
    s.prepare(kg)
    # The Data was built with DOUBLED relations (2R) and the relation graph set.
    assert s._data.num_relations == 2  # 2 * 1  # noqa: SLF001
    assert s._data.relation_graph is not None  # noqa: SLF001
    # Score a (head=0, relation=0) tail query → a length-num_entities vector; tail 2 is #1.
    vectors = s.score_tails([(0, 0)])
    assert len(vectors) == 1
    assert len(vectors[0]) == 4
    assert vectors[0].index(max(vectors[0])) == 2


def test_unfaithful_scorer_would_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stub whose model returns the WRONG shape must break the scorer (the false-green trap)."""

    class _WrongShapeUltra(_FakeUltra):
        def __call__(self, data: _FakeData, batch: Any) -> Any:
            # WRONG: return a scalar per batch, not [B, num_nodes]. A faithful consumer breaks.
            return torch.zeros(batch.shape[0])

    def ultra_factory(cfg: dict[str, Any]) -> _WrongShapeUltra:
        return _WrongShapeUltra({"class": "RelNBFNet", **cfg}, {"class": "EntityNBFNet", **cfg})

    def load_state(checkpoint: str, device: str) -> dict[str, Any]:
        return {"model": {"preferred_tail": 0}}

    def fake_import() -> tuple[Any, Any, Any, Any]:
        return torch, _FakeTasks(), ultra_factory, load_state

    monkeypatch.setattr(scorer_mod, "_import_ultra", fake_import)
    import sys
    import types

    fake_pyg = types.ModuleType("torch_geometric")
    fake_pyg_data = types.ModuleType("torch_geometric.data")
    fake_pyg_data.Data = _FakeData  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch_geometric", fake_pyg)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", fake_pyg_data)

    kg = BuiltKG(num_entities=3, num_relations=1, edges=[(0, 0, 1)])
    s = UltraScorer(checkpoint="dummy", device="cpu")
    s.prepare(kg)
    # A faithful consumer parses model output as [B, num_nodes] (one per-entity row per query). A
    # [B]-shaped (scalar-per-query) output cannot be parsed that way: `scores.tolist()` yields a
    # flat list of floats, so `for x in row` iterates a float and raises TypeError. The anchor:
    # an un-faithful (wrong-shape) scorer does NOT silently pass — score_tails RAISES.
    with pytest.raises(TypeError):
        s.score_tails([(0, 0)])
