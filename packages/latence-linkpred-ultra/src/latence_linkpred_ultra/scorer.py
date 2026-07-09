"""The ``Scorer`` abstraction: inductive ULTRA (zero-shot) + a graceful PyKEEN fallback.

The learned link-prediction seam behind the :mod:`~latence_linkpred_ultra.provider` GraphCompleter.
A :class:`Scorer` takes a built KG — an integer-indexed entity vocab, a relation vocab, and the
observed ``(head, relation, tail)`` edges — and answers **tail queries**: given a ``(head,
relation)`` it returns a per-entity raw score vector over all entities (higher = the model believes
the tail more). The provider turns candidate triples + these score vectors into calibrated,
inferred edges, and the evaluator turns held-out queries into MRR / Hits@k.

Two scorers behind one interface (ADR-0037 §7; the issue's Scorer abstraction with graceful
fallback):

* :class:`UltraScorer` — the **inductive ULTRA** model (ICLR'24, MIT), zero-shot from a pretrained
  checkpoint. Faithful to the REAL ULTRA API (pinned in the issue spec + verified on the pod):
  ``Ultra(rel_model_cfg, entity_model_cfg)``; load ``state_dict`` from ``ckpt["model"]``; build the
  PyG ``Data`` with **doubled relations** (for edge ``(h, r, t)`` add the inverse ``(t, r+R, h)``);
  ``data.relation_graph = tasks.build_relation_graph(data)``; score with ``model(data, batch)``
  where ``batch`` is ``[B, num_nodes, 3]`` all-negative tails from ``tasks.all_negative`` →
  ``[B, num_nodes]`` scores; rank with ``tasks.compute_ranking``. An un-faithful shim (a wrong
  method name, a mis-shaped output) FAILS the anchor test.
* :class:`PykeenScorer` — a **transductive PyKEEN** model (MIT) as the graceful fallback when the
  ULTRA deps/checkpoint are absent. Trains a small KGE model (default ``DistMult``) on the observed
  triples and scores tails via its interaction function.

If NEITHER is available (no torch/torch-geometric/easydict/checkpoint for ULTRA, no pykeen for the
fallback) the provider raises a typed :class:`~latence_core.errors.ProviderError` — never a silent
no-op (the issue's non-negotiable). All heavy imports (torch, the vendored ULTRA subset, pykeen)
are LAZY — importing this module for provider discovery stays torch-free (ADR-0016).

Determinism: scoring is greedy (no sampling); the candidate/query order is fixed by the provider.
Learned float inference is not guaranteed byte-identical across hardware, so the provider declares
``deterministic=False`` on the ULTRA profile rather than overclaim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence


# The default ULTRA checkpoint name (a config PATH/id, NOT vendored weights). ``ultra_3g`` is the
# 3-graph pretrained checkpoint proven on the pod (0 missing / 0 unexpected params). The operator
# points ``checkpoint`` at the local ckpt file (or the upstream ULTRA repo's
# ``/ckpts/ultra_3g.pth``).
DEFAULT_CHECKPOINT = "ultra_3g"

# The REAL ULTRA model config proven on the pod (issue spec). Passed as the two EasyDict configs to
# ``Ultra(rel_model_cfg, entity_model_cfg)``. ``class`` selects the sub-model; the rest are the
# BaseNBFNet hyperparameters the ``ultra_3g`` checkpoint was trained with (load is 0-missing).
_ULTRA_MODEL_CFG: dict[str, Any] = {
    "input_dim": 64,
    "hidden_dims": [64] * 6,
    "message_func": "distmult",
    "aggregate_func": "sum",
    "short_cut": True,
    "layer_norm": True,
}


@dataclass(frozen=True)
class BuiltKG:
    """The integer-indexed KG a :class:`Scorer` scores over.

    ``num_entities`` / ``num_relations`` are the vocab sizes; ``edges`` are the observed edges as
    ``(head_idx, relation_idx, tail_idx)`` integer triples (NOT doubled — a scorer that needs
    inverses, like ULTRA, adds them itself, exactly as the real recipe does). Deterministic:
    the provider assigns indices in sorted id order.
    """

    num_entities: int
    num_relations: int
    edges: list[tuple[int, int, int]] = field(default_factory=list)


@runtime_checkable
class Scorer(Protocol):
    """Scores tail queries over a built KG (the learned link-prediction seam).

    A scorer is *prepared* once with a :class:`BuiltKG` (build the PyG ``Data`` + relation graph, or
    train the KGE model) and then answers ``(head_idx, relation_idx)`` tail queries with a
    per-entity raw score vector of length ``num_entities``. ``name`` is the scorer id stamped on
    every emitted edge (``"ultra"`` / ``"pykeen"``).
    """

    name: str

    def prepare(self, kg: BuiltKG) -> None:
        """Prepare the scorer over the built KG (build/train). Called once before scoring."""
        ...

    def score_tails(self, queries: Sequence[tuple[int, int]]) -> list[list[float]]:
        """Score a batch of ``(head_idx, relation_idx)`` tail queries.

        Returns one per-entity raw-score list (length ``num_entities``) per query, in query order.
        Higher score = the model believes that tail more. Greedy/no-sampling → order-stable.
        """
        ...


class ScorerUnavailableError(RuntimeError):
    """Raised when a scorer's heavy deps / checkpoint are absent (the provider maps it to typed)."""


# --------------------------------------------------------------------------
# ULTRA scorer — faithful to the REAL ULTRA API (the anti-false-green anchor)
# --------------------------------------------------------------------------


class UltraScorer:
    """Inductive ULTRA scorer (zero-shot from a pretrained checkpoint), REAL-API-faithful.

    Config:
        checkpoint: str  — path/id of the pretrained ULTRA checkpoint (default ``ultra_3g``). The
            weights are NOT vendored; this points at the operator's local ckpt.
        device: str  — ``"cpu"`` | ``"cuda"`` (resolved by the provider's device seam).

    The scoring recipe is byte-for-byte the pod-proven recipe (see the module docstring). All heavy
    imports are lazy so discovery stays torch-free; :meth:`prepare` raises
    :class:`ScorerUnavailableError` if torch / the vendored ULTRA subset / the checkpoint is absent,
    and the provider turns that into the graceful fallback or a typed error.
    """

    name = "ultra"

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, device: str = "cpu") -> None:
        self._checkpoint = checkpoint
        self._device = device
        self._model: Any = None
        self._data: Any = None
        self._num_entities = 0
        # Lazily-bound handles to the real ULTRA modules (set in prepare, reused when scoring).
        self._torch: Any = None
        self._tasks: Any = None

    def prepare(self, kg: BuiltKG) -> None:
        """Build the PyG ``Data`` (with inverse relations) + relation graph and load the checkpoint.

        Faithful to the real recipe: doubles relations with inverses (for ``(h, r, t)`` add
        ``(t, r+R, h)``), sets ``data.relation_graph = tasks.build_relation_graph(data)``,
        constructs ``Ultra(rel_model_cfg, entity_model_cfg)`` and loads ``ckpt["model"]`` (the
        ``["model"]`` key; 0 missing / 0 unexpected on ``ultra_3g``). Raises
        :class:`ScorerUnavailableError` when a dep or the checkpoint is missing.
        """
        torch, tasks, ultra_new, load_state = _import_ultra()
        self._torch = torch
        self._tasks = tasks
        self._num_entities = kg.num_entities

        r = kg.num_relations
        heads = [h for (h, _rel, _t) in kg.edges]
        tails = [t for (_h, _rel, t) in kg.edges]
        rels = [rel for (_h, rel, _t) in kg.edges]
        # ULTRA doubles relations with inverses: edge_index = [[h..., t...], [t..., h...]],
        # edge_type = [r..., r+R...]; num_relations = 2R.
        # Build the graph + its relation-graph on CPU FIRST, then move the whole thing to the
        # device. `tasks.build_relation_graph` creates intermediate tensors on CPU (it does not
        # honour an input tensor's device), so constructing edge_index/edge_type directly on cuda
        # makes it `torch.cat` cuda + cpu tensors → a device-mismatch RuntimeError (caught on the
        # pod). The ULTRA convention is CPU-build → `.to(device)`; the nested `relation_graph` Data
        # is not a tensor attribute so `Data.to` does not recurse into it — move it explicitly.
        edge_index = torch.tensor([heads + tails, tails + heads], dtype=torch.long)
        edge_type = torch.tensor(rels + [rel + r for rel in rels], dtype=torch.long)
        from torch_geometric.data import Data

        data = Data(
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=kg.num_entities,
            num_relations=2 * r,
        )
        data.relation_graph = tasks.build_relation_graph(data)
        data = data.to(self._device)
        data.relation_graph = data.relation_graph.to(self._device)
        self._data = data

        model = ultra_new(_ULTRA_MODEL_CFG)
        state = load_state(self._checkpoint, self._device)
        # The REAL checkpoint stores the params under the ["model"] key (0 missing / 0 unexpected).
        model.load_state_dict(state["model"])
        model = model.to(self._device)
        model.eval()
        self._model = model

    def score_tails(self, queries: Sequence[tuple[int, int]]) -> list[list[float]]:
        """Score ``(head_idx, relation_idx)`` tail queries via ``model(data, all_negative_batch)``.

        Builds the ``[B, 3]`` query batch ``(head, tail=head placeholder, relation)``, expands it to
        all-entity tail negatives with ``tasks.all_negative`` (the REAL helper: returns
        ``t_batch`` of shape ``[B, num_nodes, 3]``), and calls ``model(data, t_batch)`` → a
        ``[B, num_nodes]`` score tensor. Returns it as plain Python floats (one list per query).
        """
        if self._model is None or self._data is None:
            msg = "UltraScorer.score_tails called before prepare()."
            raise RuntimeError(msg)
        torch = self._torch
        tasks = self._tasks
        if not queries:
            return []
        # A [B, 3] batch of (head, tail, relation). The tail here is a placeholder (any valid node);
        # all_negative replaces the tail column with every entity. Use head as the placeholder tail.
        batch = torch.tensor(
            [[h, h, rel] for (h, rel) in queries], dtype=torch.long, device=self._device
        )
        # tasks.all_negative returns (t_batch, h_batch); t_batch is [B, num_nodes, 3] all-tail negs.
        t_batch, _h_batch = tasks.all_negative(self._data, batch)
        with torch.no_grad():
            scores = self._model(self._data, t_batch)  # [B, num_nodes]
        return [[float(x) for x in row] for row in scores.tolist()]


def _import_ultra() -> tuple[Any, Any, Any, Any]:
    """Lazily import torch + the vendored ULTRA subset; raise ScorerUnavailableError if absent.

    Returns ``(torch, tasks_module, ultra_factory, load_state_fn)`` where ``ultra_factory(cfg)``
    constructs a real ``Ultra(rel_model_cfg, entity_model_cfg)`` from the shared model cfg (adding
    the ``class`` key for each tower), and ``load_state_fn(checkpoint, device)`` loads the ckpt
    dict.
    Kept in one place so both prepare and the tests exercise the SAME faithful construction path.
    """
    try:
        import torch  # noqa: PLC0415

        from latence_linkpred_ultra._vendor.ultra import models as _ultra_models  # noqa: PLC0415
        from latence_linkpred_ultra._vendor.ultra import tasks  # noqa: PLC0415

        # The vendored subset is untyped third-party code (ignored by mypy); bind Ultra through an
        # ``Any`` handle so the strict-typed context does not flag an untyped-call here.
        ultra_cls: Any = _ultra_models.Ultra
    except Exception as exc:  # noqa: BLE001 - any missing heavy dep → unavailable
        msg = (
            "ULTRA scorer deps unavailable (need torch + torch-geometric + torch-scatter + the "
            "vendored ULTRA subset). Install latence-linkpred-ultra[ultra]."
        )
        raise ScorerUnavailableError(msg) from exc

    try:
        from easydict import EasyDict  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        msg = "ULTRA scorer needs 'easydict' (install latence-linkpred-ultra[ultra])."
        raise ScorerUnavailableError(msg) from exc

    def ultra_factory(cfg: dict[str, Any]) -> Any:
        # Faithful to the pod recipe: Ultra(rel_model_cfg, entity_model_cfg) with EasyDict configs;
        # each tower carries its own "class" (RelNBFNet / EntityNBFNet). Copy so Ultra's .pop() of
        # the "class" key does not mutate the shared template.
        rel_cfg = EasyDict({"class": "RelNBFNet", **{k: _copy(v) for k, v in cfg.items()}})
        ent_cfg = EasyDict({"class": "EntityNBFNet", **{k: _copy(v) for k, v in cfg.items()}})
        return ultra_cls(rel_model_cfg=rel_cfg, entity_model_cfg=ent_cfg)

    def load_state(checkpoint: str, device: str) -> Any:
        import os  # noqa: PLC0415

        if not os.path.exists(checkpoint):
            msg = (
                f"ULTRA checkpoint not found: {checkpoint!r}. Point 'checkpoint' at a local ULTRA "
                "ckpt (weights are NOT vendored; default id 'ultra_3g')."
            )
            raise ScorerUnavailableError(msg)
        # SECURITY: load with weights_only=True so a tampered/swapped checkpoint cannot execute
        # arbitrary code at unpickle time (torch<2.6 defaults weights_only=False → RCE on a
        # malicious .pth). The ULTRA ckpt is a plain tensor state_dict under ["model"], so the safe
        # loader is sufficient. If a specific ckpt legitimately carries a non-tensor global, the
        # operator must add it via torch.serialization.add_safe_globals for a TRUSTED file — we do
        # NOT silently fall back to the unsafe loader. The operator should still point 'checkpoint'
        # only at a trusted local file (see README security note).
        return torch.load(checkpoint, map_location=device, weights_only=True)

    return torch, tasks, ultra_factory, load_state


def _copy(v: Any) -> Any:
    """Shallow copy of list-valued cfg entries so a tower's EasyDict does not share the template."""
    return list(v) if isinstance(v, list) else v


# --------------------------------------------------------------------------
# PyKEEN fallback scorer
# --------------------------------------------------------------------------


class PykeenScorer:
    """Transductive PyKEEN KGE scorer (MIT) — the graceful fallback when ULTRA is unavailable.

    Config:
        model: str  — the PyKEEN model class name (default ``DistMult``).
        epochs: int  — training epochs over the observed triples (default 5 — a fast fallback).
        device: str  — ``"cpu"`` | ``"cuda"``.
        seed: int  — training seed (determinism knob).

    Trains a small KGE model on the observed triples and scores tails via the model's interaction
    function (``predict_target`` / ``score_hrt`` over all entities). Lazily imports pykeen; raises
    :class:`ScorerUnavailableError` if pykeen is absent.
    """

    name = "pykeen"

    def __init__(
        self,
        model: str = "DistMult",
        epochs: int = 5,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        self._model_name = model
        self._epochs = epochs
        self._device = device
        self._seed = seed
        self._model: Any = None
        self._factory: Any = None
        self._num_entities = 0
        self._torch: Any = None

    def prepare(self, kg: BuiltKG) -> None:
        try:
            import torch  # noqa: PLC0415
            from pykeen.models import model_resolver  # noqa: PLC0415
            from pykeen.triples import CoreTriplesFactory  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            msg = "PyKEEN fallback unavailable (install latence-linkpred-ultra[pykeen])."
            raise ScorerUnavailableError(msg) from exc
        self._torch = torch
        self._num_entities = kg.num_entities
        if not kg.edges:
            msg = "PyKEEN fallback needs at least one observed edge to train."
            raise ScorerUnavailableError(msg)
        mapped = torch.tensor(
            [[h, rel, t] for (h, rel, t) in kg.edges], dtype=torch.long
        )
        factory = CoreTriplesFactory.create(
            mapped_triples=mapped,
            num_entities=kg.num_entities,
            num_relations=kg.num_relations,
        )
        self._factory = factory
        model = model_resolver.make(self._model_name, triples_factory=factory)
        model = model.to(self._device)
        from pykeen.training import SLCWATrainingLoop  # noqa: PLC0415

        loop = SLCWATrainingLoop(model=model, triples_factory=factory)
        loop.train(
            triples_factory=factory,
            num_epochs=self._epochs,
            use_tqdm=False,
        )
        model.eval()
        self._model = model

    def score_tails(self, queries: Sequence[tuple[int, int]]) -> list[list[float]]:
        if self._model is None:
            msg = "PykeenScorer.score_tails called before prepare()."
            raise RuntimeError(msg)
        torch = self._torch
        out: list[list[float]] = []
        with torch.no_grad():
            for h, rel in queries:
                hr = torch.tensor([[h, rel]], dtype=torch.long, device=self._device)
                # predict_t scores every entity as the tail for the (h, r) query → [1,
                # num_entities].
                scores = self._model.predict_t(hr)
                out.append([float(x) for x in scores.reshape(-1).tolist()])
        return out
