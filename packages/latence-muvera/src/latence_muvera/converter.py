"""``MuveraFdeConverter`` — the canonical, training-free MUVERA FDE (ADR-0050, #163).

The blessed *real* Fixed-Dimensional-Encoding the in-core ``fde.reference`` stands in for. It
satisfies the framework's :class:`~latence_core.capability.FdeConverter` Protocol
(``dimension`` + ``convert(multivectors)``), so #160's Export wires it exactly like the reference —
emitting the dense ``fde_embedding`` column that lands in the customer's *dense* ANN index — but its
vector is a genuine MUVERA FDE whose dot-product **approximates the Chamfer / MaxSim similarity**
(the property the reference deliberately disclaims, ADR-0050 anti-false-green).

What makes it canonical MUVERA rather than the reference's naive sum-per-bucket (Dai et al.,
*MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings*, 2024):

* **SimHash space partitioning.** ``k_sim = log2(buckets)`` fixed Gaussian hyperplanes
  (deterministic from ``seed``) split the token-vector space into ``buckets`` cells; a token lands
  in the cell named by the sign pattern of its projections.
* **The query/document asymmetry — the crux of the MaxSim guarantee.** For a *document* each cell's
  vectors are **averaged** (a cell centroid), and an **empty cell is filled** from its nearest
  non-empty cell (Hamming distance on the cell code) so no query token ever meets a zero cell; for a
  *query* each cell's vectors are **summed**. Then ``<FDE_query, FDE_doc> = Σ_q <q, centroid_of(q's
  cell in D)> ≈ Σ_q max_d <q, d> = MaxSim(Q, D)`` — a single dense dot-product that estimates the
  late-interaction score. The reference sums both sides with no fill, which is a valid FDE *shape*
  but carries no such approximation, which is exactly why it is honest-but-quality-uncaveated.
* **Repetitions.** ``repetitions`` independent partitions are concatenated to shrink the estimator's
  variance (each block is an independent MaxSim estimate; their concatenation's dot-product sums
  them). ``repetitions=1`` (the default) keeps the output width a drop-in
  ``buckets × input_dimension`` match for the reference — the same emission shape — so a Pipeline
  swaps one for the other by a single config line (ADR-0007).

Design (codebase-design): a **deep module** — a small seam (``convert`` for the index-time document
column) hides the whole SimHash-partition / asymmetric-aggregation / fill-empty / repetition
machinery, and the companion :meth:`encode_query` is the *same construction's* query face (one
converter, two clocks — the query FDE is computed at query time from the identical partition). It is
a **pure transform that holds nothing** (ADR-0048) and names **no model** — MUVERA FDE is
data-oblivious and training-free, so there are no weights and no per-corpus training (the whole
point versus the deferred LightOn regularizer, ADR-0050). Pure-Python and dependency-free, so
it runs in the CPU-first offline dev env and is exercised for real by the conformance suite
(ADR-0007/0016). Deterministic (seeded hyperplanes, rounded sums) → a seeded run is
byte-reproducible (the Baseline bar).

Experimental and off by default (ADR-0050): whether canonical FDE-alone is reliable enough to bless
as a production default — versus the deferred regularizer-refined converter — is the open research
question, so this ships real and evaluable but makes **no** blanket per-corpus quality guarantee.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from latence_core.errors import ConfigError
from latence_core.providers.profile import (
    REPO_LICENSE_CODE,
    REPO_LICENSE_SOURCE,
    ProviderProfile,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

_DEFAULT_BUCKETS = 8
_DEFAULT_REPETITIONS = 1
_DEFAULT_SEED = 0
_ROUND_NDIGITS = 8


class MuveraFdeConverter:
    """Canonical MUVERA FDE Provider satisfying the ``FdeConverter`` Protocol (ADR-0050).

    Config:
        input_dimension: int  — the width of each input token vector (required, >= 1). A token
            vector of any other width is **rejected** with a typed
            :class:`~latence_core.errors.ConfigError` — the faithful contract a real converter
            enforces (anti-false-green: never a silent mis-encode).
        buckets: int  — the number of SimHash cells; must be a power of two >= 1 (``k_sim =
            log2(buckets)`` hyperplanes). Default 8. Finer partitions tighten the MaxSim
            approximation at the cost of a wider output.
        repetitions: int  — independent partitions concatenated to reduce the estimator's variance
            (>= 1). Default 1 (so the output width is ``buckets × input_dimension``, the reference's
            shape — a drop-in). ``dimension`` is ``repetitions × buckets × input_dimension``.
        seed: int  — the deterministic seed for the fixed Gaussian hyperplanes. Default 0 (two runs,
            and two processes, share the identical partition).
        fill_empty: bool  — whether a document's empty cell is filled from its nearest non-empty
            cell (the MUVERA fill-empty-clusters step that gives the asymmetric MaxSim estimate its
            quality). Default True. Applies to :meth:`convert` (documents) only;
            :meth:`encode_query` never fills.

    Two entry points onto one construction:

    * :meth:`convert` — the ``FdeConverter`` Protocol method #160's Export calls: the **document**
      FDE (average + fill-empty per cell), materialised as the ``fde_embedding`` column.
    * :meth:`encode_query` — the deep companion: the **query** FDE (sum per cell) from the identical
      partition, for the query-time dense lookup (one construction, two clocks).
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): a pure-code, CPU, deterministic,
        PolyForm-Noncommercial-1.0.0 FDE.

        MUVERA FDE is **data-oblivious and training-free** — it ships no model weights
        (``model_id=None``, ``license_weights=None``), so its license is trivially the package's own
        verified PolyForm-Noncommercial-1.0.0 (this repo's ``LICENSE``); there is no separate
        weights license to defer (unlike ``sparse.splade`` / the ColBERT multi-vector encoder).
        Deterministic (seeded hyperplanes, rounded sums) and CPU-only; it does not batch and has no
        per-call vendor cost — so a bake-off / conformance run reads honest, non-fabricated columns
        off it (ADR-0036 §2).
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=64,
            model_id=None,
            license=REPO_LICENSE_CODE,
            license_verified=True,
            license_code=REPO_LICENSE_CODE,
            license_weights=None,
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-16",
            deterministic=True,
            batch=False,
            cost_per_1k=None,
            perf_note="reference-correct pure-Python MUVERA FDE; a numpy/SIMD kernel is deferred",
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        input_dimension = int(config.get("input_dimension", 0))
        if input_dimension < 1:
            msg = (
                "MuveraFdeConverter requires a positive 'input_dimension' (the width of each input "
                f"token vector); got {input_dimension}."
            )
            raise ConfigError(msg)
        buckets = int(config.get("buckets", _DEFAULT_BUCKETS))
        if buckets < 1 or (buckets & (buckets - 1)) != 0:
            msg = f"MuveraFdeConverter 'buckets' must be a power of two >= 1, got {buckets}."
            raise ConfigError(msg)
        repetitions = int(config.get("repetitions", _DEFAULT_REPETITIONS))
        if repetitions < 1:
            msg = f"MuveraFdeConverter 'repetitions' must be >= 1, got {repetitions}."
            raise ConfigError(msg)
        self._input_dimension = input_dimension
        self._buckets = buckets
        self._bits = buckets.bit_length() - 1  # log2(buckets); 0 bits ⇒ a single cell
        self._repetitions = repetitions
        self._seed = int(config.get("seed", _DEFAULT_SEED))
        self._fill_empty = bool(config.get("fill_empty", True))
        # The fixed Gaussian hyperplanes, one bundle per repetition, drawn in a fixed order from a
        # single seeded PRNG so two independent processes build the byte-identical partition (the
        # determinism the byte-identical export relies on). Canonical MUVERA uses Gaussian planes
        # for the SimHash locality guarantee (uniformly-random directions on the sphere).
        self._planes = self._build_planes()

    @property
    def dimension(self) -> int:
        """The fixed output width: ``repetitions × buckets × input_dimension``."""
        return self._repetitions * self._buckets * self._input_dimension

    # -- the two faces of the one construction ------------------------------------------------- #

    def convert(self, multivectors: Iterable[list[list[float]]]) -> Iterator[list[float]]:
        """Map each multi-vector to one **document** FDE, in input order (the Protocol method).

        Per SimHash cell the document's token vectors are **averaged**, and an empty cell is filled
        from its nearest non-empty cell (when ``fill_empty``), so the emitted dense vector estimates
        the late-interaction target of a query lookup. An empty multi-vector yields the zero vector
        of width :attr:`dimension`; a token vector of the wrong width is rejected (``ConfigError``).
        Deterministic → byte-reproducible.
        """
        for multivector in multivectors:
            yield self._encode(multivector, average=True, fill=self._fill_empty)

    def encode_query(self, multivectors: Iterable[list[list[float]]]) -> Iterator[list[float]]:
        """Map each multi-vector to one **query** FDE, in input order (the query-time companion).

        Per SimHash cell the query's token vectors are **summed** (never averaged, never filled), so
        that ``<encode_query(Q), convert(D)>`` sums, over each query token, its dot-product with the
        centroid of the document cell it falls in — the MUVERA MaxSim estimate. Same partition,
        same shape, same determinism as :meth:`convert`; this is the *query* clock of the one
        construction, invoked at query time against the ``fde_embedding`` column :meth:`convert`
        wrote at index time.
        """
        for multivector in multivectors:
            yield self._encode(multivector, average=False, fill=False)

    # -- the hidden machinery ------------------------------------------------------------------ #

    def _encode(self, multivector: list[list[float]], *, average: bool, fill: bool) -> list[float]:
        self._validate_widths(multivector)
        out: list[float] = []
        for rep in range(self._repetitions):
            out.extend(self._encode_repetition(multivector, rep, average=average, fill=fill))
        return [round(value, _ROUND_NDIGITS) for value in out]

    def _encode_repetition(
        self, multivector: list[list[float]], rep: int, *, average: bool, fill: bool
    ) -> list[float]:
        planes = self._planes[rep]
        members: dict[int, list[list[float]]] = {}
        for token in multivector:
            cell = self._cell_of(token, planes)
            members.setdefault(cell, []).append(token)
        # The per-cell aggregate (sum for a query, mean for a document) over the tokens that fell in
        # it — the only non-empty cells so far.
        aggregate: dict[int, list[float]] = {
            cell: self._aggregate(tokens, average=average) for cell, tokens in members.items()
        }
        block = [0.0] * (self._buckets * self._input_dimension)
        for cell in range(self._buckets):
            vector = aggregate.get(cell)
            if vector is None:
                # A document's empty cell is filled from its nearest non-empty cell (MUVERA
                # fill-empty-clusters) so a query token landing here still meets a real centroid; a
                # query's empty cell (or a document with no non-empty cell at all) stays zero.
                if fill and aggregate:
                    vector = aggregate[self._nearest_cell(cell, aggregate.keys())]
                else:
                    vector = [0.0] * self._input_dimension
            base = cell * self._input_dimension
            block[base : base + self._input_dimension] = vector
        return block

    def _aggregate(self, tokens: list[list[float]], *, average: bool) -> list[float]:
        acc = [0.0] * self._input_dimension
        for token in tokens:
            for i, value in enumerate(token):
                acc[i] += value
        if not average:
            return acc
        count = len(tokens)
        return [value / count for value in acc]

    def _cell_of(self, token: list[float], planes: list[list[float]]) -> int:
        cell = 0
        for bit, plane in enumerate(planes):
            projection = sum(a * b for a, b in zip(plane, token, strict=True))
            if projection >= 0.0:
                cell |= 1 << bit
        return cell

    @staticmethod
    def _nearest_cell(target: int, candidates: Iterable[int]) -> int:
        """The non-empty cell closest to ``target`` by Hamming distance on the SimHash code.

        Ties are broken by the smallest cell id, so the fill is deterministic (byte-reproducible).
        """
        return min(candidates, key=lambda cell: (bin(target ^ cell).count("1"), cell))

    def _validate_widths(self, multivector: list[list[float]]) -> None:
        for token in multivector:
            if len(token) != self._input_dimension:
                msg = (
                    "MuveraFdeConverter received a token vector of width "
                    f"{len(token)}, expected {self._input_dimension} (the declared "
                    "'input_dimension')."
                )
                raise ConfigError(msg)

    def _build_planes(self) -> list[list[list[float]]]:
        rng = random.Random(self._seed)
        return [
            [
                [rng.gauss(0.0, 1.0) for _ in range(self._input_dimension)]
                for _ in range(self._bits)
            ]
            for _ in range(self._repetitions)
        ]


__all__ = ["MuveraFdeConverter"]
