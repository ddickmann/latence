"""Built-in reference TypeConsolidator Providers (deterministic, CPU-viable, zero-dep) — T3.

The corpus-level TYPE-vocabulary canonicalizer of the resolution phase (ADR-0054). It ships in
core like the S3 Chunker/Screeners, the S4 gazetteer, the S6 hybrid redactor, the S7 profiler and
the S8 cascade disambiguator, so the demo runs end to end with no extra deps (ADR-0007 CPU-first,
ADR-0016 thin core).

**Why a Stage of its own, not a second job bolted onto ``disambiguation.glinker``.** The
disambiguator resolves entity *mentions* into canonical *entities*; this resolves the *type
vocabulary* those mentions are labelled with. They are two different decisions over two different
vocabularies, and fusing them would (a) make the entity resolver's already-large surface larger
without making it deeper, (b) tie a stack's ability to canonicalize types to its choice of entity
resolver, and (c) prevent either from being swapped alone. Separating them is the Capability/
Provider discipline (ADR-0004/0016): one Capability per decision that genuinely varies, each
independently swappable.

**Why it runs BEFORE the disambiguator.** The entity resolver votes a cluster's type from its
members' labels and gates its low-precision rungs (acronym, substring) on *type compatibility*.
Feeding it a drifted vocabulary therefore both splits clusters that belong together (``org`` vs
``organization`` reads as a type mismatch) and mistypes the ones that survive. Canonical types in,
consistent entities and a consistent graph out.

The Providers here wrap the engine in :mod:`latence_core.type_consolidation` and turn the run's
mentions + relations into the SAME records with ``label`` rewritten to canonical form and
``raw_label`` carrying the extractor's original induced type — the audit trail that makes every
remap inspectable and correctable rather than silent.

Two real strategies behind one seam (ADR-0035, the second-reference bar):

* ``type_consolidation.cascade`` — the full audited cascade (normalized surface → operator alias →
  acronym → gated prefix-abbreviation → optional embedding over the type strings).
* ``type_consolidation.exact_surface`` — normalized surface + operator aliases ONLY. The
  high-precision baseline: it never folds two different type strings unless an explicit alias links
  them, so it collapses pure surface drift (``ORG``/``org``/``Org.``) and nothing else. That is the
  posture a regulated adopter wants when a wrong TYPE retypes every node of that type.

Determinism: the engine processes labels in sorted order through Union-Find with a lexicographic
root, the canonical election is a salience vote with total tiebreaks, and records are re-emitted in
input order — so a seeded run is byte-identical (Baseline bar).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import ValidationError

from latence_core.capability import TypeConsolidation
from latence_core.contracts import EntityMention, RelationMention, TypeKind, TypeVocabulary
from latence_core.errors import ConfigError
from latence_core.pipeline import CapabilityKind
from latence_core.providers.profile import (
    REPO_LICENSE_CODE,
    REPO_LICENSE_SOURCE,
    ProviderProfile,
)
from latence_core.type_consolidation import TypeCanonicalizer

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from latence_core.capability import Embedder

# The two label-bearing mention carriers the remap rewrites. Constrained (not bound) so a remapped
# EntityMention stays an EntityMention to the type checker — the Stage must not silently widen a
# carrier on its way downstream.
_R = TypeVar("_R", EntityMention, RelationMention)

# The in-core deterministic Embedder the semantic rung falls back to when a stack asks for the rung
# but names no Provider — the same default the glinker/embedding disambiguators use, so the seam is
# exercised offline with no model and no weights (ADR-0012/0016).
_DEFAULT_EMBEDDER = "embedding.hashing"


class CascadeTypeConsolidator:
    """TypeConsolidator Provider: drifted induced types → one canonical type vocabulary.

    Config:
        aliases: dict[str, str] — the operator's always-wins ``{raw type: target type}`` fold.
            Both sides match on their normalized form, so ``Org.`` and ``org`` are one key. This
            is the correction loop: an operator who disagrees with a cluster on the Quality
            Report fixes it here, in config, rather than in a model's head.
        relation_aliases: dict[str, str] — the same, for the RELATION vocabulary. Entity and
            relation types are canonicalized in the same pass but never mixed (an entity type
            ``organization`` and a relation type ``organization of`` are different vocabularies),
            so their alias maps are separate too. Absent ⇒ ``aliases`` is used for both.
        enable_acronym / enable_prefix: bool — turn those lexical rungs off for a strict posture.
            Default True.
        max_abbreviation_len: int / min_expansion_ratio: float — the prefix rung's precision
            gates. Defaults (5, 2.0) fold ``org``→``organization`` while refusing
            ``product``→``production``.
        use_embedding_merge: bool — enable the semantic rung over the type strings (``org`` ↔
            ``company`` is a synonym no lexical rule can see). Default False, mirroring the
            glinker resolver's ``use_embedding_merge``: precise by default, semantic on request.
        embedding_threshold: float — cosine floor for that rung. Default 0.86, deliberately
            strict — an over-merged TYPE retypes every node of that type, a costlier error than
            leaving two near-synonyms apart.
        embedder: {"provider": name, "config": {...}} — the Embedder Provider backing the
            semantic rung. Absent ⇒ the in-core ``embedding.hashing``. Only resolved when
            ``use_embedding_merge`` is on, so a stack that never asks for the rung never loads a
            model.
        consolidate_relations: bool — also canonicalize the RELATION type vocabulary. Default
            True; relations flow through untouched when off.
        seed_vocabulary: the JSON dump of a previously committed
            :class:`~latence_core.contracts.TypeVocabulary` (T4). The incremental
            :class:`~latence_core.delta_runner.DeltaRunner` injects the Corpus Version's
            persisted vocabulary here, and this run's freshly-built one is merged INTO it under
            the append-don't-churn rule — so a delta's new type either aliases to an existing
            canonical entry or appends a new one, and an existing canonical label is never
            renamed (renaming it would retype every node already in the KG). It is config rather
            than a side channel because the Pipeline is the run's whole input: the same YAML
            replays the same vocabulary, which is what makes a delta reproducible. Absent ⇒ the
            full-run path (canonicalize from scratch), which is also what ``--reinduce`` selects.

    A note on LLM canonical NAMING (T3's optional step): it is deliberately not wired as a
    default. The corpus's own dominant usage already answers "which surface should this cluster be
    called" — that is what the salience vote reads — and an LLM call here would add a network
    dependency, a second model and a determinism risk to a decision that is already deterministic
    and, crucially, *reviewable*: the vocabulary is emitted as data on the Quality Report, so an
    operator who prefers ``organization`` over ``org`` sets ``aliases`` and gets it, permanently
    and reproducibly. A review loop beats a hidden model.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic,
        PolyForm-Noncommercial-1.0.0.

        Ships no model weights (``model_id=None``) — the optional semantic rung *borrows* an
        Embedder Provider, which declares its own profile, so this Provider never claims a model
        it does not own.
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=64,
            model_id=None,
            license=REPO_LICENSE_CODE,
            license_verified=True,
            license_code=REPO_LICENSE_CODE,
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-25",
            deterministic=True,
            batch=False,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = dict(config or {})
        self._aliases = _str_map(cfg.get("aliases"), "aliases", self.provider_name)
        self._relation_aliases = (
            _str_map(cfg.get("relation_aliases"), "relation_aliases", self.provider_name)
            if cfg.get("relation_aliases") is not None
            else self._aliases
        )
        self._enable_acronym = bool(cfg.get("enable_acronym", True))
        self._enable_prefix = bool(cfg.get("enable_prefix", True))
        self._max_abbreviation_len = int(cfg.get("max_abbreviation_len", 5))
        self._min_expansion_ratio = float(cfg.get("min_expansion_ratio", 2.0))
        self._use_embedding_merge = bool(cfg.get("use_embedding_merge", False))
        self._embedding_threshold = float(cfg.get("embedding_threshold", 0.86))
        self._embedder_spec = cfg.get("embedder")
        self._consolidate_relations = bool(cfg.get("consolidate_relations", True))
        self._seed: TypeVocabulary | None = _seed_vocabulary(
            cfg.get("seed_vocabulary"), self.provider_name
        )
        self._embedder: Embedder | None = None

    @property
    def provider_name(self) -> str:
        return "type_consolidation.cascade"

    def seed_with(self, vocabulary: TypeVocabulary | None) -> None:
        """Seed the next :meth:`consolidate` with a previously persisted vocabulary (T4).

        The programmatic form of the ``seed_vocabulary`` config key, for an embedder of this
        Provider that already holds the vocabulary object (the DeltaRunner goes through config,
        so the Pipeline stays the run's whole input). The freshly-induced vocabulary is merged
        INTO the seed under the append-don't-churn rule
        (:meth:`~latence_core.contracts.TypeVocabulary.merge`), so a new type aliases into an
        existing canonical cluster without renaming — and therefore without retyping every node
        already in the KG. Passing ``None`` restores the full-run path.
        """
        self._seed = vocabulary

    # -- the Capability ----------------------------------------------------

    def consolidate(
        self,
        mentions: Iterable[EntityMention],
        relations: Iterable[RelationMention] = (),
    ) -> TypeConsolidation:
        """Canonicalize the run's type vocabulary and remap every record onto it.

        Materialises the streams once (the vocabulary is a corpus-level decision — it cannot be
        made from a prefix of the labels), counts each raw label's observations, builds the
        vocabulary, then re-emits every record in input order with ``label`` canonical and
        ``raw_label`` set to the original induced type. A record whose label was already
        canonical is re-emitted with ``raw_label`` **unset**, so ``raw_label`` means exactly
        "this was rewritten" and never adds noise to a record nothing happened to.
        """
        mention_list = list(mentions)
        relation_list = list(relations) if self._consolidate_relations else []
        passthrough_relations = [] if self._consolidate_relations else list(relations)

        entity_counts = _count(m.label for m in mention_list)
        relation_counts = _count(r.label for r in relation_list)
        vocabulary = self._canonicalizer().build(entity_counts, relation_counts)

        entity_map = _remap_table(vocabulary, TypeKind.ENTITY)
        relation_map = _remap_table(vocabulary, TypeKind.RELATION)
        return TypeConsolidation(
            mentions=[_relabel(m, entity_map) for m in mention_list],
            relations=[
                *(_relabel(r, relation_map) for r in relation_list),
                *passthrough_relations,
            ],
            vocabulary=vocabulary,
        )

    # -- internals ---------------------------------------------------------

    def _canonicalizer(self) -> TypeCanonicalizer:
        return TypeCanonicalizer(
            aliases=self._aliases,
            relation_aliases=self._relation_aliases,
            enable_acronym=self._enable_acronym,
            enable_prefix=self._enable_prefix,
            max_abbreviation_len=self._max_abbreviation_len,
            min_expansion_ratio=self._min_expansion_ratio,
            embedder=self._resolve_embedder() if self._use_embedding_merge else None,
            embedding_threshold=self._embedding_threshold,
            seed=self._seed,
        )

    def _resolve_embedder(self) -> Embedder:
        """Resolve (once) the semantic rung's Embedder from ``config['embedder']``.

        The nested block is parsed and capability-checked by the one module that owns it
        (:mod:`latence_core.subprovider`): ``spec`` is ``None`` (⇒ this Provider's own default, the
        in-core deterministic ``embedding.hashing``) or a ``{"provider", "config"}`` mapping naming
        any Embedder Provider. A malformed shape is a :class:`~latence_core.errors.ConfigError` and
        a non-Embedder Provider a :class:`~latence_core.errors.ContractError` — both LOUD at wiring
        time, never a silent fallback to a different vocabulary than the operator asked for
        (ADR-0034: no silent errors).

        Two behaviours changed with the move, deliberately. The capability mismatch now raises
        ``ContractError`` (as the Export and Graph-Assembly slots already did) instead of
        ``ConfigError``: one mistake, one type. And the check is
        :func:`~latence_core.capability.satisfies_capability` rather than ``isinstance``, which on
        a ``runtime_checkable`` Protocol invokes ``Embedder.dimension`` via ``hasattr`` — loading a
        served Embedder's backend during config validation, exactly the ADR-0016 violation audit R6
        removed from the Export.
        """
        if self._embedder is not None:
            return self._embedder
        # Imported here, not at module scope: the registry pulls entry points, which is startup
        # cost a stack that never enables the semantic rung should not pay.
        from latence_core.capability import Embedder as EmbedderProtocol
        from latence_core.capability import ProviderRegistry
        from latence_core.subprovider import (
            SubProviderRef,
            SubProviderSlot,
            build_sub_provider,
            sub_provider_ref,
        )

        owner = f"{self.provider_name}: config"
        ref = sub_provider_ref(
            self._embedder_spec,
            capability=CapabilityKind.TYPE_CONSOLIDATION,
            key="embedder",
            owner=owner,
        ) or SubProviderRef(
            slot=SubProviderSlot("embedder", EmbedderProtocol),
            provider=_DEFAULT_EMBEDDER,
        )
        candidate: Embedder = build_sub_provider(
            ref, registry=ProviderRegistry(), owner=owner
        )
        self._embedder = candidate
        return candidate


class ExactSurfaceTypeConsolidator(CascadeTypeConsolidator):
    """Second reference TypeConsolidator (ADR-0035): normalized surface + aliases only.

    Behind the identical :class:`~latence_core.capability.TypeConsolidator` seam as
    :class:`CascadeTypeConsolidator` and satisfying the identical contract — the same records
    back, remapped, with ``raw_label`` provenance and a :class:`TypeVocabulary` — but restricted
    to the two highest-precision rungs: **normalized-surface equality** and **configured
    aliases**. The acronym, prefix-abbreviation and embedding rungs are off and not overridable.

    It therefore collapses pure surface drift (``ORG``/``org``/``Org.``/``orgs`` → one type) and
    refuses everything else: it will not fold ``org`` into ``organization`` without an explicit
    alias. That is the exact-blocking baseline a fuzzy consolidator is measured against, and it is
    the posture a regulated adopter chooses when the cost of a wrong merge (every node of a type
    silently retyped) outweighs the cost of residual drift (two types where one was meant).
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic, licensed
        PolyForm-Noncommercial-1.0.0.
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=32,
            model_id=None,
            license=REPO_LICENSE_CODE,
            license_verified=True,
            license_code=REPO_LICENSE_CODE,
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-25",
            deterministic=True,
            batch=False,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = dict(config or {})
        # The defining properties; not overridable (the seam's varying behaviour IS the rung set).
        cfg["enable_acronym"] = False
        cfg["enable_prefix"] = False
        cfg["use_embedding_merge"] = False
        super().__init__(cfg)

    @property
    def provider_name(self) -> str:
        return "type_consolidation.exact_surface"


# -- helpers -----------------------------------------------------------------


def _count(labels: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts


def _remap_table(vocabulary: TypeVocabulary, kind: TypeKind) -> dict[str, str]:
    """``{raw label: canonical label}`` for one kind — only the labels that actually change.

    Built once per run instead of calling
    :meth:`~latence_core.contracts.TypeVocabulary.canonical_for` per record: the vocabulary is
    small but the mention stream is not, and a linear scan per mention would be O(mentions ×
    types).
    """
    return vocabulary.alias_map(kind)


def _relabel(record: _R, table: Mapping[str, str], /) -> _R:
    """Rewrite ``record.label`` to its canonical form, preserving the raw label for audit.

    Returns the record UNCHANGED (identity, not a copy) when its label is already canonical, so a
    run whose vocabulary consolidated nothing is byte-identical to a run with no consolidation
    Stage at all — the off-by-default guarantee, verified rather than asserted.
    """
    canonical = table.get(record.label)
    if canonical is None:
        return record
    return record.model_copy(update={"label": canonical, "raw_label": record.label})


def _seed_vocabulary(value: object, provider: str) -> TypeVocabulary | None:
    """Validate the ``seed_vocabulary`` config into a :class:`TypeVocabulary`, loudly (ADR-0034).

    A malformed seed must NOT degrade silently to "no seed": that would canonicalize the delta's
    corpus from scratch and could rename a canonical type already in the KG — exactly the churn
    T4 exists to prevent — while the run reported success. So a wrong shape is a typed
    :class:`~latence_core.errors.ConfigError`, not a fallback.
    """
    if value is None:
        return None
    if isinstance(value, TypeVocabulary):
        return value
    if not isinstance(value, dict):
        msg = (
            f"{provider}: config 'seed_vocabulary' must be a TypeVocabulary JSON object "
            f"(as committed in the Corpus Version); got {type(value).__name__}."
        )
        raise ConfigError(msg)
    try:
        return TypeVocabulary.model_validate(value)
    except ValidationError as exc:
        msg = f"{provider}: config 'seed_vocabulary' is not a valid TypeVocabulary: {exc}"
        raise ConfigError(msg) from exc


def _str_map(value: object, key: str, provider: str) -> dict[str, str]:
    """Validate an operator-supplied ``{str: str}`` config map, loudly (ADR-0034)."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        msg = f"{provider}: config {key!r} must be a mapping of type label -> type label."
        raise ConfigError(msg)
    out: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, str):
            msg = f"{provider}: config {key!r} must map strings to strings."
            raise ConfigError(msg)
        if not raw_key.strip() or not raw_value.strip():
            msg = f"{provider}: config {key!r} must not contain empty type labels."
            raise ConfigError(msg)
        out[raw_key] = raw_value
    return out
