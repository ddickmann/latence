"""The corpus-wide TYPE-vocabulary canonicalization engine (T3) — pure Python, deterministic.

**Why this exists.** Extraction induces its labels **per chunk** (the extraction redesign), which
is what gives each chunk a small, relevant, uncapped label set — the recall + precision win, and
the root fix for the label-dimension OOM. The price is **type drift**: chunk A induces ``org``,
chunk B ``organization``, chunk C ``company``, all naming the same type. Corpus-mode used to buy
consistency upstream by freezing (and salience-capping) ONE schema before extraction, paying
recall for it.

This module earns that consistency **downstream** instead. It folds the run's raw induced type
labels into clusters, elects one canonical label per cluster, and emits a
:class:`~latence_core.contracts.TypeVocabulary` — a canonical vocabulary plus the
alias→canonical map. The Stage that wraps it (``type_consolidation.cascade``) then rewrites every
mention's and relation's ``label`` to its canonical form *before* entity resolution, keeping the
raw label on the record for audit. Consistency is EARNED after extraction, never IMPOSED before
it.

**The cascade** mirrors the entity resolver's audited-rung design (``latence_core.disambiguation``)
but over TYPE STRINGS rather than entity surfaces — same discipline, different vocabulary:

1. ``normalized`` — equal after surface normalization (case, punctuation/underscores/hyphens,
   whitespace, trailing plural ``s``): ``ORG``/``org``/``Org.`` → one cluster. Always on.
2. ``alias`` — an operator-supplied ``{label: label}`` map. The escape hatch that always wins,
   because a domain sometimes needs a fold no rule can infer. Always on.
3. ``acronym`` — a multi-word label's initials equal a single-token label: ``point of contact`` ↔
   ``poc``.
4. ``prefix`` — a short single token that ABBREVIATES a longer single token: ``org`` ↔
   ``organization``, ``corp`` ↔ ``corporation``. Deliberately gated (``max_abbreviation_len`` and
   a length-ratio floor) because an ungated prefix rule over-merges real pairs such as
   ``product``/``production``: an abbreviation is SHORT and its expansion is MUCH longer, so both
   gates must hold.
5. ``embedding`` — optional semantic rung over the type strings themselves (``org`` ↔ ``company``
   is a *synonym*, not a surface variant, so no lexical rule can catch it). It consumes the
   framework's existing :class:`~latence_core.capability.Embedder` seam — a dependency accepted,
   not a new seam invented — and is OFF unless a stack wires an embedder, keeping core torch-free.

**Canonical election** is deterministic and salience-first, matching the resolver's dominant-label
vote: the cluster's most-observed label wins; ties break on token count, then length, then
lexicographic order. (An LLM naming pass is deliberately NOT wired here: it would add a second
model call, a network dependency and a determinism risk for a decision the corpus's own dominant
usage already answers. The vocabulary is emitted as data, so an operator who disagrees corrects it
through the ``aliases`` config — a review loop, not a hidden model.)

**Determinism.** Every rung is order-independent (labels are processed in sorted order, folds go
through Union-Find, alias lists and the emitted vocabulary are sorted), so identical inputs give a
byte-identical vocabulary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from latence_core.contracts import CanonicalType, TypeKind, TypeVocabulary

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from latence_core.capability import Embedder

__all__ = ["TypeCanonicalizer", "normalize_type"]

# Rung names stamped on a cluster's ``strategies`` (the audit trail).
_RUNG_NORMALIZED = "normalized"
_RUNG_ALIAS = "alias"
_RUNG_ACRONYM = "acronym"
_RUNG_PREFIX = "prefix"
_RUNG_EMBEDDING = "embedding"

_PUNCTUATION_TO_SPACE = str.maketrans({c: " " for c in "_-/\\.,;:()[]{}\"'`"})


def normalize_type(label: str) -> str:
    """The normalized comparison form of a type label (lowercase, depunctuated, depluralized).

    ``ORG``, ``org.``, ``Org``, ``orgs`` and ``ORG_`` all normalize to ``org``; ``Point of
    Contact`` and ``point-of-contact`` to ``point of contact``. The trailing-plural strip is
    conservative (``s`` only, never ``ss``, and never on a ≤3-character token) so ``address``
    survives and ``organizations`` folds onto ``organization``.
    """
    lowered = label.strip().lower().translate(_PUNCTUATION_TO_SPACE)
    tokens = [_depluralize(token) for token in lowered.split() if token]
    return " ".join(tokens)


def _depluralize(token: str) -> str:
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


class _UnionFind:
    """Minimal Union-Find over label strings (the resolver's fold primitive, type-side)."""

    def __init__(self, items: Iterable[str]) -> None:
        self._parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:  # path compression
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: str, right: str) -> bool:
        """Fold two labels into one cluster; ``True`` when this call actually merged them."""
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return False
        # Deterministic root choice (lexicographic), so the fold order never affects the result.
        low, high = sorted((left_root, right_root))
        self._parent[high] = low
        return True

    def groups(self) -> dict[str, list[str]]:
        clusters: dict[str, list[str]] = {}
        for item in sorted(self._parent):
            clusters.setdefault(self.find(item), []).append(item)
        return clusters


class TypeCanonicalizer:
    """Folds a corpus's raw induced type labels into a canonical :class:`TypeVocabulary`.

    Config mirrors the entity resolver's knobs so an operator reads one mental model:

        aliases: ``{raw label: target label}`` — the always-wins operator fold (both sides are
            matched on their normalized form, so ``Org.`` and ``org`` are the same key).
        enable_acronym / enable_prefix: turn those lexical rungs off for a strict posture.
        max_abbreviation_len / min_expansion_ratio: the ``prefix`` rung's precision gates — a
            candidate abbreviation must be at most ``max_abbreviation_len`` characters AND the
            expansion at least ``min_expansion_ratio``× its length. Defaults (5, 2.0) fold
            ``org``→``organization`` while refusing ``product``→``production``.
        embedder: an optional :class:`~latence_core.capability.Embedder` for the semantic rung
            (synonyms no lexical rule can see). ``None`` ⇒ the rung never runs.
        embedding_threshold: cosine floor for that rung (default 0.86 — deliberately strict; an
            over-merged TYPE retypes every node of that type, a costlier error than drift).
        seed: a previously persisted :class:`TypeVocabulary` (T4 incremental runs). The freshly
            built vocabulary is merged INTO it with the append-don't-churn rule, so an existing
            canonical label is never renamed by a later run.
    """

    def __init__(
        self,
        *,
        aliases: Mapping[str, str] | None = None,
        enable_acronym: bool = True,
        enable_prefix: bool = True,
        max_abbreviation_len: int = 5,
        min_expansion_ratio: float = 2.0,
        embedder: Embedder | None = None,
        embedding_threshold: float = 0.86,
        seed: TypeVocabulary | None = None,
    ) -> None:
        self._aliases = {
            normalize_type(k): normalize_type(v) for k, v in (aliases or {}).items()
        }
        self._enable_acronym = enable_acronym
        self._enable_prefix = enable_prefix
        self._max_abbreviation_len = max_abbreviation_len
        self._min_expansion_ratio = min_expansion_ratio
        self._embedder = embedder
        self._embedding_threshold = embedding_threshold
        self._seed = seed

    # -- public surface ----------------------------------------------------

    def build(
        self,
        entity_labels: Mapping[str, int],
        relation_labels: Mapping[str, int] | None = None,
    ) -> TypeVocabulary:
        """Cluster the observed labels (``{label: observation count}``) into a vocabulary.

        Entity and relation labels are clustered SEPARATELY (they are different vocabularies —
        folding one into the other would corrupt the graph), then emitted as one vocabulary
        keyed by :class:`~latence_core.contracts.TypeKind`, ordered by ``(kind, canonical)``.
        """
        types = [
            *self._cluster(TypeKind.ENTITY, entity_labels),
            *self._cluster(TypeKind.RELATION, relation_labels or {}),
        ]
        built = TypeVocabulary(
            types=sorted(types, key=lambda t: (t.kind.value, t.canonical))
        )
        return self._seed.merge(built) if self._seed is not None else built

    # -- clustering --------------------------------------------------------

    def _cluster(self, kind: TypeKind, counts: Mapping[str, int]) -> list[CanonicalType]:
        labels = sorted(counts)
        if not labels:
            return []
        folds = _UnionFind(labels)
        # Rung strategies are tracked per FOLD and rolled up per cluster, so the audit says which
        # rule(s) built a cluster rather than which rules merely ran.
        strategies: dict[str, set[str]] = {}

        def fold(left: str, right: str, rung: str) -> None:
            if folds.union(left, right):
                root = folds.find(left)
                strategies.setdefault(root, set()).add(rung)

        # 1. normalized-surface equality, and 2. the operator alias map (both purely lexical and
        # pairwise, so one sorted double loop covers them without an ordering dependency).
        normalized = {label: normalize_type(label) for label in labels}
        for i, left in enumerate(labels):
            for right in labels[i + 1 :]:
                rung = self._lexical_rung(normalized[left], normalized[right])
                if rung is not None:
                    fold(left, right, rung)

        # 3. the optional semantic rung over the type strings themselves.
        for left, right in self._embedding_pairs(labels, normalized):
            fold(left, right, _RUNG_EMBEDDING)

        # Roll each fold's strategies up to its final cluster root (a chain of folds can move the
        # root, so re-resolve rather than trusting the root captured at fold time).
        rolled: dict[str, set[str]] = {}
        for member, rungs in strategies.items():
            rolled.setdefault(folds.find(member), set()).update(rungs)

        return [
            self._canonical_type(kind, members, counts, sorted(rolled.get(root, ())))
            for root, members in sorted(folds.groups().items())
        ]

    def _lexical_rung(self, left: str, right: str) -> str | None:
        """The first lexical rung that folds two NORMALIZED labels, or ``None``."""
        if left == right:
            return _RUNG_NORMALIZED
        if self._aliases.get(left) == right or self._aliases.get(right) == left:
            return _RUNG_ALIAS
        if self._enable_acronym and (_is_acronym_of(left, right) or _is_acronym_of(right, left)):
            return _RUNG_ACRONYM
        if self._enable_prefix and self._is_abbreviation(left, right):
            return _RUNG_PREFIX
        return None

    def _is_abbreviation(self, left: str, right: str) -> bool:
        """Is one single-token label a gated prefix-abbreviation of the other?

        Both gates are required, and both exist for the same reason: an ungated "one is a prefix
        of the other" rule merges ``product``/``production`` and ``person``/``personnel``, which
        would retype every node of those types. A real abbreviation is SHORT (≤
        ``max_abbreviation_len``) and its expansion is MUCH longer (≥ ``min_expansion_ratio``×),
        which ``org``/``organization`` (3, 4.0×) satisfies and ``product``/``production``
        (7 characters — too long to be an abbreviation) does not.
        """
        short, long = sorted((left, right), key=len)
        if " " in short or " " in long or not short:
            return False
        return (
            long.startswith(short)
            and len(short) <= self._max_abbreviation_len
            and len(long) >= len(short) * self._min_expansion_ratio
        )

    def _embedding_pairs(
        self, labels: Sequence[str], normalized: Mapping[str, str]
    ) -> list[tuple[str, str]]:
        """Label pairs the optional semantic rung folds (empty when no embedder is wired).

        Embeds each label's NORMALIZED form (so casing/punctuation never moves a vector) once,
        then folds every pair at or above the cosine threshold. Pairs are returned in sorted
        order so the fold sequence — and therefore the elected roots — is deterministic.
        """
        if self._embedder is None or len(labels) < 2:
            return []
        vectors = list(self._embedder.embed(normalized[label] for label in labels))
        if len(vectors) != len(labels):
            msg = (
                f"Embedder returned {len(vectors)} vectors for {len(labels)} type labels — the "
                "Embedder contract is one vector per input, in input order."
            )
            raise ValueError(msg)
        pairs: list[tuple[str, str]] = []
        for i, left in enumerate(labels):
            for j in range(i + 1, len(labels)):
                if _cosine(vectors[i], vectors[j]) >= self._embedding_threshold:
                    pairs.append((left, labels[j]))
        return pairs

    def _canonical_type(
        self,
        kind: TypeKind,
        members: Sequence[str],
        counts: Mapping[str, int],
        strategies: Sequence[str],
    ) -> CanonicalType:
        """Elect the cluster's canonical label and package it as a :class:`CanonicalType`.

        Election is salience-first — the most-observed member wins, the same dominant-label vote
        the entity resolver uses for an entity's type — with deterministic tiebreaks: more
        tokens, then more characters (an expansion reads better than an abbreviation when both
        are equally frequent), then lexicographic order.
        """
        canonical = min(
            members,
            key=lambda label: (
                -counts.get(label, 0),
                -len(label.split()),
                -len(label),
                label,
            ),
        )
        return CanonicalType(
            kind=kind,
            canonical=canonical,
            aliases=sorted(m for m in members if m != canonical),
            mention_count=sum(counts.get(m, 0) for m in members),
            strategies=list(strategies),
        )


def _is_acronym_of(candidate: str, expansion: str) -> bool:
    """Is ``candidate`` the initials of ``expansion``'s words (``poc`` ← ``point of contact``)?"""
    words = expansion.split()
    if " " in candidate or len(words) < 2 or len(candidate) < 2:
        return False
    return candidate == "".join(word[0] for word in words)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity of two equal-width vectors; ``0.0`` when either is a zero vector."""
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
