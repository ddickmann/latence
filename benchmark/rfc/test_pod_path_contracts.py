"""T-RET-4 — the pod-path contracts the offline suite never checked.

Three defects sat inside ``# pragma: no cover - pod path`` blocks in the RFC harness. Being
uncovered, the green suite said **nothing** about them, and each fails only on a real pod:

1. ``run_benchmark.py`` queried the exported edge parquet as ``SELECT source, target``. The actual
   schema is ``source_node_id`` / ``target_node_id`` — a DuckDB binder error that kills the whole
   hub-filter and multi-hop leg before the first query. (:class:`DuckDBGraphSource` itself was
   always correct; only the harness was wrong, which is exactly why nothing caught it.)
2. Both harnesses passed the SPLADE checkpoint as ``{"model_id": ...}``. The Provider reads
   ``cfg.get("model", "")`` and does **not** reject unknown keys, so the id resolved to ``""`` in
   silence — a SPLADE leg that runs, returns nothing useful, and reports no error.
3. ``build_questions.py`` passed a :class:`SparseVector` straight into
   ``RetrievalQuery.sparse_terms``, which is ``dict[str, float]``.

Each one is a *silent* failure of the same kind: a contract asserted in one module and violated in
another, with no shared check between them. So these tests do not restate the fixed values — they
derive the expectation from the **producing** side (the exporter's own schema, the Provider's own
config key, the contract's own annotation) and assert the **consuming** side matches. Rename a
column and this file goes red before a pod ever sees it.
"""

from __future__ import annotations

import ast
import re
import typing
from pathlib import Path

import pytest
from latence_core.capability import SparseVector

_HARNESS = Path(__file__).resolve().parent
_REPO_ROOT = _HARNESS.parents[1]


def _source(name: str) -> str:
    return (_HARNESS / name).read_text(encoding="utf-8")


# --- 1. the exported edge-parquet columns -------------------------------------------------


def _exporter_edge_columns() -> set[str]:
    """The edge column names the export actually carries — derived from the CONTRACT that defines
    them (:class:`~latence_core.contracts.GraphEdge`), never restated here. Rename a field on the
    contract and this gate moves with it.
    """
    from latence_core.contracts import GraphEdge  # noqa: PLC0415

    return set(GraphEdge.model_fields)


def test_the_harness_selects_edge_columns_the_exporter_actually_writes() -> None:
    """Every identifier the harness SELECTs from the edge parquet must exist in the export schema.

    The pre-fix harness selected ``source, target``; the exporter writes ``source_node_id`` /
    ``target_node_id``. DuckDB raises a binder error, so the failure is loud on a pod and invisible
    here — the worst combination, since the pod is the expensive place to discover it.
    """
    exported = _exporter_edge_columns()
    text = _source("run_benchmark.py")

    selects = re.findall(r"SELECT\s+(.+?)\s+FROM\s+read_parquet\(", text, re.I | re.S)
    assert selects, (
        "no `SELECT ... FROM read_parquet(...)` found in run_benchmark.py — either the graph query "
        "was restructured or this gate's pattern has gone stale. Either way it is checking nothing."
    )

    for clause in selects:
        for column in (c.strip() for c in clause.split(",")):
            bare = column.split()[0].strip("\"'`")
            if bare == "*" or not bare.isidentifier():
                continue
            assert bare in exported, (
                f"run_benchmark.py selects {bare!r} from the edge parquet, but the exporter writes "
                f"{sorted(exported)}. This is a DuckDB binder error on the pod and the multi-hop "
                "leg dies before the first query."
            )


# --- 2. the SPLADE provider config key ----------------------------------------------------


def _splade_config_key() -> str:
    """The checkpoint key the SPLADE config is actually READ with.

    ``SpladeQueryEncoder`` (``latence_retrieval.sparse``) reads nothing itself — it passes its
    config straight through to :class:`~latence_splade.provider.SpladeCompositionEmbedder`, *"the
    same class the pipeline ran at index time"*, so query- and index-time are configured
    identically. So the consuming site is in ``latence-splade``, and that is where this gate looks.
    Deriving it from ``sparse.py`` — the module the harness names — would have found nothing, which
    is the same indirection that let ``model_id`` look plausible in the first place.
    """
    provider = (
        _REPO_ROOT / "packages" / "latence-splade" / "src" / "latence_splade" / "provider.py"
    ).read_text(encoding="utf-8")
    keys = set(re.findall(r"""(?:cfg|config)\.get\(\s*["'](\w+)["']""", provider))
    assert "model" in keys, (
        "SpladeCompositionEmbedder no longer reads a 'model' config key; this gate's derivation is "
        f"stale (found {sorted(keys)}). Fix the derivation, do not hard-code the key."
    )
    assert "model_id" not in keys, (
        "SpladeCompositionEmbedder now reads 'model_id' too — the defect this gate exists for has "
        "become legal, so the gate must be revisited rather than left asserting the old key."
    )
    return "model"


@pytest.mark.parametrize("harness", ["run_benchmark.py", "build_questions.py"])
def test_the_harness_passes_the_splade_key_the_provider_reads(harness: str) -> None:
    """A wrong key is accepted silently — the Provider does not reject unknown config keys.

    So the SPLADE leg runs, resolves its checkpoint to ``""``, and reports success. There is no
    error to notice; the only symptom is a sparse signal that quietly contributes nothing. That is
    why this is asserted structurally rather than left to a runtime check.
    """
    expected = _splade_config_key()
    text = _source(harness)

    blocks = re.findall(r"""query_sparse\.splade["']\s*,\s*["']config["']\s*:\s*\{(.*?)\}""", text)
    if not blocks:
        blocks = re.findall(r"""query_sparse\.splade.*?config["']\s*:\s*\{(.*?)\}""", text, re.S)
    assert blocks, f"no `query_sparse.splade` config literal found in {harness}"

    for block in blocks:
        keys = re.findall(r"""["'](\w+)["']\s*:""", block)
        assert expected in keys, (
            f"{harness} configures query_sparse.splade with {keys}, but the Provider reads "
            f"{expected!r}. An unknown key is NOT rejected — the checkpoint silently resolves to "
            "the empty string and the SPLADE leg contributes nothing while reporting success."
        )


# --- 3. sparse_terms is a mapping, not a SparseVector -------------------------------------


def test_sparse_terms_is_annotated_as_a_mapping_not_a_vector() -> None:
    """Pin the contract this gate derives from, so a widened annotation is noticed here."""
    from latence_retrieval.contracts import RetrievalQuery  # noqa: PLC0415

    annotation = typing.get_type_hints(RetrievalQuery)["sparse_terms"]
    assert "dict" in str(annotation), (
        f"RetrievalQuery.sparse_terms is now {annotation!r}; if it accepts a SparseVector directly "
        "this gate is obsolete and should be deleted rather than left passing vacuously."
    )


@pytest.mark.parametrize("harness", ["run_benchmark.py", "build_questions.py"])
def test_the_harness_converts_the_sparse_vector_before_the_query(harness: str) -> None:
    """``sparse_terms=`` must never be handed the encoder's return value unconverted.

    :meth:`SpladeQueryEncoder.encode` returns a :class:`SparseVector` (parallel ``indices`` /
    ``values``); the query contract wants ``dict[str, float]``. Passing the vector through is a type
    error that only fires on the pod, because nothing offline ever constructs a real encoder.
    """
    tree = ast.parse(_source(harness))
    offenders: list[str] = []

    def _is_converted(node: ast.expr) -> bool:
        """True when the expression is wrapped in the shared converter."""
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "sparse_vector_to_terms"
        )

    def _flag(node: ast.expr, where: str) -> None:
        """An encoder result reaching the query unconverted, however it was routed."""
        if _is_converted(node):
            return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"encode", "__call__"}:
                offenders.append(f"line {node.lineno}: {where} <- <encoder>.{node.func.attr}(...)")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) is False:
            return

    # (a) the keyword handed straight to the query
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "sparse_terms":
                    _flag(kw.value, "sparse_terms=")

    # (b) the assignment that BINDS the name — the shape the first version of this gate missed,
    #     because both harnesses assign to a local and pass the local, so checking only the
    #     keyword site was vacuous. Derive the binding too.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "sparse_terms":
                _flag(node.value, "sparse_terms =")

    assert not offenders, (
        f"{harness} routes an encoder result into sparse_terms unconverted: {offenders}. "
        "SpladeQueryEncoder.encode returns a SparseVector; RetrievalQuery.sparse_terms is "
        "dict[str, float]. Wrap it in sparse_vector_to_terms()."
    )


def test_the_shared_converter_round_trips_a_sparse_vector() -> None:
    """The one conversion both harnesses must use — indices become keys, values stay floats."""
    from sparse_terms import sparse_vector_to_terms  # noqa: PLC0415

    vector = SparseVector(indices=(3, 17, 90), values=(0.5, 1.25, 0.125))
    terms = sparse_vector_to_terms(vector)

    assert terms == {"3": 0.5, "17": 1.25, "90": 0.125}
    assert all(isinstance(k, str) for k in terms)
    assert all(isinstance(v, float) for v in terms.values())


def test_the_shared_converter_maps_an_empty_vector_to_an_empty_mapping() -> None:
    """A zero vector is the empty SparseVector, and must not become a run of zeros."""
    from sparse_terms import sparse_vector_to_terms  # noqa: PLC0415

    assert sparse_vector_to_terms(SparseVector()) == {}


def test_the_shared_converter_accepts_a_mapping_unchanged() -> None:
    """Idempotent, so a caller that already converted cannot double-convert into nonsense."""
    from sparse_terms import sparse_vector_to_terms  # noqa: PLC0415

    already = {"3": 0.5}
    assert sparse_vector_to_terms(already) == already
