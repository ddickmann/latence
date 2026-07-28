"""``docs/RETRIEVAL-ENGINE-INPUT-SPEC.md`` must name the versions the code actually stamps.

That document is the contract a THIRD-PARTY retrieval engine integrates against: it tells an
outside team which ``schema_version`` to expect on every record, and therefore which records to
accept. It sat at **15** through v16, v17, v18 and v19 — four contract revisions, one of them
(v19) BREAKING — while every record the pipeline emitted stamped the current
:data:`~latence_core.contracts.SCHEMA_VERSION`. An integrator who validated against the published
number would have rejected every record the pipeline produced.

The repo already gates its other operator-facing surfaces this way
(``test_blackwell_flashinfer_docs_truth``, ``test_tutorial_sota_stack_truth``,
``test_adr_page_map_carriage_currency``). This one was simply never written — which is exactly why
this document, and only this document, drifted four versions.

Anti-tautology: nothing here hard-codes a version. Both sides are read at run time — the expected
values by IMPORTING the constants from the modules that define them, the actual values by PARSING
the committed markdown. Bump ``SCHEMA_VERSION`` and this test fails until the document follows.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from latence_core.contracts import SCHEMA_VERSION
from latence_core.pipeline import PIPELINE_SCHEMA_VERSION
from latence_core.quality import QUALITY_SCHEMA_VERSION

# packages/latence-core/tests/ -> repo root is three parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SPEC_PATH = _REPO_ROOT / "docs" / "RETRIEVAL-ENGINE-INPUT-SPEC.md"

#: The § "Version constants" table maps a constant NAME to the version the document claims it
#: holds. Keyed by the name as it appears in the doc; valued by the constant actually imported.
_CONSTANTS: dict[str, int] = {
    "SCHEMA_VERSION": SCHEMA_VERSION,
    "QUALITY_SCHEMA_VERSION": QUALITY_SCHEMA_VERSION,
    "PIPELINE_SCHEMA_VERSION": PIPELINE_SCHEMA_VERSION,
}

#: A row of the version-constants table: ``| `NAME` (…) | **N** | `path` |``.
_DECLARED_ROW = re.compile(r"^\|\s*`(\w+)`[^|]*\|\s*\*\*(\d+)\*\*\s*\|", re.M)

#: A ``schema_version`` value inside the document's JSON record examples.
_JSON_VALUE = re.compile(r'"schema_version"\s*:\s*(\d+)')

#: A ``schema_version`` row in a per-contract field table, whose last cell states the version —
#: either bare (``| 19. |``) or in prose (``| Contract version (19). |``).
_TABLE_VALUE = re.compile(r"^\|\s*`schema_version`\s*\|[^|]*\|[^|]*\|[^|]*?(\d+)\.?\s*\|", re.M)


@pytest.fixture(scope="module")
def spec_text() -> str:
    """The committed spec — the single source both assertions read."""
    return _SPEC_PATH.read_text(encoding="utf-8")


def test_the_version_constants_table_matches_the_imported_constants(spec_text: str) -> None:
    """Every ``**N**`` in the version table equals the constant that module actually defines."""
    declared = {name: int(value) for name, value in _DECLARED_ROW.findall(spec_text)}

    missing = sorted(set(_CONSTANTS) - set(declared))
    assert not missing, (
        f"{_SPEC_PATH.name} § 'Version constants' no longer declares {missing}. The table is the "
        "one place an integrator reads the contract version from — a constant that vanishes from "
        "it is not documented, and this gate stops being able to check it."
    )

    wrong = {
        name: (declared[name], expected)
        for name, expected in _CONSTANTS.items()
        if declared[name] != expected
    }
    assert not wrong, (
        f"{_SPEC_PATH.name} declares stale version constant(s): "
        + "; ".join(f"{n} says {got}, code stamps {exp}" for n, (got, exp) in sorted(wrong.items()))
        + ". A third-party engine validating against the published number would reject every "
        "record this pipeline emits. Update the document in the same change as the bump."
    )


def test_no_record_example_or_field_table_names_a_stale_contract_version(spec_text: str) -> None:
    """Each ``schema_version`` the doc SHOWS is a version the code can actually stamp.

    Every contract's ``schema_version`` field defaults to the same module-level
    :data:`~latence_core.contracts.SCHEMA_VERSION`, so a record dumped today carries that value in
    every nested object. The Quality Report versions independently, so its own number is the one
    other value a reader may legitimately meet. Anything else is a fossil of an older revision —
    which is precisely the shape the four-version drift took: a JSON example whose every nested
    record still read ``15``.
    """
    allowed = {SCHEMA_VERSION, QUALITY_SCHEMA_VERSION}
    shown = {int(v) for v in _JSON_VALUE.findall(spec_text)}
    shown |= {int(v) for v in _TABLE_VALUE.findall(spec_text)}

    assert shown, (
        f"{_SPEC_PATH.name} shows no `schema_version` value at all — either the record examples "
        "and field tables were removed, or the patterns this gate scans with have gone stale. "
        "Either way it is no longer checking anything."
    )

    stale = sorted(shown - allowed)
    assert not stale, (
        f"{_SPEC_PATH.name} shows stale `schema_version` value(s) {stale}; a record this pipeline "
        f"emits carries {SCHEMA_VERSION} (or {QUALITY_SCHEMA_VERSION} for the Quality Report). "
        "An integrator copying the example builds a validator that rejects real output."
    )
