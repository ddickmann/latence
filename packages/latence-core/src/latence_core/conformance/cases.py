"""Per-Capability conformance cases — the fixture registry (P3-F3 §1).

Providers span heterogeneous Capabilities, so a single canonical input won't do. This module
maps **each Capability** to a :class:`ConformanceCase`: a canonical valid input, the config a
Provider of that Capability needs, an **invoke** that runs the Provider and returns its produced
records, an **output-validity predicate** asserting the Capability's typed-record contract
(Provenance + Classification + in-range offsets, reusing the S10 completeness + ADR-0031 offset
seams — not a reinvented weaker check), and an **adversarial input** with its **expected graceful
outcome** (a typed :class:`~latence_core.errors.LatenceError`, or a Screening quarantine /
PARSE_ERROR disposition where the Capability defines one — never a bare crash).

A Provider is mapped to its Capability by **name prefix** (``entity.gliner`` → ENTITY), so a new
Provider is auto-covered the moment its Capability has a case. A Provider whose Capability has
**no** case must FAIL the suite loudly (:func:`case_for_provider` raises) rather than silently
skip — no silent coverage gaps (the issue's explicit requirement).

Endpoint / GPU Providers are ``stub_scoped``: the invoke runs them against a deterministic stub
(:mod:`~latence_core.conformance.stub`) so the contract/graceful-failure/license/determinism
subset is checked without a live backend (their real numbers need a rig, #64).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from latence_core.conformance import fixtures
from latence_core.conformance import stub as stubs
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    Disposition,
    DocumentRecord,
    ParserInput,
    Provenance,
    Record,
)
from latence_core.errors import LatenceError
from latence_core.pagemap import PageOffsetIndex
from latence_core.storage import Storage


class Capability(StrEnum):
    """The Capabilities the conformance suite parametrizes over (CONTEXT ``Capability``).

    Named to match the Provider entry-point name **prefix** (``<capability>.<variant>``) so
    :func:`capability_of` derives a Provider's Capability from its registered name.
    """

    SOURCE = "source"
    PARSE = "parse"
    CHUNK = "chunk"
    INTAKE_SCREENING = "intake_screening"
    CONTENT_SCREENING = "content_screening"
    SCHEMA_INDUCTION = "schema_induction"
    ENTITY = "entity"
    RELATION = "relation"
    REDACTION = "redaction"
    PROFILING = "profiling"
    DISAMBIGUATION = "disambiguation"
    GRAPH = "graph"
    GRAPH_COMPLETION = "graph_completion"
    GRAPH_FEATURES = "graph_features"
    CONTEXT_ENRICHMENT = "context_enrichment"
    EXPORT = "export"
    EMBEDDING = "embedding"
    SPARSE_EMBEDDING = "sparse"
    MULTIVECTOR_EMBEDDING = "multivector"
    FDE = "fde"
    TOKENIZER = "tokenizer"
    DELTA = "delta"


# Map a Provider entry-point **name prefix** to its Capability. The prefix is the substring
# before the first ``.`` (``entity.gliner`` → ``entity``), EXCEPT where two Screening
# Capabilities share the ``screening`` prefix (``screening.intake_*`` vs ``screening.content_*``)
# and where ``parser.*`` and ``parse.*`` both name the Parse Capability. Those are handled by
# :func:`capability_of`, which reads the second segment for Screening.
_PREFIX_TO_CAPABILITY: dict[str, Capability] = {
    "source": Capability.SOURCE,
    "parse": Capability.PARSE,
    "parser": Capability.PARSE,
    "chunk": Capability.CHUNK,
    "label_inducer": Capability.SCHEMA_INDUCTION,
    "entity": Capability.ENTITY,
    "relation": Capability.RELATION,
    # A FUSED Entity+Relation provider (extract_fused) is checked against the RELATION case (the
    # fused branch drives extract_fused). ``relation.gliner_relex`` uses the ``relation`` prefix for
    # back-compat; ``fused_entity_relation.gliner2`` (W9) names the fused Capability explicitly.
    "fused_entity_relation": Capability.RELATION,
    "redaction": Capability.REDACTION,
    "profiling": Capability.PROFILING,
    "disambiguation": Capability.DISAMBIGUATION,
    "graph": Capability.GRAPH,
    "graph_completion": Capability.GRAPH_COMPLETION,
    "graph_features": Capability.GRAPH_FEATURES,
    "context": Capability.CONTEXT_ENRICHMENT,
    "export": Capability.EXPORT,
    "embedding": Capability.EMBEDDING,
    # Index-time retrieval-signal generators (ADR-0049/0050): siblings of the dense Embedder under
    # the same latence.providers seam.
    "sparse": Capability.SPARSE_EMBEDDING,
    "multivector": Capability.MULTIVECTOR_EMBEDDING,
    "fde": Capability.FDE,
    # #164: the Tokenizer seam of the BM25 term-statistics exporter (ADR-0048/0049).
    "tokenizer": Capability.TOKENIZER,
    "delta": Capability.DELTA,
}


def capability_of(provider_name: str) -> Capability:
    """Derive a Provider's Capability from its registered ``<capability>.<variant>`` name.

    The prefix (before the first ``.``) names the Capability, with two disambiguations:
    ``screening.intake_*`` → Intake Screening and ``screening.content_*`` → Content Screening
    (both share the ``screening`` prefix), and ``export.knowledge_graph``/``export.demo_site``/
    ``export.jsonl_parquet`` all → Export. A name whose prefix names no known Capability raises
    ``KeyError`` — a Provider the suite cannot classify is a loud failure, not a silent skip.
    """
    prefix, _, rest = provider_name.partition(".")
    if prefix == "screening":
        if rest.startswith("intake"):
            return Capability.INTAKE_SCREENING
        if rest.startswith("content"):
            return Capability.CONTENT_SCREENING
        msg = f"Screening Provider {provider_name!r} is neither intake_* nor content_*."
        raise KeyError(msg)
    try:
        return _PREFIX_TO_CAPABILITY[prefix]
    except KeyError as exc:
        msg = (
            f"Provider {provider_name!r} has an unknown Capability prefix {prefix!r}; "
            "add it to _PREFIX_TO_CAPABILITY (or write a ConformanceCase) — a Provider the "
            "conformance suite cannot classify is a loud failure, never a silent skip."
        )
        raise KeyError(msg) from exc


# The invoke signature: given a constructed Provider instance, run it and return the records it
# produced (a flat list of inter-Stage Records). Screening returns its passed records; Export
# returns the URIs-as-strings check is done separately (see ExportCase). A stub-scoped invoke
# installs the stub itself. Typed ``Sequence[object]`` because a few Capabilities produce
# non-Record outputs that C1/C4 still must audit (Embedding yields raw ``list[float]`` vectors,
# not Records) — the harness serializes whatever is returned for the C4 byte-comparison.
InvokeFn = Callable[[Any], "Sequence[object]"]
# The output-validity predicate: given the produced records, return True iff they satisfy the
# Capability's typed contract (or are a legitimately-empty result the case allows).
ValidityFn = Callable[["Sequence[object]"], bool]
# The adversarial invoke: run the Provider on a malformed/oversized/injection/empty input. It
# must EITHER raise a typed LatenceError, OR return an :class:`AdversarialOutcome` describing a
# graceful non-raising path (a Screening quarantine, a PARSE_ERROR disposition).
AdversarialFn = Callable[[Any], "AdversarialOutcome"]


@dataclass(frozen=True)
class AdversarialOutcome:
    """The non-raising graceful outcome of an adversarial input (C2's alternative to a raise).

    A Capability that does not *raise* on bad input still must handle it gracefully, not emit
    silent-wrong output: Screening **quarantines** the record (``quarantined=True``), a Parser
    emits a **PARSE_ERROR** record (``parse_error=True``), or a Provider legitimately yields **no
    records** for an empty/degenerate input (``empty=True``). ``text`` is any string the Provider
    surfaced (a quarantine reason, an error message) that the C2 no-PII assertion scans — it must
    never contain the adversarial payload's PII.
    """

    raised: bool = False
    quarantined: bool = False
    parse_error: bool = False
    empty: bool = False
    noop: bool = False
    text: str = ""

    @property
    def graceful(self) -> bool:
        """True if the Provider took a defined graceful path rather than a crash/hang.

        Any defined non-crashing path counts: it **raised a typed LatenceError**, it
        **Quarantined** the record, it emitted a **PARSE_ERROR**, it returned **no** records, or
        it returned a valid result without crashing on the degenerate input (``noop`` — e.g. a
        corpus Profiler over an empty corpus still emits its one corpus record). C2's job is to
        forbid a *bare* crash / hang / silent-wrong output, which none of these are.
        """
        return self.raised or self.quarantined or self.parse_error or self.empty or self.noop


@dataclass(frozen=True)
class ConformanceCase:
    """The fixed contract one Capability's Providers are checked against (C1–C6 inputs)."""

    capability: Capability
    #: Config a Provider of this Capability needs to construct + run (zero-shot labels, etc.).
    config: dict[str, Any]
    #: Run the (valid-config) Provider on the canonical input; return produced records.
    invoke: InvokeFn
    #: Assert the produced records satisfy the Capability's typed contract.
    valid_output: ValidityFn
    #: Run the Provider on the adversarial input; must raise LatenceError or return a graceful path.
    adversarial: AdversarialFn
    #: True when C1 legitimately produces no records (documented per case — e.g. a Source over an
    #: empty folder). Never a blanket escape hatch; only where the Capability defines "none".
    empty_ok: bool = False
    #: True for endpoint/GPU Providers checked against a stub; the harness flags "needs a rig".
    stub_scoped: bool = False


# ---------------------------------------------------------------------------
# Shared output-validity helpers (reuse the S10 completeness + ADR-0031 offset seams)
# ---------------------------------------------------------------------------


class _HasProvenance(Protocol):
    """A produced item the C1 audit inspects — anything carrying Provenance + Classification.

    Most producers emit :class:`Record`s, but a Source emits raw :class:`ParserInput`s (which are
    NOT Records yet still carry Provenance + Classification), so the C1 audit is duck-typed on the
    boundary contract rather than a ``Record`` isinstance — the same items the original check ran
    over. Non-provenance-carrying outputs (an Embedder's raw ``list[float]`` vectors) are filtered
    out; those Capabilities validate via their own ``valid_output`` predicate instead.
    """

    provenance: Any
    classification: Any


def _as_records(items: Sequence[object]) -> list[_HasProvenance]:
    """Narrow a produced sequence to Provenance-carrying items (Records or ParserInputs)."""
    return [r for r in items if hasattr(r, "provenance") and hasattr(r, "classification")]


def _provenance_and_classification_present(records: Sequence[_HasProvenance]) -> bool:
    """Every record carries Provenance + Classification (the enforced boundary contract)."""
    return all(r.provenance is not None and r.classification is not None for r in records)


def _offsets_in_range(records: Sequence[_HasProvenance]) -> bool:
    """Every offset-bearing record has a well-ordered char span, in-range against its page map.

    Reuses the S10 completeness invariant (``char_end >= char_start``, page span well-ordered)
    AND the ADR-0031 upper bound: an offset must not run past the assembled text. The canonical
    document's :class:`PageMap` gives the document length (``total_chars``); a char span whose
    ``char_end`` exceeds it is an out-of-range mention (a real regression), not just a
    well-ordered one. This is the "offsets present and IN-RANGE" half of C1 — a bounded check,
    not merely a well-ordering one.
    """
    total_chars = _canonical_total_chars()
    for r in records:
        prov = r.provenance
        if prov.char_start is None and prov.char_end is None:
            continue
        if prov.char_start is None or prov.char_end is None or prov.char_end < prov.char_start:
            return False
        # Upper bound: the half-open span must lie within the assembled text. A negative start or
        # an end past the document length is an out-of-range offset (S10/ADR-0031 in-range half).
        if prov.char_start < 0 or prov.char_end > total_chars:
            return False
        if (
            prov.page_start is not None
            and prov.page_end is not None
            and prov.page_end < prov.page_start
        ):
            return False
    return True


def _canonical_total_chars() -> int:
    """The assembled length of the canonical document (its page map's ``total_chars``)."""
    page_map = fixtures.canonical_corpus().documents[0].page_map
    if page_map is None:  # pragma: no cover - the canonical doc always has a page map
        return len(fixtures.CANONICAL_TEXT)
    return page_map.total_chars


def _records_contract_valid(records: Sequence[object], *, empty_ok: bool) -> bool:
    """C1: non-empty (unless empty_ok), Provenance/Classification present, offsets in-range."""
    recs = _as_records(records)
    if not recs:
        return empty_ok
    return _provenance_and_classification_present(recs) and _offsets_in_range(recs)


def _page_offsets_resolve(records: Sequence[object]) -> bool:
    """A stronger offset audit for sub-document carriers: the char span resolves with ZERO drift.

    For an EntityMention/RelationMention the char span must resolve back to a source page through
    the parent's page map (the ADR-0031/S10 round-trip). :meth:`PageOffsetIndex.page_for_offset`
    is deliberately non-raising — it clamps an out-of-range offset and *records the drift* rather
    than crashing — so the in-range assertion is the drift diagnostic, not a caught exception: a
    faithful mention resolves EXACTLY (``max_drift_chars == 0``); an out-of-bounds mention drifts.
    We reuse the same :class:`PageOffsetIndex` the Runner's drift audit uses.
    """
    page_map = fixtures.canonical_corpus().documents[0].page_map
    if page_map is None:  # pragma: no cover - the canonical doc always has a page map
        return True
    index = PageOffsetIndex(page_map)
    for r in _as_records(records):
        prov = r.provenance
        if prov.char_start is None or prov.char_end is None:
            continue
        # Resolving never raises; an out-of-range offset clamps + records drift. A faithful
        # mention resolves with zero drift — the ADR-0031 round-trip invariant.
        index.page_for_offset(prov.char_start)
        index.page_for_offset(max(prov.char_end - 1, prov.char_start))
    return index.diagnostics.max_drift_chars == 0


# ---------------------------------------------------------------------------
# Adversarial inputs
# ---------------------------------------------------------------------------

# A malformed raw input: bytes that are not decodable text, oversized, or empty, depending on the
# Capability. The PII-bearing text payload proves the C2 no-PII-in-error assertion.
_ADVERSARIAL_PROVENANCE = Provenance(
    source_uri="mem://conformance/adversarial.bin",
    file_name="adversarial.bin",
    file_type="bin",
    file_size=8,
    document_id="sha256:conformance-adversarial",
)
_ADVERSARIAL_CLASSIFICATION = Classification(language="und", sensitivity="confidential")


def _adversarial_parser_input(content: bytes | str) -> ParserInput:
    return ParserInput(
        provenance=_ADVERSARIAL_PROVENANCE,
        classification=_ADVERSARIAL_CLASSIFICATION,
        content=content,
    )


def _pii_document() -> DocumentRecord:
    """A DocumentRecord whose content is the PII-bearing adversarial payload (for RE)."""
    from latence_core.contracts import PageMap

    text = fixtures.PII_ADVERSARIAL_TEXT
    assembly = PageMap.from_page_texts([text])
    prov = _ADVERSARIAL_PROVENANCE.model_copy(update={"char_start": 0, "char_end": len(text)})
    return DocumentRecord(
        record_id="sha256:conformance-adversarial:doc",
        provenance=prov,
        classification=_ADVERSARIAL_CLASSIFICATION,
        content=assembly.text,
        page_map=assembly.page_map,
        disposition=Disposition.PARSED,
    )


def _pii_chunk() -> ChunkRecord:
    """A ChunkRecord whose content is the PII-bearing adversarial payload (chunk-level Redaction).

    W16 (ADR-0042): Redaction is a chunk→chunk transform, so its adversarial input is a chunk
    carrying the PII payload. A Redactor must handle it gracefully (mask + return the chunk, never
    a crash, never PII in any raised error). Built with a single-page page_map + identity
    offset_map so a chunk-local span still resolves to a page (the shared chunk page-resolution
    seam).
    """
    from latence_core.contracts import OffsetMap, OffsetSegment, PageMap

    text = fixtures.PII_ADVERSARIAL_TEXT
    assembly = PageMap.from_page_texts([text])
    prov = _ADVERSARIAL_PROVENANCE.model_copy(update={"char_start": 0, "char_end": len(text)})
    return ChunkRecord(
        record_id="sha256:conformance-adversarial:chunk:0",
        document_record_id="sha256:conformance-adversarial:doc",
        index=0,
        provenance=prov,
        classification=_ADVERSARIAL_CLASSIFICATION,
        content=text,
        token_count=len(text.split()),
        page_map=assembly.page_map,
        offset_map=OffsetMap(
            segments=[OffsetSegment(local_start=0, doc_start=0)], length=len(text)
        ),
    )


def _capture_raise(fn: Callable[[], Any]) -> AdversarialOutcome:
    """Run ``fn``; a raised LatenceError is a graceful typed failure (no PII in the message).

    A raised **LatenceError** is the expected typed failure (recorded ``raised=True`` with the
    message for the C2 no-PII scan). A raised NON-LatenceError is re-raised so the harness records
    it as a C2 failure (an untyped crash — the thing C2 forbids). A non-raising return is classified
    for its graceful path (quarantine / parse-error / empty / noop).
    """
    try:
        result = fn()
    except LatenceError as exc:
        return AdversarialOutcome(raised=True, text=str(exc))
    # It did not raise — inspect the result for a graceful non-raising path.
    return _classify_nonraising(result)


def _classify_nonraising(result: Any) -> AdversarialOutcome:
    """Classify a non-raising adversarial result as graceful (quarantine/parse-error/empty/noop)."""
    from latence_core.capability import ScreeningOutcome

    if isinstance(result, ScreeningOutcome):
        reasons = " ".join(f.reason for f in result.findings)
        return AdversarialOutcome(
            quarantined=bool(result.quarantined),
            noop=not result.quarantined,
            text=reasons,
        )
    records = list(result) if result is not None else []
    if any(
        isinstance(r, DocumentRecord) and r.disposition is Disposition.PARSE_ERROR
        for r in records
    ):
        reasons = " ".join(r.error or "" for r in records if isinstance(r, DocumentRecord))
        return AdversarialOutcome(parse_error=True, text=reasons)
    # Returned without crashing: empty is the classic defined "none"; a non-empty valid return on
    # a degenerate input (a corpus Profiler's single corpus record) is a graceful no-op.
    if not records:
        return AdversarialOutcome(empty=True)
    return AdversarialOutcome(noop=True)


# ---------------------------------------------------------------------------
# The per-Capability case registry
# ---------------------------------------------------------------------------


def _source_case() -> ConformanceCase:
    """Source: read a tiny folder; an empty folder produces no records (graceful empty).

    A Source reads a real folder on Storage, so this case materialises a tiny temp folder on the
    local (``file://``) filesystem — the natural fixture — with one canonical document, and points
    the Provider's ``path`` config at it. Because the Provider's ``path`` is fixed at construction,
    the case builds the folder AND supplies its ``config`` together (via :func:`_source_config`).
    """
    import tempfile
    from pathlib import Path

    def _prepare_folder() -> str:
        tmp = Path(tempfile.mkdtemp(prefix="latence-conformance-source-"))
        (tmp / "canonical.txt").write_text(fixtures.CANONICAL_TEXT, encoding="utf-8")
        return str(tmp)

    folder = _prepare_folder()

    def invoke(provider: Any) -> Sequence[Record]:
        return list(provider.produce(Storage.for_uri(f"file://{folder}")))

    def valid_output(records: Sequence[object]) -> bool:
        return _records_contract_valid(records, empty_ok=False)

    def adversarial(provider: Any) -> AdversarialOutcome:
        # An empty folder: the Source produces no records (graceful empty), never a crash. Build a
        # fresh Provider pointed at an empty temp folder.
        empty = tempfile.mkdtemp(prefix="latence-conformance-source-empty-")
        empty_provider = type(provider)({"path": empty})
        return _capture_raise(
            lambda: list(empty_provider.produce(Storage.for_uri(f"file://{empty}")))
        )

    return ConformanceCase(
        capability=Capability.SOURCE,
        config={"path": folder},
        invoke=invoke,
        valid_output=valid_output,
        adversarial=adversarial,
    )


def _sharepoint_source_case() -> ConformanceCase:
    """Source (``source.sharepoint``, #182): list + fetch a fake SharePoint drive over Graph.

    A source *system* connector cannot read a temp folder, so the base Source case does not apply.
    It runs instead against the faithful offline Microsoft Graph fake the connector package ships
    (``latence_source_sharepoint.testing``) — the same fake that rejects a missing/expired token,
    a query-string token, ``$skip`` paging, an unknown drive, and a credentialed request to the
    pre-authenticated download host. ``stub_scoped=True``, so the harness flags this Provider
    "verified against a stub; live numbers need a tenant" and nobody mistakes it for a live run.

    The adversarial input is an empty drive: the Source produces no records (the Capability's
    defined "none"), never a crash.
    """
    import os

    token_env = "LATENCE_CONFORMANCE_SHAREPOINT_TOKEN"

    @contextlib.contextmanager
    def _token_in_env() -> Iterator[None]:
        """Secrets are environment-only, so the case supplies the fake's token that way too."""
        previous = os.environ.get(token_env)
        os.environ[token_env] = "valid-token"
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(token_env, None)
            else:
                os.environ[token_env] = previous

    def _fake_drive(*, empty: bool) -> dict[str, Any]:
        from latence_source_sharepoint.testing import (
            GRAPH_HOST,
            FakeFile,
            FakeGraphApi,
            FakeGraphTransport,
        )

        files = (
            []
            if empty
            else [FakeFile(path="/canonical.txt", content=fixtures.CANONICAL_TEXT.encode("utf-8"))]
        )
        api = FakeGraphApi(files=files)
        return {
            "drive_id": api.drive_id,
            "graph_base_url": GRAPH_HOST,
            "transport": FakeGraphTransport(api),
            "auth": {"mode": "env_token", "token_env": token_env},
        }

    def invoke(provider: Any) -> Sequence[Record]:
        # The Provider under test was constructed from ``config`` (a populated fake drive).
        with _token_in_env():
            return list(provider.produce(Storage.for_uri("file:///tmp")))

    def valid_output(records: Sequence[object]) -> bool:
        if not _records_contract_valid(records, empty_ok=False):
            return False
        # A source-system connector must stamp the source system + a resolvable identity.
        return all(
            r.provenance.source_system == "sharepoint"
            and r.provenance.source_uri.startswith("sharepoint://")
            for r in _as_records(records)
        )

    def adversarial(provider: Any) -> AdversarialOutcome:
        empty_provider = type(provider)(_fake_drive(empty=True))
        with _token_in_env():
            return _capture_raise(
                lambda: list(empty_provider.produce(Storage.for_uri("file:///tmp")))
            )

    # Constructed once so C4's two runs share the same fake drive (a deterministic Source over
    # an unchanged drive must produce byte-identical output).
    config = _fake_drive(empty=False)
    return ConformanceCase(
        capability=Capability.SOURCE,
        config=config,
        invoke=invoke,
        valid_output=valid_output,
        adversarial=adversarial,
        stub_scoped=True,
    )


def _parse_case(*, stub_scoped: bool) -> ConformanceCase:
    """Parse: decode a ParserInput; an undecodable input yields a PARSE_ERROR (graceful)."""

    def invoke(provider: Any) -> Sequence[Record]:
        return list(provider.parse([fixtures.canonical_parser_input()]))

    def valid_output(records: Sequence[object]) -> bool:
        if not _records_contract_valid(records, empty_ok=False):
            return False
        # A Parser must emit PARSED DocumentRecords carrying a page map for the canonical input.
        return all(
            isinstance(r, DocumentRecord)
            and r.disposition is Disposition.PARSED
            and r.page_map is not None
            for r in records
        )

    def adversarial(provider: Any) -> AdversarialOutcome:
        # Undecodable bytes with a bad encoding hint → PARSE_ERROR record, never a crash.
        bad = _adversarial_parser_input(b"\xff\xfe\x00\x01\x02broken")
        bad = bad.model_copy(update={"config": {"encoding": "not-a-real-codec"}})
        return _capture_raise(lambda: list(provider.parse([bad])))

    if stub_scoped:
        base_invoke = invoke

        def invoke(provider: Any) -> Sequence[Record]:
            with stubs.install_openai(reply=stubs.parse_chat_reply()):
                return base_invoke(provider)

        def adversarial(provider: Any) -> AdversarialOutcome:
            # A terminal endpoint failure (empty reply → the endpoint parser degrades to
            # PARSE_ERROR); the stub returns a non-page-delimited empty transcription.
            with stubs.install_openai(reply=""):
                bad = _adversarial_parser_input(b"\xff\xfe\x00\x01\x02broken")
                return _capture_raise(lambda: list(provider.parse([bad])))

    return ConformanceCase(
        capability=Capability.PARSE,
        config={"base_url": "http://stub/v1"} if stub_scoped else {},
        invoke=invoke,
        valid_output=valid_output,
        adversarial=adversarial,
        stub_scoped=stub_scoped,
    )


def _render_case() -> ConformanceCase:
    """Parse (parser.render, W5): render a fake PDF → OCR via a stub → PARSED with a page map.

    ``parser.render`` (compute="either") RUNS through conformance on a CPU host (not
    device-skipped). Its native rasterisers are faked offline via
    :func:`stubs.install_render_offline` (a faithful fake ``pypdfium2``/``PIL`` so the REAL render
    code path runs), and its OCR is delegated to a stub
    (:class:`stubs.StubRenderOcr`) wired through the Provider's ``ocr_registry`` config seam. The
    canonical fixture text is rendered as ONE page so the produced offsets match the canonical page
    map — a genuine C1 offset check. C2's adversarial input is an unsupported ``.txt`` (a page with
    no renderer path) → a graceful PARSE_ERROR (no PII leaked).
    """
    canonical_pdf = stubs.render_fake_pdf([fixtures.CANONICAL_TEXT])
    config = {"ocr": {"provider": "parser.lighton"}, "ocr_registry": stubs.render_stub_registry()}

    def _canonical_input() -> ParserInput:
        prov = fixtures.canonical_provenance().model_copy(
            update={"file_name": "canonical.pdf", "file_type": "pdf"}
        )
        return ParserInput(
            provenance=prov,
            classification=fixtures.canonical_classification(),
            content=canonical_pdf,
        )

    def invoke(provider: Any) -> Sequence[Record]:
        with stubs.install_render_offline():
            return list(provider.parse([_canonical_input()]))

    def valid_output(records: Sequence[object]) -> bool:
        if not _records_contract_valid(records, empty_ok=False):
            return False
        return all(
            isinstance(r, DocumentRecord)
            and r.disposition is Disposition.PARSED
            and r.page_map is not None
            for r in records
        )

    def adversarial(provider: Any) -> AdversarialOutcome:
        # An unsupported .txt file (no renderer path) → a graceful PARSE_ERROR record, not a crash.
        bad = _adversarial_parser_input(fixtures.PII_ADVERSARIAL_TEXT.encode("utf-8"))
        bad = bad.model_copy(
            update={
                "provenance": bad.provenance.model_copy(
                    update={"file_name": "adversarial.txt", "file_type": "txt"}
                )
            }
        )
        with stubs.install_render_offline():
            return _capture_raise(lambda: list(provider.parse([bad])))

    return ConformanceCase(
        capability=Capability.PARSE,
        config=config,
        invoke=invoke,
        valid_output=valid_output,
        adversarial=adversarial,
        stub_scoped=False,
    )


def _lighton_vllm_case() -> ConformanceCase:
    """Parse (parser.lighton_vllm, W10): rasterise a fake PDF → served vLLM OCR (stub) → PARSED.

    ``parser.lighton_vllm`` (compute="either") is **infra-gated** like ``parser.endpoint`` — it
    needs a running vLLM server — so this case is ``stub_scoped``: the async ``AsyncOpenAI`` stub
    (:func:`stubs.install_openai`) stands in for the served endpoint. But it also **rasterises** its
    input (unlike the raw-bytes ``parser.endpoint``), so its REAL shared-rasteriser code path runs
    offline against a faithful fake ``pypdfium2``/``PIL``
    (:func:`stubs.install_lighton_vllm_offline`): the fake PDF renders to a fake page image whose
    ``.size``/``.save`` drive the real 1540px downscale + the base64 data-URL encode, and the stub
    vLLM transcribes it — the exact rasterise → POST → PageMap round-trip the Provider performs. A
    single fake page → one PARSED record with a page map (a genuine C1 offset check). C2's
    adversarial input is a type-spoofed ``.pdf`` (bytes are not a PDF) → a graceful PARSE_ERROR
    before any endpoint call.
    """
    canonical_pdf = stubs.lighton_vllm_fake_pdf([fixtures.CANONICAL_TEXT])
    config = {"base_url": "http://stub-vllm/v1"}

    def _canonical_input() -> ParserInput:
        prov = fixtures.canonical_provenance().model_copy(
            update={"file_name": "canonical.pdf", "file_type": "pdf"}
        )
        return ParserInput(
            provenance=prov,
            classification=fixtures.canonical_classification(),
            content=canonical_pdf,
        )

    def invoke(provider: Any) -> Sequence[Record]:
        # The stub vLLM transcribes each page image back to the canonical text so the produced
        # offsets match the canonical page map — a genuine C1 offset check, not a no-op.
        with (
            stubs.install_lighton_vllm_offline(),
            stubs.install_openai(reply=fixtures.CANONICAL_TEXT),
        ):
            return list(provider.parse([_canonical_input()]))

    def valid_output(records: Sequence[object]) -> bool:
        if not _records_contract_valid(records, empty_ok=False):
            return False
        return all(
            isinstance(r, DocumentRecord)
            and r.disposition is Disposition.PARSED
            and r.page_map is not None
            for r in records
        )

    def adversarial(provider: Any) -> AdversarialOutcome:
        # A type-spoofed .pdf (bytes are not a PDF) → a graceful PARSE_ERROR before any endpoint
        # call (the shared rasteriser's offset-0 spoof guard), no PII leaked.
        spoof = _adversarial_parser_input(fixtures.PII_ADVERSARIAL_TEXT.encode("utf-8"))
        spoof = spoof.model_copy(
            update={
                "provenance": spoof.provenance.model_copy(
                    update={"file_name": "adversarial.pdf", "file_type": "pdf"}
                )
            }
        )
        with (
            stubs.install_lighton_vllm_offline(),
            stubs.install_openai(reply=fixtures.CANONICAL_TEXT),
        ):
            return _capture_raise(lambda: list(provider.parse([spoof])))

    return ConformanceCase(
        capability=Capability.PARSE,
        config=config,
        invoke=invoke,
        valid_output=valid_output,
        adversarial=adversarial,
        stub_scoped=True,
    )


def _chunk_case() -> ConformanceCase:
    """Chunk: split the canonical documents; a PARSE_ERROR doc yields no chunks (graceful empty)."""

    def invoke(provider: Any) -> Sequence[Record]:
        return list(provider.chunk(fixtures.canonical_corpus().documents))

    def valid_output(records: Sequence[object]) -> bool:
        return _records_contract_valid(records, empty_ok=False)

    def adversarial(provider: Any) -> AdversarialOutcome:
        # A PARSE_ERROR document (no usable content) must yield no chunks — the defined skip.
        err_doc = DocumentRecord(
            record_id="err:doc",
            provenance=_ADVERSARIAL_PROVENANCE.model_copy(update={"char_start": 0, "char_end": 0}),
            classification=_ADVERSARIAL_CLASSIFICATION,
            content="",
            disposition=Disposition.PARSE_ERROR,
            error="unparseable",
        )
        return _capture_raise(lambda: list(provider.chunk([err_doc])))

    return ConformanceCase(
        capability=Capability.CHUNK,
        config={},
        invoke=invoke,
        valid_output=valid_output,
        adversarial=adversarial,
    )


def _intake_screening_case() -> ConformanceCase:
    """Intake Screening: partition raw inputs; a dangerous input is Quarantined (graceful)."""

    def invoke(provider: Any) -> Sequence[Record]:
        outcome = provider.screen_intake([fixtures.canonical_parser_input()])
        return list(outcome.passed)

    def valid_output(records: Sequence[object]) -> bool:
        return _records_contract_valid(records, empty_ok=True)

    def adversarial(provider: Any) -> AdversarialOutcome:
        # An oversized / zip-bomb-shaped input is Quarantined, not crashed. A giant declared
        # file size + suspicious content triggers the size/signature screener.
        big = _adversarial_parser_input(b"PK\x03\x04" + b"\x00" * 64)
        big = big.model_copy(
            update={
                "provenance": _ADVERSARIAL_PROVENANCE.model_copy(
                    update={"file_size": 10**12, "file_type": "zip", "file_name": "bomb.zip"}
                )
            }
        )
        return _capture_raise(lambda: provider.screen_intake([big]))

    return ConformanceCase(
        capability=Capability.INTAKE_SCREENING,
        config={},
        invoke=invoke,
        valid_output=valid_output,
        adversarial=adversarial,
        empty_ok=True,
    )


def _content_screening_case() -> ConformanceCase:
    """Content Screening: pass chunks (some flagged); an injection chunk is flagged/quarantined."""

    def invoke(provider: Any) -> Sequence[Record]:
        outcome = provider.screen_content(fixtures.canonical_corpus().chunks)
        return list(outcome.passed)

    def valid_output(records: Sequence[object]) -> bool:
        return _records_contract_valid(records, empty_ok=True)

    def adversarial(provider: Any) -> AdversarialOutcome:
        # A prompt-injection-bearing chunk: Content Screening flags or quarantines it, never a
        # crash. Build a chunk carrying the injection payload.
        from latence_core.contracts import ChunkRecord

        base = fixtures.canonical_corpus().chunks[0]
        injected = ChunkRecord(
            record_id="adv:chunk",
            document_record_id=base.document_record_id,
            index=0,
            provenance=base.provenance,
            classification=base.classification,
            content=fixtures.PII_ADVERSARIAL_TEXT,
            token_count=base.token_count,
            page_map=base.page_map,
        )
        return _capture_raise(lambda: provider.screen_content([injected]))

    return ConformanceCase(
        capability=Capability.CONTENT_SCREENING,
        config={},
        invoke=invoke,
        valid_output=valid_output,
        adversarial=adversarial,
        empty_ok=True,
    )


def _schema_induction_case() -> ConformanceCase:
    """Schema Induction (W4): induce a doc's labels + annotate its chunks; a bad reply fails open.

    Stub-scoped (an OpenAI-compatible LLM endpoint): the valid invoke installs the ``openai`` stub
    with a canned ``{entity,relation,pii}`` reply and asserts every returned chunk is the SAME chunk
    (1:1, order-stable) now carrying a populated
    :class:`~latence_core.contracts.InducedLabels`. The adversarial invoke installs a reply that is
    NOT valid JSON of the required shape (a schema-invalid safety-net trigger): the Provider FAILS
    OPEN — it returns the chunks unchanged with ``induced_labels`` still ``None`` and never crashes
    (the W4 fail-open-never-crash posture), a graceful no-op C2 accepts.
    """

    def invoke(provider: Any) -> Sequence[Record]:
        with stubs.install_openai(reply=stubs.schema_induction_reply()):
            return list(provider.induce(fixtures.canonical_corpus().chunks))

    def valid_output(records: Sequence[object]) -> bool:
        # C1 boundary contract (Provenance/Classification/offsets) — a chunk→chunk transform.
        if not _records_contract_valid(records, empty_ok=True):
            return False
        from latence_core.contracts import ChunkRecord, InducedLabels

        chunks = [r for r in records if isinstance(r, ChunkRecord)]
        if not chunks:
            return False
        # Every chunk carries the induced schema (the canned reply had non-empty slots), and the
        # model_id is stamped (audit) — the genuine annotation check, not a no-op.
        return all(
            isinstance(c.induced_labels, InducedLabels)
            and bool(c.induced_labels.entity_types)
            and bool(c.induced_labels.model_id)
            for c in chunks
        )

    def adversarial(provider: Any) -> AdversarialOutcome:
        # A schema-invalid reply (valid JSON, wrong shape): the Provider falls back to config labels
        # (induced_labels stays None) and returns the chunks unchanged — a graceful, non-crashing
        # no-op. NOT a raise: fail-open is the W4 requirement.
        with stubs.install_openai(reply='{"not_the_shape": [1, 2, 3]}'):
            return _capture_raise(lambda: list(provider.induce(fixtures.canonical_corpus().chunks)))

    return ConformanceCase(
        capability=Capability.SCHEMA_INDUCTION,
        config={"base_url": "http://stub/v1"},
        invoke=invoke,
        valid_output=valid_output,
        adversarial=adversarial,
        empty_ok=True,
        stub_scoped=True,
    )


def _entity_case(*, stub_scoped: bool, gliner: bool) -> ConformanceCase:
    """Entity Extraction: tag mentions in chunks; an empty chunk yields none (graceful empty)."""
    labels_dict = {
        "person": {"terms": ["Alice"]},
        "org": {"terms": ["Acme Corp"]},
    }
    labels_list = ["person", "org"]

    def valid_output(records: Sequence[object]) -> bool:
        if not _records_contract_valid(records, empty_ok=True):
            return False
        return _page_offsets_resolve(records)

    def _run_valid(provider: Any) -> Sequence[Record]:
        return list(provider.extract(fixtures.canonical_corpus().chunks))

    def _run_adversarial(provider: Any) -> AdversarialOutcome:
        # An empty chunk yields no mentions (the S4 defined "none", never an error).
        from latence_core.contracts import ChunkRecord

        base = fixtures.canonical_corpus().chunks[0]
        empty_chunk = ChunkRecord(
            record_id="adv:chunk",
            document_record_id=base.document_record_id,
            index=0,
            provenance=base.provenance.model_copy(update={"char_start": 0, "char_end": 0}),
            classification=base.classification,
            content="",
            token_count=0,
        )
        return _capture_raise(lambda: list(provider.extract([empty_chunk])))

    if stub_scoped and gliner:
        def invoke(provider: Any) -> Sequence[Record]:
            with stubs.install_gliner(entities=stubs.gliner_entities()):
                return _run_valid(provider)

        def adversarial(provider: Any) -> AdversarialOutcome:
            with stubs.install_gliner(entities=[]):
                return _run_adversarial(provider)

        config: dict[str, Any] = {"labels": labels_list}
    elif stub_scoped:
        def invoke(provider: Any) -> Sequence[Record]:
            with stubs.install_openai(reply=stubs.entity_chat_reply()):
                return _run_valid(provider)

        def adversarial(provider: Any) -> AdversarialOutcome:
            with stubs.install_openai(reply='{"entities": []}'):
                return _run_adversarial(provider)

        config = {"labels": labels_list, "base_url": "http://stub/v1"}
    else:
        invoke = _run_valid
        adversarial = _run_adversarial
        config = {"labels": labels_dict}

    return ConformanceCase(
        capability=Capability.ENTITY,
        config=config,
        invoke=invoke,
        valid_output=valid_output,
        adversarial=adversarial,
        empty_ok=True,
        stub_scoped=stub_scoped,
    )


def _relation_case(*, stub_scoped: bool, gliner: bool, backend: str = "gliner") -> ConformanceCase:
    """Relation/Fused Extraction: relate mentions; <2 mentions yields none (graceful empty).

    ``backend`` selects the stub-scoped FUSED provider's fake engine when ``gliner=True``:
    ``"gliner"`` installs the fake ``gliner`` module (the relex ``inference`` tuple —
    ``relation.gliner_relex``); ``"gliner2"`` installs the fake ``gliner2`` module (the fastino
    GLiNER2 nested ``extract_entities`` + ``extract_relations`` dicts — the W9 unified fused
    ``fused_entity_relation.gliner2``). Both drive the SAME ``extract_fused`` contract offline.
    """
    labels = ["works_for"]
    entity_labels = ["person", "org"]

    def valid_output(records: Sequence[object]) -> bool:
        return _records_contract_valid(records, empty_ok=True) and _page_offsets_resolve(records)

    corpus = fixtures.canonical_corpus

    def _run_valid_relate(provider: Any) -> Sequence[Record]:
        c = corpus()
        return list(provider.relate(c.chunks, c.mentions))

    def _run_valid_fused(provider: Any) -> Sequence[Record]:
        c = corpus()
        fused = provider.extract_fused(c.chunks)
        return [*fused.mentions, *fused.relations]

    def _run_adversarial(provider: Any, *, fused: bool) -> AdversarialOutcome:
        # A single mention (no relatable pair) yields no relations — the S5 defined "none".
        c = corpus()
        if fused:
            from latence_core.contracts import ChunkRecord

            base = c.chunks[0]
            empty_chunk = ChunkRecord(
                record_id="adv:chunk",
                document_record_id=base.document_record_id,
                index=0,
                provenance=base.provenance.model_copy(update={"char_start": 0, "char_end": 0}),
                classification=base.classification,
                content="",
                token_count=0,
            )
            return _capture_raise(
                lambda: [*provider.extract_fused([empty_chunk]).mentions,
                         *provider.extract_fused([empty_chunk]).relations]
            )
        one = c.mentions[:1]
        return _capture_raise(lambda: list(provider.relate(c.chunks, one)))

    if stub_scoped and gliner and backend == "gliner2":
        # fused_entity_relation.gliner2 is a FUSED provider (extract_fused) over the fastino GLiNER2
        # framework: extract_entities + extract_relations, the REAL nested per-label/per-relation
        # dict shapes (NOT the relex tuple) — the W9 unified default. Same extract_fused contract.
        def invoke(provider: Any) -> Sequence[Record]:
            with stubs.install_gliner2(
                entities=stubs.gliner2_entities(), relations=stubs.gliner2_relations()
            ):
                return _run_valid_fused(provider)

        def adversarial(provider: Any) -> AdversarialOutcome:
            with stubs.install_gliner2(entities=[], relations=[]):
                return _run_adversarial(provider, fused=True)

        config: dict[str, Any] = {"labels": entity_labels, "relation_labels": labels}
    elif stub_scoped and gliner:
        # gliner-relex is a FUSED provider (extract_fused).
        def invoke(provider: Any) -> Sequence[Record]:
            with stubs.install_gliner(
                entities=stubs.gliner_entities(), relations=stubs.gliner_relations()
            ):
                return _run_valid_fused(provider)

        def adversarial(provider: Any) -> AdversarialOutcome:
            with stubs.install_gliner(entities=[], relations=[]):
                return _run_adversarial(provider, fused=True)

        config = {"labels": entity_labels, "relation_labels": labels}
    elif stub_scoped:
        # relation.llm is an LLM RelationExtractor (relate).
        def invoke(provider: Any) -> Sequence[Record]:
            with stubs.install_openai(reply=stubs.relation_chat_reply()):
                return _run_valid_relate(provider)

        def adversarial(provider: Any) -> AdversarialOutcome:
            with stubs.install_openai(reply='{"relations": []}'):
                return _run_adversarial(provider, fused=False)

        config = {"labels": labels, "base_url": "http://stub/v1"}
    else:
        invoke = _run_valid_relate

        def adversarial(provider: Any) -> AdversarialOutcome:
            return _run_adversarial(provider, fused=False)

        config = {"labels": {"works_for": {"triggers": ["works for"]}}}

    return ConformanceCase(
        capability=Capability.RELATION,
        config=config,
        invoke=invoke,
        valid_output=valid_output,
        adversarial=adversarial,
        empty_ok=True,
        stub_scoped=stub_scoped,
    )


def _redaction_case(*, stub_scoped: bool, backend: str = "gliner") -> ConformanceCase:
    """Redaction (W16, ADR-0042): a chunk→chunk transform — mask PII per chunk, never truncate.

    ``backend`` selects the stub-scoped Provider's fake detection engine: ``"gliner"`` installs
    the fake ``gliner`` module (the learned ``redaction.gliner_pii``), ``"gliner2"`` installs the
    fake ``gliner2`` module (the fastino GLiNER2 ``redaction.gliner2`` — a per-label-dict API,
    NOT the gliner flat list), and ``"presidio"`` installs the fake ``presidio_analyzer`` module
    (the real Microsoft Presidio ``redaction.presidio``). All return canned PII spans keyed to the
    canonical fixture chunk, so the same contract / graceful-failure / license / no-PII-leak subset
    is checked without a live model.
    """
    pii_labels = ["person", "email"]

    def valid_output(records: Sequence[object]) -> bool:
        return _records_contract_valid(records, empty_ok=True)

    def _run_valid(provider: Any) -> Sequence[Record]:
        return list(provider.redact(fixtures.canonical_corpus().chunks))

    def _run_adversarial(provider: Any) -> AdversarialOutcome:
        # The PII-bearing adversarial chunk: the Redactor masks it and returns it gracefully (never
        # a crash, never PII in any raised error).
        return _capture_raise(lambda: list(provider.redact([_pii_chunk()])))

    if stub_scoped:
        @contextlib.contextmanager
        def _install_backend(*, empty: bool = False) -> Iterator[None]:
            if backend == "presidio":
                spans = [] if empty else stubs.presidio_pii_spans()
                with stubs.install_presidio(spans=spans):
                    yield
            elif backend == "gliner2":
                rows = [] if empty else stubs.gliner2_pii_entities()
                with stubs.install_gliner2(entities=rows):
                    yield
            else:
                entities = [] if empty else stubs.gliner_pii_entities()
                with stubs.install_gliner(entities=entities):
                    yield

        def invoke(provider: Any) -> Sequence[Record]:
            with _install_backend():
                return _run_valid(provider)

        def adversarial(provider: Any) -> AdversarialOutcome:
            with _install_backend(empty=True):
                return _run_adversarial(provider)

        # Presidio keys off Presidio entity types (config ``entities``); gliner off zero-shot
        # ``pii_labels``. Each Provider's required config shape.
        config: dict[str, Any] = (
            {"entities": ["EMAIL_ADDRESS", "PERSON"]}
            if backend == "presidio"
            else {"pii_labels": pii_labels}
        )
    else:
        invoke = _run_valid
        adversarial = _run_adversarial
        config = {}

    return ConformanceCase(
        capability=Capability.REDACTION,
        config=config,
        invoke=invoke,
        valid_output=valid_output,
        adversarial=adversarial,
        empty_ok=True,
        stub_scoped=stub_scoped,
    )


def _profiling_case() -> ConformanceCase:
    """Profiling: corpus features from docs (+ mentions); an empty corpus yields none (graceful)."""

    def invoke(provider: Any) -> Sequence[Record]:
        c = fixtures.canonical_corpus()
        return list(provider.profile(c.documents, c.mentions))

    def valid_output(records: Sequence[object]) -> bool:
        return _records_contract_valid(records, empty_ok=False)

    def adversarial(provider: Any) -> AdversarialOutcome:
        # An empty corpus (no documents) yields no feature records — graceful empty.
        return _capture_raise(lambda: list(provider.profile([], [])))

    return ConformanceCase(
        capability=Capability.PROFILING,
        config={},
        invoke=invoke,
        valid_output=valid_output,
        adversarial=adversarial,
    )


def _disambiguation_case() -> ConformanceCase:
    """Disambiguation: canonical entities/relations from mentions; no mentions → none (graceful)."""

    def invoke(provider: Any) -> Sequence[Record]:
        c = fixtures.canonical_corpus()
        # Thread the run's chunks (W1 rework, #94): the algorithmic cascade ignores them, and a
        # context-embedding Disambiguator reads them — so the case exercises both drop-in Providers
        # through the widened signature.
        return list(provider.disambiguate(c.mentions, c.relations, c.chunks))

    def valid_output(records: Sequence[object]) -> bool:
        return _records_contract_valid(records, empty_ok=False)

    def adversarial(provider: Any) -> AdversarialOutcome:
        # No mentions to resolve → no records, never a crash (the defined empty). Empty chunks too.
        return _capture_raise(lambda: list(provider.disambiguate([], [], [])))

    return ConformanceCase(
        capability=Capability.DISAMBIGUATION,
        config={},
        invoke=invoke,
        valid_output=valid_output,
        adversarial=adversarial,
    )


def _graph_case() -> ConformanceCase:
    """Graph Assembly: nodes/edges from DisambiguationRecords; no records → none (graceful)."""

    def invoke(provider: Any) -> Sequence[Record]:
        return list(provider.assemble(fixtures.graph_records()))

    def valid_output(records: Sequence[object]) -> bool:
        return _records_contract_valid(records, empty_ok=False)

    def adversarial(provider: Any) -> AdversarialOutcome:
        return _capture_raise(lambda: list(provider.assemble([])))

    return ConformanceCase(
        capability=Capability.GRAPH,
        config={},
        invoke=invoke,
        valid_output=valid_output,
        adversarial=adversarial,
    )


def _graph_completion_case() -> ConformanceCase:
    """Graph Completion: predicted inferred EDGE records from the assembled graph (W2-linkpred).

    Invokes the completer on the reference-assembled graph (NODE + EDGE GraphRecords). The output
    is a (possibly empty) set of predicted EDGE-scope GraphRecords — ``empty_ok=True`` because a
    small graph may have no 2-hop closure to propose, yet the run must still be contract-valid +
    deterministic (C4). An empty input yields no predictions, never a crash (the defined empty).
    """

    def invoke(provider: Any) -> Sequence[Record]:
        return list(provider.complete(fixtures.assembled_graph()))

    def valid_output(records: Sequence[object]) -> bool:
        return _records_contract_valid(records, empty_ok=True)

    def adversarial(provider: Any) -> AdversarialOutcome:
        return _capture_raise(lambda: list(provider.complete([])))

    return ConformanceCase(
        capability=Capability.GRAPH_COMPLETION,
        config={},
        invoke=invoke,
        valid_output=valid_output,
        adversarial=adversarial,
    )


def _graph_completion_ultra_case() -> ConformanceCase:
    """Graph Completion, LEARNED ULTRA/PyKEEN Provider (W2-linkpred Slice 2) — stub-scoped.

    ``graph_completion.ultra`` (latence-linkpred-ultra) ships its scorer's heavy deps
    (torch-geometric / torch-scatter / easydict / pykeen) in a SEPARATE package (ADR-0016), absent
    from the CPU-only conformance env — so latence-core cannot import a real ULTRA to exercise here.
    The honest disposition (the same skip-with-flag posture the gpu-sota stack uses for this Stage):
    run it with ``skip_if_unavailable=True`` so on this host it yields the contract-defined EMPTY
    (no scorer → no inferred edges, logged) — NOT a fabricated prediction. C1 (empty_ok is
    contract-valid), C2 (empty input is graceful), C3 (its declared MIT license), C4 (empty is
    trivially deterministic), C5 (device), C6 (no PII) are all checked without a rig; the row is
    flagged ``stub_scoped`` ("needs a rig"). The REAL ULTRA inference + its C1 over real predictions
    are validated on the pod (scripts/validate-linkpred-on-gpu.md), never faked here.
    """

    def invoke(provider: Any) -> Sequence[Record]:
        return list(provider.complete(fixtures.assembled_graph()))

    def valid_output(records: Sequence[object]) -> bool:
        return _records_contract_valid(records, empty_ok=True)

    def adversarial(provider: Any) -> AdversarialOutcome:
        return _capture_raise(lambda: list(provider.complete([])))

    return ConformanceCase(
        capability=Capability.GRAPH_COMPLETION,
        # skip_if_unavailable: on this CPU host with no scorer deps, yield empty (skip-with-flag),
        # never a crash and never a fabricated prediction — the honest, contract-valid disposition.
        config={"scorer": "auto", "skip_if_unavailable": True},
        invoke=invoke,
        valid_output=valid_output,
        adversarial=adversarial,
        empty_ok=True,
        stub_scoped=True,
    )


def _graph_features_case() -> ConformanceCase:
    """Graph-feature enrichment (ADR-0051): centrality + community per node over the assembled KG.

    Invokes the computer on the reference-assembled graph (NODE + EDGE GraphRecords) and RETURNS the
    per-node features (as sorted ``(node_id, centrality, community)`` tuples) so C4 byte-compares
    them — a computer that lied about ``deterministic`` (an unrounded PageRank, a randomised
    label-propagation) FAILs. C1 (custom ``valid_output``) asserts EVERY node is covered, each
    centrality a finite float and each community a non-empty string (a genuine compute, not a
    no-op). C2's adversarial input is the EMPTY graph → the empty feature set (graceful empty).
    """
    from latence_core.contracts import GraphScope

    def _split(records: list[Any]) -> tuple[list[Any], list[Any]]:
        nodes = [r.node for r in records if r.scope is GraphScope.NODE and r.node is not None]
        edges = [r.edge for r in records if r.scope is GraphScope.EDGE and r.edge is not None]
        return nodes, edges

    def _flatten(features: Any) -> list[object]:
        # Sorted (node_id, centrality, community) tuples — a deterministic, serialisable surface the
        # harness's ``str()``-based dump byte-compares for C4.
        return [
            (node_id, features.centrality[node_id], features.community[node_id])
            for node_id in sorted(features.centrality)
        ]

    node_ids = {r.node.node_id for r in fixtures.assembled_graph() if r.node is not None}

    def invoke(provider: Any) -> Sequence[object]:
        nodes, edges = _split(fixtures.assembled_graph())
        return _flatten(provider.compute_features(nodes, edges))

    def valid_output(produced: Sequence[object]) -> bool:
        rows = [row for row in produced if isinstance(row, tuple)]
        if not rows:
            return False
        # Every assembled node is covered exactly once (the total-coverage contract).
        if {row[0] for row in rows} != node_ids:
            return False
        import math

        return all(
            isinstance(row[1], float)
            and math.isfinite(row[1])
            and isinstance(row[2], str)
            and bool(row[2])
            for row in rows
        )

    def adversarial(provider: Any) -> AdversarialOutcome:
        # The empty graph yields the empty feature set — a graceful empty, never a crash.
        return _capture_raise(lambda: _flatten(provider.compute_features([], [])))

    return ConformanceCase(
        capability=Capability.GRAPH_FEATURES,
        config={},
        invoke=invoke,
        valid_output=valid_output,
        adversarial=adversarial,
        empty_ok=True,
    )


def _context_enrichment_case() -> ConformanceCase:
    """Context Enrichment (W13): project the assembled KG back onto each chunk as a context_header.

    Invokes the enricher on the canonical corpus's chunks + mentions + reference-assembled graph.
    The
    output is the SAME chunks (1:1, order-stable) — a chunk with a confidence-gated canonical entity
    carries a populated :class:`~latence_core.contracts.ContextHeader`. ``empty_ok=False``: the
    canonical corpus has entities, so at least one chunk must be enriched (a genuine annotation
    check,
    not a no-op). An empty input yields no chunks, never a crash (the defined graceful empty, C2).
    """

    def invoke(provider: Any) -> Sequence[Record]:
        corpus = fixtures.canonical_corpus()
        return list(
            provider.enrich(corpus.chunks, corpus.mentions, fixtures.assembled_graph())
        )

    def valid_output(records: Sequence[object]) -> bool:
        # C1 boundary contract (Provenance/Classification/offsets) — a chunk→chunk transform.
        if not _records_contract_valid(records, empty_ok=False):
            return False
        from latence_core.contracts import ChunkRecord, ContextHeader

        chunks = [r for r in records if isinstance(r, ChunkRecord)]
        if not chunks:
            return False
        # At least one chunk carries a well-formed header whose entities are populated (the
        # canonical corpus evidences entities) — the genuine annotation check, not a no-op.
        headers = [c.context_header for c in chunks if isinstance(c.context_header, ContextHeader)]
        return bool(headers) and all(bool(h.entities) for h in headers)

    def adversarial(provider: Any) -> AdversarialOutcome:
        # Empty inputs: the enricher yields no chunks (nothing to enrich) — a graceful, non-crashing
        # empty, not a raise.
        return _capture_raise(lambda: list(provider.enrich([], [], [])))

    return ConformanceCase(
        capability=Capability.CONTEXT_ENRICHMENT,
        config={},
        invoke=invoke,
        valid_output=valid_output,
        adversarial=adversarial,
    )


def _export_case() -> ConformanceCase:
    """Export: write records to Storage and return URIs; a disallowed URI is a graceful failure."""

    def invoke(provider: Any) -> Sequence[Record]:
        # Export returns URIs (strings), not records; we validate the URIs then re-affirm the
        # inputs' contract (Export is a pass-through). Write to an in-memory Storage.
        storage = Storage.for_uri("memory://conformance-export/")
        records = list(fixtures.canonical_corpus().documents)
        uris = provider.export(records, storage, "memory://conformance-export/out")
        # A conforming Export returns at least one output URI for a non-empty input.
        if not uris or not all(isinstance(u, str) and u for u in uris):
            msg = "Export produced no output URIs for a non-empty input."
            raise AssertionError(msg)
        return records

    def valid_output(records: Sequence[object]) -> bool:
        return _records_contract_valid(records, empty_ok=False)

    def adversarial(provider: Any) -> AdversarialOutcome:
        # A disallowed Storage scheme is a SchemeNotAllowedError — a typed ConfigError (a
        # LatenceError), never a bare crash. Building a Storage for a disallowed scheme raises it.
        def run() -> Any:
            storage = Storage.for_uri("ftp://evil/out/")
            return provider.export([], storage, "ftp://evil/out")

        return _capture_raise(run)

    return ConformanceCase(
        capability=Capability.EXPORT,
        config={},
        invoke=invoke,
        valid_output=valid_output,
        adversarial=adversarial,
    )


def _embedding_case(*, stub_scoped: bool, backend: str = "endpoint") -> ConformanceCase:
    """Embedding: map texts to fixed-width vectors — returned to the harness so C4 audits them.

    Unlike the record-producing Capabilities, an Embedder yields raw ``list[float]`` vectors (not
    Records). The invoke **returns those vectors** (as tuples so :func:`_dump` serializes them
    stably): this is essential so C4's byte-comparison sees the actual embedding values and a
    non-deterministic embedder that lied about ``deterministic=True`` FAILs — the embedder is the
    archetypal case where determinism is a real, checkable property of the *output*, and it is the
    signal the bake-off (P3-F5) trusts. :func:`_embedding_valid` is this case's C1: one vector per
    input text, each of the declared width (order preserved by construction of the tuple list).

    ``backend`` selects the stub-scoped Provider's fake backend: ``"endpoint"`` installs the fake
    ``openai`` client (``embedding.endpoint``), ``"sentence_transformers"`` installs the fake
    ``sentence_transformers`` model (``embedding.sentence_transformers``) — both return
    deterministic fixed-width vectors so the contract subset is checked offline (their live numbers
    need a rig, #64). It is ignored when ``stub_scoped`` is False (the in-core hashing embedder).
    """

    texts = ["alpha beta", "gamma delta", "epsilon"]

    def _run(provider: Any) -> Sequence[object]:
        # Return the vectors themselves (as tuples) so C1 can audit width/order AND C4 can
        # byte-compare the real embedding values — never discard them.
        return [tuple(v) for v in provider.embed(texts)]

    def _embedding_valid(vectors: Sequence[object], *, dim: int) -> bool:
        vecs = [v for v in vectors if isinstance(v, tuple)]
        if len(vecs) != len(texts):
            return False
        return all(len(v) == dim for v in vecs)

    def adversarial(provider: Any) -> AdversarialOutcome:
        # An empty input list yields an empty vector list — graceful empty, never a crash.
        return _capture_raise(lambda: list(provider.embed([])))

    if stub_scoped:
        stub_dim = 8

        @contextlib.contextmanager
        def _install_backend() -> Iterator[None]:
            if backend == "sentence_transformers":
                with stubs.install_sentence_transformers(dimension=stub_dim):
                    yield
            else:
                with stubs.install_openai(dimension=stub_dim):
                    yield

        def invoke(provider: Any) -> Sequence[object]:
            with _install_backend():
                return _run(provider)

        def valid_output(vectors: Sequence[object]) -> bool:
            return _embedding_valid(vectors, dim=stub_dim)

        def adversarial_stub(provider: Any) -> AdversarialOutcome:
            with _install_backend():
                return _capture_raise(lambda: list(provider.embed([])))

        if backend == "sentence_transformers":
            # The ST Provider needs no endpoint base_url; it reads the width from the loaded
            # (stub) model, so declaring the stub width keeps the C1 check exact.
            config: dict[str, Any] = {"dimension": stub_dim}
        else:
            config = {"dimension": stub_dim, "base_url": "http://stub/v1"}
        adversarial_fn = adversarial_stub
    else:
        cpu_dim = int(_embedding_cpu_dimension())

        def invoke(provider: Any) -> Sequence[object]:
            return _run(provider)

        def valid_output(vectors: Sequence[object]) -> bool:
            return _embedding_valid(vectors, dim=cpu_dim)

        adversarial_fn = adversarial
        config = {}

    return ConformanceCase(
        capability=Capability.EMBEDDING,
        config=config,
        invoke=invoke,
        valid_output=valid_output,
        adversarial=adversarial_fn,
        empty_ok=True,
        stub_scoped=stub_scoped,
    )


def _embedding_cpu_dimension() -> int:
    """The width the in-core hashing Embedder produces (its default dimension) — this case's C1."""
    from latence_core.embedding import HashingEmbedder

    return HashingEmbedder({}).dimension


def _sparse_embedding_case(*, stub_scoped: bool = False) -> ConformanceCase:
    """Sparse embedding (ADR-0050/0053): map texts to canonical :class:`SparseVector`s.

    Like the dense :func:`_embedding_case`, the produced sparse vectors are RETURNED to the harness
    so C4 byte-compares them (a non-deterministic sparse encoder that lied about ``deterministic``
    FAILs). C1 asserts one vector per text, each a well-formed :class:`SparseVector` (ascending
    indices in ``[0, vocab_size)``, matching value lengths) with at least one non-empty vector (a
    genuine encode, not an all-zero no-op). The adversarial input is an empty text batch → an empty
    result (graceful empty, never a crash).

    ``stub_scoped`` selects the real ``sparse.splade`` composition Provider (ADR-0053), whose heavy
    ``transformers`` model can't run in the offline env: the invoke installs a faithful fake
    ``transformers`` (:func:`stubs.install_splade`) so its real tokenise → MLM → SPLADE-pool path is
    checked without weights (its live numbers need a rig + a *cleared* checkpoint). Otherwise the
    in-core deterministic ``sparse.hashing`` reference runs for real.
    """
    from latence_core.capability import SparseVector

    texts = ["alpha beta alpha", "gamma delta", "epsilon"]
    vocab = stubs.SPLADE_STUB_VOCAB if stub_scoped else _sparse_vocab_size()

    def _run(provider: Any) -> Sequence[object]:
        return list(provider.encode_sparse(texts))

    def valid_output(vectors: Sequence[object]) -> bool:
        vecs = [v for v in vectors if isinstance(v, SparseVector)]
        if len(vecs) != len(texts):
            return False
        for v in vecs:
            if len(v.indices) != len(v.values):
                return False
            if any(not (0 <= idx < vocab) for idx in v.indices):
                return False
            if list(v.indices) != sorted(set(v.indices)):
                return False
        # At least one non-empty vector — the canonical texts have tokens, so an all-empty result
        # would be a broken encoder, not a valid one.
        return any(v.indices for v in vecs)

    if stub_scoped:

        def invoke(provider: Any) -> Sequence[object]:
            with stubs.install_splade():
                return _run(provider)

        def adversarial(provider: Any) -> AdversarialOutcome:
            with stubs.install_splade():
                return _capture_raise(lambda: list(provider.encode_sparse([])))

        config: dict[str, Any] = {"model": "stub/splade"}
    else:

        def invoke(provider: Any) -> Sequence[object]:
            return _run(provider)

        def adversarial(provider: Any) -> AdversarialOutcome:
            # An empty input batch yields an empty result — graceful empty, never a crash.
            return _capture_raise(lambda: list(provider.encode_sparse([])))

        config = {}

    return ConformanceCase(
        capability=Capability.SPARSE_EMBEDDING,
        config=config,
        invoke=invoke,
        valid_output=valid_output,
        adversarial=adversarial,
        empty_ok=True,
        stub_scoped=stub_scoped,
    )


def _sparse_vocab_size() -> int:
    """The vocab size the in-core hashing SparseEmbedder uses (default) — this case's C1 bound."""
    from latence_core.signals import SparseHashingEmbedder

    return SparseHashingEmbedder({}).vocab_size


_MULTIVECTOR_DIM = 16


def _multivector_embedding_case() -> ConformanceCase:
    """Multi-vector embedding (ADR-0050): map texts to variable-length lists of token vectors.

    The produced multi-vectors are RETURNED so C4 byte-compares them. C1 asserts one multi-vector
    per text, every token vector of the declared width, and at least one non-empty multi-vector (a
    genuine per-token encode). The adversarial input is an empty batch → an empty result (graceful).
    """
    texts = ["alpha beta alpha", "gamma delta", "epsilon"]

    def invoke(provider: Any) -> Sequence[object]:
        return list(provider.encode_multivector(texts))

    def valid_output(multivectors: Sequence[object]) -> bool:
        mvs = [m for m in multivectors if isinstance(m, list)]
        if len(mvs) != len(texts):
            return False
        for mv in mvs:
            for token_vector in mv:
                if not isinstance(token_vector, list) or len(token_vector) != _MULTIVECTOR_DIM:
                    return False
        return any(mv for mv in mvs)

    def adversarial(provider: Any) -> AdversarialOutcome:
        return _capture_raise(lambda: list(provider.encode_multivector([])))

    return ConformanceCase(
        capability=Capability.MULTIVECTOR_EMBEDDING,
        config={"dimension": _MULTIVECTOR_DIM},
        invoke=invoke,
        valid_output=valid_output,
        adversarial=adversarial,
        empty_ok=True,
    )


_FDE_INPUT_DIM = 8
_FDE_BUCKETS = 4


def _fde_input_multivectors() -> list[list[list[float]]]:
    """A small, deterministic set of multi-vectors (``_FDE_INPUT_DIM`` wide) for the FDE case."""
    return [
        [[float((i + j) % 3) for j in range(_FDE_INPUT_DIM)] for i in range(4)],
        [[float((i * j) % 2) for j in range(_FDE_INPUT_DIM)] for i in range(2)],
        [],  # an empty multi-vector → the zero dense vector (the defined degenerate case)
    ]


def _fde_case() -> ConformanceCase:
    """FDE conversion (ADR-0050): reduce each multi-vector to one fixed-width dense vector.

    The produced dense vectors are RETURNED so C4 byte-compares them (a non-deterministic converter
    FAILs). C1 asserts one dense vector per input multi-vector, each of the fixed output width
    (``buckets × input_dimension``). The adversarial input is a token vector of the WRONG width — a
    real converter rejects it, so C2 expects a typed :class:`~latence_core.errors.LatenceError`
    (the faithful contract, anti-false-green: the converter does not silently swallow a mismatch).
    """
    inputs = _fde_input_multivectors()
    output_dim = _FDE_BUCKETS * _FDE_INPUT_DIM

    def invoke(provider: Any) -> Sequence[object]:
        return [tuple(v) for v in provider.convert(inputs)]

    def valid_output(vectors: Sequence[object]) -> bool:
        vecs = [v for v in vectors if isinstance(v, tuple)]
        if len(vecs) != len(inputs):
            return False
        return all(len(v) == output_dim for v in vecs)

    def adversarial(provider: Any) -> AdversarialOutcome:
        # A token vector of the wrong width: the converter raises a typed LatenceError
        # (ConfigError), not a silent mis-encode — the faithful rejection a real converter performs.
        bad = [[[0.0] * (_FDE_INPUT_DIM + 1)]]
        return _capture_raise(lambda: list(provider.convert(bad)))

    return ConformanceCase(
        capability=Capability.FDE,
        config={"input_dimension": _FDE_INPUT_DIM, "buckets": _FDE_BUCKETS},
        invoke=invoke,
        valid_output=valid_output,
        adversarial=adversarial,
        empty_ok=True,
    )


def _tokenizer_case() -> ConformanceCase:
    """Tokenizer (#164): split texts into terms — the varying seam of the BM25 export (ADR-0048).

    Like the Embedder/Sparse cases, a Tokenizer yields raw ``list[str]`` (not Records), so the
    invoke RETURNS the token lists (as tuples) and C4 byte-compares them — a tokenizer that lied
    about ``deterministic`` FAILs. C1 (custom ``valid_output``) asserts one token list per text,
    every token a string, and at least one non-empty list (a genuine split of the canonical texts,
    not an all-empty no-op). The adversarial input is an empty batch → an empty result (graceful
    empty, never a crash).
    """
    texts = ["Alpha, beta ALPHA.", "gamma delta", "epsilon"]

    def invoke(provider: Any) -> Sequence[object]:
        return [tuple(toks) for toks in provider.tokenize(texts)]

    def valid_output(produced: Sequence[object]) -> bool:
        lists = [t for t in produced if isinstance(t, tuple)]
        if len(lists) != len(texts):
            return False
        if not all(isinstance(tok, str) for toks in lists for tok in toks):
            return False
        return any(toks for toks in lists)

    def adversarial(provider: Any) -> AdversarialOutcome:
        return _capture_raise(lambda: list(provider.tokenize([])))

    return ConformanceCase(
        capability=Capability.TOKENIZER,
        config={},
        invoke=invoke,
        valid_output=valid_output,
        adversarial=adversarial,
        empty_ok=True,
    )


def _delta_case() -> ConformanceCase:
    """Delta: apply a Delta to a parent corpus; an empty delta is a graceful no-op."""

    def _build_delta() -> Any:
        from latence_core.contracts import Delta

        return Delta(parent_version=None, entries=[], drift=0.0)

    def invoke(provider: Any) -> Sequence[Record]:
        # A no-op empty delta over an empty parent/current set produces an empty DeltaOutcome —
        # a valid, graceful result. The DeltaProcessor is pure over records; conformance checks
        # that it does not crash and returns a typed outcome whose record set is contract-valid.
        outcome = provider.apply_delta([], [], _build_delta())
        return list(outcome.records)

    def valid_output(records: Sequence[object]) -> bool:
        return _records_contract_valid(records, empty_ok=True)

    def adversarial(provider: Any) -> AdversarialOutcome:
        # A delta referencing a document not in the current set must be handled gracefully
        # (empty affected set), never a crash.
        return _capture_raise(lambda: provider.apply_delta([], [], _build_delta()) and None)

    return ConformanceCase(
        capability=Capability.DELTA,
        config={},
        invoke=invoke,
        valid_output=valid_output,
        adversarial=adversarial,
        empty_ok=True,
    )


# The Capability → base ConformanceCase registry. Providers whose Capability appears here are
# auto-covered; a Provider whose Capability is absent FAILs loudly via :func:`case_for_provider`.
CONFORMANCE_CASES: dict[Capability, ConformanceCase] = {
    Capability.SOURCE: _source_case(),
    Capability.PARSE: _parse_case(stub_scoped=False),
    Capability.CHUNK: _chunk_case(),
    Capability.INTAKE_SCREENING: _intake_screening_case(),
    Capability.CONTENT_SCREENING: _content_screening_case(),
    Capability.SCHEMA_INDUCTION: _schema_induction_case(),
    Capability.ENTITY: _entity_case(stub_scoped=False, gliner=False),
    Capability.RELATION: _relation_case(stub_scoped=False, gliner=False),
    Capability.REDACTION: _redaction_case(stub_scoped=False),
    Capability.PROFILING: _profiling_case(),
    Capability.DISAMBIGUATION: _disambiguation_case(),
    Capability.GRAPH: _graph_case(),
    Capability.GRAPH_COMPLETION: _graph_completion_case(),
    Capability.GRAPH_FEATURES: _graph_features_case(),
    Capability.CONTEXT_ENRICHMENT: _context_enrichment_case(),
    Capability.EXPORT: _export_case(),
    Capability.EMBEDDING: _embedding_case(stub_scoped=False),
    Capability.SPARSE_EMBEDDING: _sparse_embedding_case(),
    Capability.MULTIVECTOR_EMBEDDING: _multivector_embedding_case(),
    Capability.FDE: _fde_case(),
    Capability.TOKENIZER: _tokenizer_case(),
    Capability.DELTA: _delta_case(),
}


# Per-Provider overrides for the stub-scoped endpoint/GPU Providers (they need a stub-installing
# invoke + the right config). Keyed by the exact Provider entry-point name.
_PROVIDER_CASE_OVERRIDES: dict[str, ConformanceCase] = {
    "parser.endpoint": _parse_case(stub_scoped=True),
    # parser.render (W5) is compute="either": it RUNS on a CPU host against a faithful offline fake
    # renderer (fake pypdfium2/PIL) + a stub OCR — not device-skipped, not endpoint-stub-scoped.
    "parser.render": _render_case(),
    # parser.lighton_vllm (W10) is compute="either" + infra-gated (needs a running vLLM server):
    # stub-scoped (async AsyncOpenAI stub for the served endpoint) but its REAL shared rasteriser
    # runs offline (fake pypdfium2/PIL) — the served OCR is the only stubbed thing.
    "parser.lighton_vllm": _lighton_vllm_case(),
    "entity.endpoint": _entity_case(stub_scoped=True, gliner=False),
    "entity.gliner": _entity_case(stub_scoped=True, gliner=True),
    "relation.llm": _relation_case(stub_scoped=True, gliner=False),
    "relation.gliner_relex": _relation_case(stub_scoped=True, gliner=True),
    "fused_entity_relation.gliner2": _relation_case(
        stub_scoped=True, gliner=True, backend="gliner2"
    ),
    "redaction.gliner_pii": _redaction_case(stub_scoped=True, backend="gliner"),
    "redaction.gliner2": _redaction_case(stub_scoped=True, backend="gliner2"),
    "redaction.presidio": _redaction_case(stub_scoped=True, backend="presidio"),
    "embedding.endpoint": _embedding_case(stub_scoped=True),
    "embedding.sentence_transformers": _embedding_case(
        stub_scoped=True, backend="sentence_transformers"
    ),
    # The real multilingual SPLADE structured-composition SparseEmbedder (ADR-0053): its heavy
    # transformers model can't run offline, so it is stub-scoped (a faithful fake transformers), and
    # it needs a 'model' id (ADR-0053 pins no default) — supplied as the stub checkpoint here.
    "sparse.splade": _sparse_embedding_case(stub_scoped=True),
    "graph_completion.ultra": _graph_completion_ultra_case(),
}


# Per-Provider overrides whose case can only be BUILT when the Provider's own package is
# installed (it supplies the faithful offline fake the case drives). Built on first use and
# cached, so importing this module never requires an optional Provider package to be present.
_LAZY_PROVIDER_CASES: dict[str, Callable[[], ConformanceCase]] = {
    # source.sharepoint (#182): the case needs latence_source_sharepoint.testing's Graph fake.
    "source.sharepoint": _sharepoint_source_case,
}
_LAZY_CASE_CACHE: dict[str, ConformanceCase] = {}


def case_for_provider(provider_name: str) -> ConformanceCase:
    """The ConformanceCase a Provider is checked against (per-Provider override else Capability).

    A Provider gets its Capability's base case unless it is a stub-scoped endpoint/GPU Provider
    with a per-Provider override (a stub-installing invoke + its required config). A Provider
    whose Capability has **no** case raises ``KeyError`` — a loud failure that forces the case to
    be written, never a silent coverage gap (the issue's explicit anti-silent-skip requirement).
    """
    if provider_name in _LAZY_PROVIDER_CASES:
        cached = _LAZY_CASE_CACHE.get(provider_name)
        if cached is None:
            cached = _LAZY_PROVIDER_CASES[provider_name]()
            _LAZY_CASE_CACHE[provider_name] = cached
        return cached
    if provider_name in _PROVIDER_CASE_OVERRIDES:
        return _PROVIDER_CASE_OVERRIDES[provider_name]
    capability = capability_of(provider_name)
    try:
        return CONFORMANCE_CASES[capability]
    except KeyError as exc:
        msg = (
            f"No ConformanceCase for Capability {capability!r} (Provider {provider_name!r}); "
            "every registered Provider must be covered — write a case rather than skip."
        )
        raise KeyError(msg) from exc
