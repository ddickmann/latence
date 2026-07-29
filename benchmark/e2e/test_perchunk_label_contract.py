"""VERIFICATION suite for the per-chunk label path (GLiNER2 extract / relations / PII).

Run explicitly (it is outside ``testpaths = ["packages"]``)::

    PYTHONHASHSEED=0 uv run pytest benchmark/e2e/test_perchunk_label_contract.py -v

Every test here asserts a property the 95,809-file production run depends on, over REAL
GovDocs1 text (env ``LATENCE_VERIFY_CORPUS`` → a directory of extracted .txt) and the REAL
``fastino/gliner2-multi-v1`` tokenizer (env ``LATENCE_VERIFY_TOKENIZER`` → tokenizer.json).
Tests needing either are skipped when it is absent, so the file is honest about what was
executed and what was not.

No torch / no gliner2 weights are needed: every assertion is about the LABEL PATH (which is
pure Python) or about tokenisation.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import re
import statistics
from collections import Counter, defaultdict

import pytest
from latence_core.chunking import ChunkBudget, chunk_document, estimate_tokens
from latence_core.contracts import ChunkRecord, Classification, Provenance
from latence_core.errors import ConfigError
from latence_core.induced import LabelFloor, LabelSlot, effective_labels
from latence_core.providers.perf import estimate_token_length, resolve_max_len
from latence_core.setup.recipe import MDEBERTA_TOKEN_EXPANSION, model_window_for
from latence_extract_gliner2.provider import Gliner2FusedExtractor
from latence_pii_gliner2.provider import Gliner2PIIRedactor
from latence_schema_inducer.lexicon import LexiconLabelInducer

REPO = pathlib.Path(__file__).resolve().parents[2]
SCHEMA = json.loads((REPO / "benchmark/e2e/label-schema.json").read_text(encoding="utf-8"))

# The production chunk budget (benchmark/e2e/supervise.sh: --chunk-max-tokens 768
# --chunk-overlap-tokens 96) and the model window the wizard derives from it.
BUDGET = ChunkBudget(max_tokens=768, overlap_tokens=96, min_tokens=8)
EMITTED_WINDOW = model_window_for(768)

_CORPUS = os.environ.get("LATENCE_VERIFY_CORPUS")
_TOKENIZER = os.environ.get("LATENCE_VERIFY_TOKENIZER")

# gliner2 1.3.2 ``processor.WhitespaceTokenSplitter._PATTERN`` — the splitter whose output
# ``max_len`` truncates (processor.py:403-411). Mirrored so the unit can be asserted without
# installing gliner2/torch.
GLINER2_WORD_PATTERN = re.compile(
    r"""(?:https?://[^\s]+|www\.[^\s]+)
    |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
    |@[a-z0-9_]+
    |\w+(?:[-_]\w+)*
    |\S""",
    re.VERBOSE | re.IGNORECASE,
)

# Ground-truth PROXIES for "this chunk really does contain a value of that type": they match the
# VALUE's own surface form, never a cue word, so they are independent of the lexicon config.
VALUE_PROBES = {
    "date": re.compile(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?"
        r"|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\.?\s+\d{1,2},?\s+(?:19|20)\d{2}\b|\b\d{1,2}/\d{1,2}/(?:19|20)?\d{2}\b"
    ),
    "monetary amount": re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?"),
    "person": re.compile(
        r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.\s+[A-Z][a-z]+|\b[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+\b"
    ),
    "organization": re.compile(
        r"\b[A-Z][\w&.'-]*(?:\s+[A-Z][\w&.'-]*){0,4}\s+"
        r"(?:Inc\.?|Corp\.?|Corporation|Company|L\.?L\.?C\.?|Association|University|Institute)\b"
    ),
}


def _chunk(text: str, *, doc: str = "d", index: int = 0, induced=None) -> ChunkRecord:
    prov = Provenance(
        source_uri=f"file:///{doc}",
        file_name=doc,
        file_type="txt",
        file_size=len(text),
        document_id=f"sha256:{doc}",
        char_start=0,
        char_end=len(text),
        page_start=1,
        page_end=1,
    )
    return ChunkRecord(
        record_id=f"{doc}#chunk-{index}",
        document_record_id=doc,
        index=index,
        provenance=prov,
        classification=Classification(language="en", sensitivity="public"),
        content=text,
        media_type="text/markdown",
        token_count=estimate_tokens(text),
        induced_labels=induced,
    )


@pytest.fixture(scope="session")
def corpus_chunks() -> list[ChunkRecord]:
    """Real corpus chunks at the production budget, capped per document.

    Capped because one 1.4 MB coordinate dump would otherwise be 63% of the sample and the
    measurement would describe that file rather than the corpus.
    """
    if not _CORPUS:
        pytest.skip("set LATENCE_VERIFY_CORPUS to a directory of extracted corpus .txt files")
    root = pathlib.Path(_CORPUS)
    per_doc: Counter[str] = Counter()
    out: list[ChunkRecord] = []
    for path in sorted(root.iterdir()):
        for i, piece in enumerate(chunk_document(path.read_text("utf-8"), BUDGET)):
            if per_doc[path.name] >= 20:
                break
            per_doc[path.name] += 1
            out.append(_chunk(piece.text, doc=path.name, index=i))
    if not out:
        pytest.skip(f"no chunks produced from {root}")
    return out


@pytest.fixture(scope="session")
def tokenizer():
    if not _TOKENIZER:
        pytest.skip("set LATENCE_VERIFY_TOKENIZER to the gliner2 tokenizer.json path")
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(_TOKENIZER)
    tok.no_truncation()
    tok.no_padding()
    return tok


@pytest.fixture(scope="session")
def induced_corpus(corpus_chunks: list[ChunkRecord]) -> list[ChunkRecord]:
    inducer = LexiconLabelInducer(
        {"granularity": "chunk", "candidates": SCHEMA, "min_hits": 1, "whole_word": True}
    )
    return list(inducer.induce(corpus_chunks))


# ==========================================================================================
# Q1 — the token budget: is the MODEL-side count actually bounded?
# ==========================================================================================


def test_real_mdeberta_token_count_of_real_chunks_fits_the_emitted_window(
    corpus_chunks: list[ChunkRecord], tokenizer
) -> None:
    """The claim the repo's calibration makes, checked with the REAL tokenizer for the first time.

    ``setup/recipe.py:210-214`` pins ``MDEBERTA_TOKEN_EXPANSION = 1.2`` as "the MAX observed
    ratio" and derives ``max_len = ceil(768 * 1.2) -> 960``; ``providers/perf.py:930-938``
    documents ``max_len`` as "the mdeberta-v3 token window ... a longer input is SILENTLY
    truncated". ``test_setup_wizard_knobs.py:203-262`` only ever checks PROXIES
    (``estimate_tokens * 1.2`` and a chars/4 heuristic). This is the same claim against the real
    ``fastino/gliner2-multi-v1`` tokenizer.
    """
    counts = [len(tokenizer.encode(c.content).ids) for c in corpus_chunks]
    ratios = [
        n / estimate_tokens(c.content)
        for n, c in zip(counts, corpus_chunks)
        if estimate_tokens(c.content)
    ]
    over = [n for n in counts if n > EMITTED_WINDOW]
    assert not over, (
        f"{len(over)}/{len(counts)} real chunks tokenize to more than the emitted "
        f"max_len={EMITTED_WINDOW} mdeberta tokens (median {statistics.median(counts):.0f}, "
        f"max {max(counts)}); measured chunker->mdeberta expansion median "
        f"{statistics.median(ratios):.2f} / MAX {max(ratios):.2f} vs the pinned "
        f"MDEBERTA_TOKEN_EXPANSION={MDEBERTA_TOKEN_EXPANSION}"
    )


def test_max_len_is_a_word_token_bound_and_no_real_chunk_reaches_it(
    corpus_chunks: list[ChunkRecord],
) -> None:
    """The unit ``max_len`` is ACTUALLY in: gliner2 word tokens, not mdeberta subword tokens.

    ``gliner2/processor.py:403-411`` truncates ``text_tokens`` produced by
    ``WhitespaceTokenSplitter`` (``processor.py:150-163``, wired at ``processor.py:236``), i.e.
    ``max_len`` caps WORDS. This is the invariant that actually protects the run.
    """
    words = [len(GLINER2_WORD_PATTERN.findall(c.content)) for c in corpus_chunks]
    over = [n for n in words if n > EMITTED_WINDOW]
    assert not over, (
        f"{len(over)}/{len(words)} chunks exceed max_len={EMITTED_WINDOW} gliner2 WORD tokens "
        f"(median {statistics.median(words):.0f}, max {max(words)}) — real truncation"
    )


def test_microbatch_vram_proxy_does_not_undercount_true_encoder_positions(
    corpus_chunks: list[ChunkRecord], tokenizer
) -> None:
    """``estimate_token_length`` (chars/4, capped at max_len) sizes the padded micro-batch.

    ``providers/perf.py:1356-1368`` + ``POD_SAFE_PADDED_POSITIONS``. The encoder's true sequence
    is ``subwords(schema labels) + [SEP] + subwords(each text word)``
    (``gliner2/processor.py::_format_input_with_mapping``), so if the proxy is smaller than the
    true subword count the VRAM headroom model is wrong by that factor.
    """
    ratios = []
    for chunk in corpus_chunks:
        words = GLINER2_WORD_PATTERN.findall(chunk.content.lower())
        true = sum(len(tokenizer.encode(w, add_special_tokens=False).ids) for w in words)
        proxy = estimate_token_length(chunk.content, EMITTED_WINDOW)
        if proxy:
            ratios.append(true / proxy)
    assert max(ratios) <= 1.0, (
        f"the padded-batch proxy UNDERCOUNTS real encoder positions by median "
        f"{statistics.median(ratios):.2f}x / max {max(ratios):.2f}x — the "
        f"POD_SAFE_PADDED_POSITIONS headroom is overstated by that factor"
    )


# ==========================================================================================
# Q2 — does per-chunk narrowing genuinely happen?
# ==========================================================================================


def test_a_chunk_receives_only_the_types_whose_cues_matched(induced_corpus) -> None:
    """No chunk may receive the whole pool, and the median must be a small fraction of it."""
    pool = len(SCHEMA["entity_types"])
    counts = [
        len(c.induced_labels.entity_types) if c.induced_labels else 0 for c in induced_corpus
    ]
    whole_pool = sum(1 for n in counts if n == pool)
    assert whole_pool == 0, f"{whole_pool}/{len(counts)} chunks received the whole {pool}-type pool"
    assert statistics.median(counts) <= pool / 3, (
        f"median labels/chunk {statistics.median(counts)} of a {pool}-type pool — narrowing is "
        "not doing real work"
    )


def test_the_effective_label_set_the_extractor_prompts_with_is_the_narrowed_one(
    induced_corpus,
) -> None:
    """The seam that decides it: ``effective_labels(..., floor=PER_CHUNK)`` under the extractor's
    ``_DEFAULT_LABEL_FLOOR`` (``latence-extract-gliner2/provider.py:160``)."""
    config_pool = list(SCHEMA["entity_types"])
    per_chunk = [
        len(effective_labels(config_pool, c.induced_labels, LabelSlot.ENTITY,
                             floor=LabelFloor.PER_CHUNK))
        for c in induced_corpus
    ]
    universal = [
        len(effective_labels(config_pool, c.induced_labels, LabelSlot.ENTITY,
                             floor=LabelFloor.UNIVERSAL))
        for c in induced_corpus
    ]
    assert sum(per_chunk) < sum(universal) / 2, (
        f"per_chunk floor scored {sum(per_chunk)} label-slots vs {sum(universal)} universal — "
        "less than a 2x saving means the redesign is not paying for itself"
    )


def test_the_extractor_resolves_the_narrowed_set_end_to_end() -> None:
    """Through the Provider itself (``Gliner2FusedExtractor._prepare``), not just the seam."""
    from latence_core.contracts import InducedLabels

    extractor = Gliner2FusedExtractor(
        {"labels": list(SCHEMA["entity_types"]), "relation_labels": list(SCHEMA["relation_types"])}
    )
    induced = InducedLabels(
        entity_types=["person", "date"],
        relation_types=["issued by"],
        pii_types=[],
        model_id="lexicon",
    )
    pending = extractor._prepare(_chunk("Signed by Mr. Adams on May 25, 2005.", induced=induced))  # noqa: SLF001
    assert pending.entity_labels == ("person", "date")
    assert pending.relation_labels == ("issued by",)


# ==========================================================================================
# Q3 — THE RISK: are the highest-value types narrowed AWAY?
# ==========================================================================================


@pytest.mark.parametrize("type_name", ["person", "date", "monetary amount", "organization"])
def test_high_value_types_are_induced_where_their_values_actually_occur(
    induced_corpus, type_name: str
) -> None:
    """A chunk that demonstrably CONTAINS a value of the type must receive that type.

    If it does not, the extractor is never asked the question and the value can never be
    extracted — a silent, total loss for that type on that chunk.
    """
    probe = VALUE_PROBES[type_name]
    present = [c for c in induced_corpus if probe.search(c.content)]
    if not present:
        pytest.skip(f"no chunk in the sample carries a {type_name} value")
    asked = [
        c
        for c in present
        # induced_labels is None ⇒ the PER_CHUNK fail-open floor hands back the full config list.
        if c.induced_labels is None or type_name in c.induced_labels.entity_types
    ]
    recall = len(asked) / len(present)
    assert recall >= 0.9, (
        f"{type_name}: only {len(asked)}/{len(present)} = {recall:.1%} of the chunks that "
        f"literally contain a {type_name} value are ever SCORED for it — the rest are narrowed "
        "away and can never be extracted"
    )


def test_a_cue_that_starts_or_ends_with_punctuation_can_never_fire() -> None:
    """Cue-authoring trap: ``_compile_cue`` wraps every cue in ``\\b`` (lexicon.py:_compile_cue).

    So ``$``, ``%``, ``Mr.`` and ``U.S.C.`` are DEAD cues under the default ``whole_word: True``
    — accepted silently at construction, matching nothing, ever.
    """
    text = "Mr. Smith of 47 U.S.C. 151 reported $3,200,000 (12.5% growth) on May 25, 2005."
    dead, alive = [], []
    for cue in ("$", "%", "Mr.", "U.S.C.", "Mr", "U.S.C", "May"):
        inducer = LexiconLabelInducer(
            {"granularity": "chunk", "candidates": {"entity_types": {"probe": [cue]}}}
        )
        fired = list(inducer.induce([_chunk(text)]))[0].induced_labels is not None
        (alive if fired else dead).append(cue)
    assert dead == [], (
        f"cues that never fire despite occurring literally in the text: {dead} "
        f"(cues that do fire: {alive})"
    )


# ==========================================================================================
# Q4 — relations
# ==========================================================================================


def test_relation_types_are_narrowed_by_the_same_mechanism(induced_corpus) -> None:
    """Same inducer, same slot machinery, its OWN bucket key in the extractor."""
    pool = len(SCHEMA["relation_types"])
    counts = [
        len(c.induced_labels.relation_types) if c.induced_labels else 0 for c in induced_corpus
    ]
    assert max(counts) < pool
    with_any = sum(1 for n in counts if n)
    assert with_any, "no chunk induced any relation type — the relation cues are all dead"


def test_a_chunk_can_induce_entity_types_and_no_relation_types() -> None:
    """The asymmetry that matters: relation narrowing is INDEPENDENT of entity narrowing, so a
    chunk can be scored for entities while being scored for zero relation types — in which case
    the PER_CHUNK floor hands back the FULL config relation list (fail-open, not empty)."""
    inducer = LexiconLabelInducer(
        {"granularity": "chunk", "candidates": SCHEMA, "min_hits": 1}
    )
    text = "Bureau of Land Management annual Report, Table 3, page 41."
    out = list(inducer.induce([_chunk(text)]))[0]
    assert out.induced_labels is not None
    assert out.induced_labels.entity_types
    assert out.induced_labels.relation_types == []
    eff = effective_labels(
        list(SCHEMA["relation_types"]),
        out.induced_labels,
        LabelSlot.RELATION,
        floor=LabelFloor.PER_CHUNK,
    )
    assert eff == list(SCHEMA["relation_types"]), (
        "an empty induced relation slot must fall back to the full config list, never to []"
    )


def test_a_relation_is_intra_chunk_by_construction() -> None:
    """A relation's endpoints must both be mentions of the SAME chunk.

    ``Gliner2FusedExtractor._build_records`` resolves head/tail by CHUNK-LOCAL ``(start, end)``
    against ``by_span``, which only ever holds this chunk's mentions, and drops a relation whose
    endpoint is not there. Nothing in the extract Stage can emit a cross-chunk relation; the only
    cross-chunk edges in the pipeline are the INFERRED ones the optional ``graph_completion``
    Stage proposes over the assembled graph after disambiguation merges coreferent entities.
    """
    from latence_core.contracts import RelationMention

    # A relation names its two endpoints by MENTION id; every mention carries a chunk_record_id,
    # and both ids can only ever be minted from the one chunk _build_records was called with.
    assert {"head_mention_id", "tail_mention_id"} <= set(RelationMention.model_fields)
    from latence_core.contracts import EntityMention

    assert "chunk_record_id" in EntityMention.model_fields
    source = pathlib.Path(
        REPO / "packages/latence-extract-gliner2/src/latence_extract_gliner2/provider.py"
    ).read_text(encoding="utf-8")
    assert "by_span.setdefault((local_start, local_end), mention)" in source
    assert "a relation whose endpoint is not among the emitted" in source


# ==========================================================================================
# The WIRING of the pool into the emitted stack (what `latence setup --label-candidates` does)
# ==========================================================================================


def _emitted_stack() -> dict:
    path = os.environ.get("LATENCE_VERIFY_STACK")
    if not path:
        pytest.skip("set LATENCE_VERIFY_STACK to a stack.yaml generated with --label-candidates")
    import yaml

    return {s["name"]: s for s in yaml.safe_load(pathlib.Path(path).read_text("utf-8"))["stages"]}


def test_the_emitted_stack_uses_lexicon_induction_and_no_llm() -> None:
    stages = _emitted_stack()
    providers = {s["provider"] for s in stages.values()}
    assert "label_inducer.lexicon" in providers
    assert "label_inducer.llm" not in providers
    candidates = stages["induce"]["config"]["candidates"]
    assert 30 <= sum(len(v) for v in candidates.values()) <= 40


def test_the_extractors_fail_open_floor_is_the_corpus_pool_not_the_demo_default() -> None:
    """The recall net a chunk that induced NOTHING falls back to.

    Under ``label_floor: per_chunk`` that net is ``extract.config.labels`` /
    ``relation_labels``. ``setup/recipe.py::_entity_labels`` only ever reads
    ``predefined_labels`` — and ``--label-candidates`` is mutually exclusive with
    ``--predefined-labels`` (``cli.py:1067-1073``) — so a pool-configured stack cannot express a
    corpus-appropriate floor.
    """
    stages = _emitted_stack()
    pool = stages["induce"]["config"]["candidates"]
    extract = stages["extract"]["config"]
    assert set(extract["labels"]) == set(pool["entity_types"]), (
        f"extract.labels={extract['labels']} is not the candidate pool "
        f"({len(pool['entity_types'])} types) — a chunk that induces nothing is scored against "
        "these and nothing else"
    )
    assert set(extract["relation_labels"]) == set(pool["relation_types"]), (
        f"extract.relation_labels={extract['relation_labels']} is not the candidate pool "
        f"({len(pool['relation_types'])} types)"
    )


# ==========================================================================================
# Q5 — the PII floor (ADR-0044): narrowing must NOT be able to drop a PII type
# ==========================================================================================


def test_a_pii_type_with_no_matching_cue_in_the_chunk_is_still_screened() -> None:
    """The security invariant, executed through the Redactor itself.

    The chunk induces ONLY ``email address``; the configured ADR-0044 floor mentions nothing that
    occurs in it. Every configured floor label must still be in the label set the model is asked.
    """
    from latence_core.contracts import InducedLabels
    from latence_core.page_index import PageIndexResolver

    floor = [
        "person",
        "email",
        "phone number",
        "address",
        "iban",
        "credit card number",
        "tax id",
        "bank account number",
        "social security number",
    ]
    redactor = Gliner2PIIRedactor({"pii_labels": floor, "guardrails": False})
    chunk = _chunk(
        "Shipping logistics update. Route 9 container manifest reconciled. "
        "Questions to ops@example.gov. Card on file 4111 1111 1111 1111.",
        induced=InducedLabels(
            entity_types=[], relation_types=[], pii_types=["email address"], model_id="lexicon"
        ),
    )
    pending = redactor._prepare(chunk, PageIndexResolver())  # noqa: SLF001
    missing = [label for label in floor if label not in pending.labels]
    assert not missing, f"PII floor labels dropped from the per-chunk screen: {missing}"
    assert "email address" in pending.labels


def test_the_redactor_refuses_to_have_its_floor_scoped_per_chunk() -> None:
    """``label_floor: per_chunk`` on the Redactor is a loud ConfigError, never a silent no-op."""
    with pytest.raises(ConfigError) as excinfo:
        Gliner2PIIRedactor({"pii_labels": ["iban"], "label_floor": "per_chunk"})
    assert "universal" in str(excinfo.value)
    # …and the accepted spelling still loads.
    Gliner2PIIRedactor({"pii_labels": ["iban"], "label_floor": "universal"})


def test_the_pii_floor_survives_every_chunk_of_the_real_corpus(induced_corpus) -> None:
    """Corpus-scale version of the same invariant, through the shared seam."""
    floor = ["iban", "credit card number", "tax id", "social security number"]
    for chunk in induced_corpus:
        eff = effective_labels(
            floor, chunk.induced_labels, LabelSlot.PII, floor=LabelFloor.UNIVERSAL
        )
        assert set(floor) <= set(eff), f"{chunk.record_id} lost part of the PII floor"
