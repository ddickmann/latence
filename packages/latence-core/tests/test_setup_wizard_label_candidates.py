"""T-WIZ-2: `latence setup --label-candidates` wires the LEXICON inducer, not the LLM one.

`label_inducer.lexicon` (a deterministic, CPU-only, zero-network tagger over an operator-supplied
candidate type space) has shipped and been registered since T5, but every stack the wizard emitted
hard-coded `label_inducer.llm` — so the only way to reach it was to hand-edit the generated YAML,
which is precisely the thing guided setup exists to make unnecessary. This is that wiring, and the
contract these tests hold:

* `--label-candidates <file>` emits `provider: label_inducer.lexicon` with the file's candidate
  space INLINED as the Provider's `candidates` config, wired to the SAME upstream the LLM branch
  used (`content_screen` with PII, `chunk` without) — so the swap is a Provider swap and nothing
  else moves;
* the emitted stack carries **no `label_inducer.llm` anywhere**, and no `base_url` / `model` /
  `api_key` for one. Zero induction calls is then a STRUCTURAL fact — there is no endpoint in the
  config to call — not a Provider promise. The shipped Provider confirms it: fed exactly the
  emitted config it reports `induction_calls == 0`;
* each chunk's induced set is a **proper subset** of the candidate pool. That is the whole point of
  per-chunk induction: if every chunk got the entire pool, the ticket has failed;
* only the keys the Provider actually READS are emitted (`LexiconLabelInducer.__init__`);
* the ADR-0044 financial-PII floor still unions onto the redactor — a non-LLM *relevance* schema is
  an operator's choice, the *safety* floor is not (README invariant 3);
* `--label-candidates` and `--predefined-labels` are mutually exclusive: one says "induce per chunk,
  from this pool", the other says "do not induce at all". A stack that silently honoured one and
  dropped the other would look operator-configured and not be;
* the wizard's determinism contract survives the flag (same inputs -> byte-identical config).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _setup_stack import header_of, stages_by_name
from latence_core.cli import app
from latence_core.setup import (
    FINANCIAL_PII_LABELS,
    RecipeContext,
    SetupAnswers,
    SetupConfigError,
    build_stack_config,
    parse_label_candidates,
    validate_config_text,
)
from latence_core.setup.recipe import InduceContribution
from latence_core.setup.wizard import (
    PARSER_OCR_VLLM,
    PARSER_UNIVERSAL,
    PROFILE_LITE,
    PROFILE_SOTA,
)
from typer.testing import CliRunner

runner = CliRunner()

#: The artifact shape `latence-label-profile --candidates-out` writes: `{slot: {type: [cues]}}`,
#: with a slot that selected nothing OMITTED rather than emitted empty.
_CANDIDATES: dict[str, dict[str, list[str]]] = {
    "entity_types": {
        "tariff heading": ["heading", "HS code", "CN code"],
        "commodity": ["commodity", "goods"],
        "consignee": ["consignee"],
    },
    "relation_types": {"classified under": ["classified under", "falls under"]},
    "pii_types": {"customer reference": ["customer ref", "cust. ref"]},
}


def _src_dir(tmp_path: Path) -> str:
    d = tmp_path / "corpus"
    d.mkdir(exist_ok=True)
    (d / "a.pdf").write_text("a", encoding="utf-8")
    (d / "b.txt").write_text("b", encoding="utf-8")
    return f"file://{d}"


def _answers(
    tmp_path: Path,
    *,
    pii: bool = True,
    profile: str = PROFILE_LITE,
    parser: str = PARSER_UNIVERSAL,
    **overrides: object,
) -> SetupAnswers:
    return SetupAnswers(
        source=_src_dir(tmp_path),
        storage=f"file://{tmp_path}/out",
        parser=parser,
        device="auto",
        pii=pii,
        profile=profile,
        extensions=("pdf", "txt"),
        **overrides,  # type: ignore[arg-type]
    )


# -- the Provider swap, and only the Provider swap ----------------------------


@pytest.mark.parametrize(
    ("profile", "parser"), [(PROFILE_LITE, PARSER_UNIVERSAL), (PROFILE_SOTA, PARSER_OCR_VLLM)]
)
@pytest.mark.parametrize("pii", [True, False])
def test_label_candidates_emit_the_lexicon_inducer_wired_where_the_llm_one_was(
    tmp_path: Path, profile: str, parser: str, pii: bool
) -> None:
    """The `induce` Stage is the lexicon Provider, on the same upstream, and the stack validates."""
    text = build_stack_config(
        _answers(tmp_path, pii=pii, profile=profile, parser=parser, label_candidates=_CANDIDATES)
    )
    validate_config_text(text)  # structural + DAG + capability wiring + phase boundary
    induce = stages_by_name(text)["induce"]
    assert induce.provider == "label_inducer.lexicon"
    assert induce.capability == "schema_induction"
    # The lexicon branch does not re-plumb the DAG: it hangs off exactly what the LLM branch did.
    assert induce.depends_on == (["content_screen"] if pii else ["chunk"])


def test_only_the_keys_the_provider_reads_are_emitted(tmp_path: Path) -> None:
    """Every emitted key is one `LexiconLabelInducer.__init__` actually consumes — nothing else.

    An emitted key the Provider ignores is a config that LOOKS tuned and is not; the LLM branch's
    `model` / `base_url` / `max_tokens` are exactly such keys here.
    """
    induce = stages_by_name(build_stack_config(_answers(tmp_path, label_candidates=_CANDIDATES)))[
        "induce"
    ]
    assert set(induce.config) == {"granularity", "min_hits", "whole_word", "dedup", "candidates"}
    assert induce.config["granularity"] == "chunk"
    assert induce.config["min_hits"] == 1
    assert induce.config["whole_word"] is True
    assert induce.config["dedup"] is True
    # The pool is INLINED verbatim, so the emitted stack is self-contained: no second file to ship
    # to the pod, and `stack validate` sees the whole configuration.
    assert induce.config["candidates"] == _CANDIDATES


def test_min_hits_and_whole_word_are_operator_answers(tmp_path: Path) -> None:
    """German compounding needs `whole_word: false` — the ticket's stated reason for the flag."""
    induce = stages_by_name(
        build_stack_config(
            _answers(
                tmp_path,
                label_candidates=_CANDIDATES,
                label_min_hits=2,
                label_whole_word=False,
            )
        )
    )["induce"]
    assert induce.config["min_hits"] == 2
    assert induce.config["whole_word"] is False


def test_the_default_is_still_the_llm_inducer(tmp_path: Path) -> None:
    """The lexicon path is opt-in — no flag, no change to any stack anyone already generates."""
    induce = stages_by_name(build_stack_config(_answers(tmp_path)))["induce"]
    assert induce.provider == "label_inducer.llm"


# -- zero LLM induction, structurally -----------------------------------------


@pytest.mark.parametrize(
    ("profile", "parser"), [(PROFILE_LITE, PARSER_UNIVERSAL), (PROFILE_SOTA, PARSER_OCR_VLLM)]
)
def test_no_llm_inducer_and_no_endpoint_survive_anywhere_in_the_stack(
    tmp_path: Path, profile: str, parser: str
) -> None:
    """Acceptance 1: zero induction LLM calls, and the config could not make one if it tried."""
    text = build_stack_config(
        _answers(tmp_path, profile=profile, parser=parser, label_candidates=_CANDIDATES)
    )
    assert "label_inducer.llm" not in text
    induce = stages_by_name(text)["induce"]
    assert not {"model", "base_url", "api_key", "max_induction_calls"} & set(induce.config)
    # The comment header must not advertise an induction key the run will never read: an operator
    # told to export a key for a stack that makes no call learns to ignore the warning.
    assert "Schema-induction LLM key" not in header_of(text)


def test_the_real_provider_makes_zero_calls_and_induces_a_proper_subset(tmp_path: Path) -> None:
    """Acceptance 1+2, against the SHIPPED Provider fed EXACTLY the emitted config.

    The structural assertions above prove the stack has no endpoint to call; this proves the
    Provider the stack names behaves as the ticket requires when handed that config: zero calls,
    and each chunk's induced set is a *proper* subset of the pool. A pool every chunk gets in full
    would be a corpus-level schema wearing a per-chunk Stage's clothes — the failure the ticket
    names explicitly.
    """
    lexicon = pytest.importorskip("latence_schema_inducer.lexicon")
    from latence_core.contracts import ChunkRecord, Classification, Provenance

    config = stages_by_name(build_stack_config(_answers(tmp_path, label_candidates=_CANDIDATES)))[
        "induce"
    ].config
    inducer = lexicon.LexiconLabelInducer(dict(config))
    chunks = [
        ChunkRecord(
            record_id=f"doc-1#chunk-{i}",
            document_record_id="doc-1",
            index=i,
            content=content,
            token_count=max(1, len(content) // 4),
            provenance=Provenance(
                source_uri="mem://t/doc.txt",
                file_name="doc.txt",
                file_type="txt",
                file_size=100,
                document_id="sha256:doc-1",
                char_start=0,
                char_end=len(content),
            ),
            classification=Classification(language="en", category="memo", sensitivity="internal"),
        )
        for i, content in enumerate(
            [
                "The commodity is described in the tariff heading and its HS code.",
                "The consignee supplied a customer ref for the shipment.",
            ]
        )
    ]
    induced = list(inducer.induce(chunks))
    assert inducer.induction_calls == 0
    assert inducer.induction_prompt_tokens == 0
    assert inducer.induction_completion_tokens == 0
    pool = {slot: set(types) for slot, types in _CANDIDATES.items()}
    got_any = False
    for chunk in induced:
        labels = chunk.induced_labels
        if labels is None:
            continue
        got_any = True
        for slot in pool:
            found = set(getattr(labels, slot))
            assert found <= pool[slot], f"{slot}: {found} escaped the candidate pool"
        # ...and strictly smaller than the pool as a whole: this chunk is not the whole corpus.
        total = sum(len(getattr(labels, slot)) for slot in pool)
        assert 0 < total < sum(len(v) for v in pool.values())
    assert got_any, "the candidate pool covered nothing at all — the fixture, not the wiring"


# -- the safety floor is not a relevance choice -------------------------------


@pytest.mark.parametrize(
    ("profile", "parser"), [(PROFILE_LITE, PARSER_UNIVERSAL), (PROFILE_SOTA, PARSER_OCR_VLLM)]
)
def test_the_financial_pii_floor_survives_the_lexicon_path(
    tmp_path: Path, profile: str, parser: str
) -> None:
    """Acceptance 3 (ADR-0044): the redactor's floor is unioned, never replaced by the pool."""
    text = build_stack_config(
        _answers(tmp_path, profile=profile, parser=parser, label_candidates=_CANDIDATES)
    )
    pii_labels = stages_by_name(text)["redact"].config["pii_labels"]
    assert list(FINANCIAL_PII_LABELS) == pii_labels[: len(FINANCIAL_PII_LABELS)]


# -- the candidate space is parsed loudly -------------------------------------


def test_parse_label_candidates_accepts_the_profiler_artifact() -> None:
    assert parse_label_candidates(json.loads(json.dumps(_CANDIDATES))) == _CANDIDATES


def test_parse_label_candidates_accepts_the_plain_name_list_form() -> None:
    """`_compile_candidates` takes a bare name list per slot (the name is its own cue) — so does
    the wizard, because a hand-written pool is the cheapest way to try the path."""
    assert parse_label_candidates({"entity_types": ["hs code", "commodity"]}) == {
        "entity_types": ["hs code", "commodity"]
    }


def test_parse_label_candidates_keeps_operator_order_and_drops_blanks() -> None:
    parsed = parse_label_candidates(
        {"entity_types": {" clause ": [" party ", "", "party"], "annex": []}}
    )
    assert parsed == {"entity_types": {"clause": ["party"], "annex": []}}


def test_parse_label_candidates_omits_a_slot_that_carries_nothing() -> None:
    assert parse_label_candidates({"entity_types": ["clause"], "pii_types": []}) == {
        "entity_types": ["clause"]
    }


@pytest.mark.parametrize(
    "bad",
    [
        ["entity_types"],  # not a mapping
        {},  # empty
        {"entities": {"a": ["b"]}},  # typo'd slot — the silent-default trap
        {"entity_types": "clause"},  # neither list nor mapping
        {"entity_types": [1, 2]},  # names must be strings
        {"entity_types": {"clause": "party"}},  # cues must be a list
        {"entity_types": {"clause": [1]}},  # ...of strings
        {"entity_types": []},  # declares nothing usable
        {"entity_types": {"  ": ["x"]}},  # ...nor after stripping
    ],
)
def test_parse_label_candidates_rejects_malformed_input(bad: object) -> None:
    with pytest.raises(SetupConfigError):
        parse_label_candidates(bad)


# -- the two label modes are mutually exclusive -------------------------------


def test_a_candidate_pool_and_a_fixed_schema_cannot_be_asked_for_together(tmp_path: Path) -> None:
    """One says "induce per chunk from this pool", the other says "never induce". Honouring
    either silently would ship a stack that looks operator-configured and is not."""
    with pytest.raises(SetupConfigError):
        _answers(
            tmp_path,
            label_candidates=_CANDIDATES,
            predefined_labels={"entity_types": ["hs code"]},
        )


# -- the recipe seam ----------------------------------------------------------


def test_the_induce_contribution_emits_the_lexicon_block_directly(tmp_path: Path) -> None:
    """C7: the block is assertable in ONE call, without rendering the whole document."""
    ctx = RecipeContext()
    ctx.set_anchor("chunk_stream", "content_screen")
    block = InduceContribution().contribute(_answers(tmp_path, label_candidates=_CANDIDATES), ctx)
    assert block == {
        "name": "induce",
        "capability": "schema_induction",
        "provider": "label_inducer.lexicon",
        "depends_on": ["content_screen"],
        "config": {
            "granularity": "chunk",
            "min_hits": 1,
            "whole_word": True,
            "dedup": True,
            "candidates": _CANDIDATES,
        },
    }
    assert ctx.anchor("labels") == "induce"


# -- the CLI surface ----------------------------------------------------------


def _cli_args(tmp_path: Path, pool: Path) -> list[str]:
    return [
        "setup",
        "--non-interactive",
        "--source",
        _src_dir(tmp_path),
        "--storage",
        f"file://{tmp_path}/out",
        "--label-candidates",
        str(pool),
        "--output",
    ]


def _pool_file(tmp_path: Path) -> Path:
    pool = tmp_path / "candidates.json"
    pool.write_text(json.dumps(_CANDIDATES, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return pool


def test_cli_label_candidates_drives_the_generated_stack_deterministically(tmp_path: Path) -> None:
    """The operator-facing switch, and the ADR-0041 §3 determinism contract with it: two identical
    `--non-interactive` invocations produce byte-identical configs."""
    pool = _pool_file(tmp_path)
    out1, out2 = tmp_path / "a.yaml", tmp_path / "b.yaml"
    args = _cli_args(tmp_path, pool)
    r1 = runner.invoke(app, [*args, str(out1)])
    r2 = runner.invoke(app, [*args, str(out2)])
    assert r1.exit_code == 0, r1.output
    assert r2.exit_code == 0, r2.output
    assert out1.read_bytes() == out2.read_bytes()
    induce = stages_by_name(out1.read_text(encoding="utf-8"))["induce"]
    assert induce.provider == "label_inducer.lexicon"
    assert induce.config["candidates"] == _CANDIDATES


def test_cli_no_label_whole_word_reaches_the_emitted_config(tmp_path: Path) -> None:
    out = tmp_path / "out.yaml"
    result = runner.invoke(
        app,
        [
            *_cli_args(tmp_path, _pool_file(tmp_path)),
            str(out),
            "--no-label-whole-word",
            "--label-min-hits",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    induce = stages_by_name(out.read_text(encoding="utf-8"))["induce"]
    assert induce.config["whole_word"] is False
    assert induce.config["min_hits"] == 2


def test_cli_warns_about_no_llm_key_it_will_never_read(tmp_path: Path) -> None:
    """The LLM-key warning is noise on a path that makes no call — and noise trains operators to
    ignore the warning on the paths where it matters."""
    result = runner.invoke(
        app, [*_cli_args(tmp_path, _pool_file(tmp_path)), str(tmp_path / "out.yaml")]
    )
    assert result.exit_code == 0, result.output
    assert "is not set" not in result.output


def test_cli_rejects_both_label_flags_together(tmp_path: Path) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"entity_types": ["hs code"]}), encoding="utf-8")
    result = runner.invoke(
        app,
        [
            *_cli_args(tmp_path, _pool_file(tmp_path)),
            str(tmp_path / "out.yaml"),
            "--predefined-labels",
            str(schema),
        ],
    )
    assert result.exit_code == 2
    assert "--label-candidates" in result.output
    assert "--predefined-labels" in result.output


def test_cli_rejects_a_malformed_candidates_file(tmp_path: Path) -> None:
    pool = tmp_path / "candidates.json"
    pool.write_text(json.dumps({"entities": {"oops": ["x"]}}), encoding="utf-8")
    result = runner.invoke(app, [*_cli_args(tmp_path, pool), str(tmp_path / "out.yaml")])
    assert result.exit_code == 2
    assert "unknown slot" in result.output


def test_cli_rejects_a_missing_candidates_file(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [*_cli_args(tmp_path, tmp_path / "nope.json"), str(tmp_path / "out.yaml")],
    )
    assert result.exit_code == 2


def test_cli_rejects_a_min_hits_below_one(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            *_cli_args(tmp_path, _pool_file(tmp_path)),
            str(tmp_path / "out.yaml"),
            "--label-min-hits",
            "0",
        ],
    )
    assert result.exit_code == 2
