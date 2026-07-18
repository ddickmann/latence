"""W15 guided-setup tests: `latence setup` generates a valid, deterministic, quality-baked stack
config and `latence process` runs a saved config (ADR-0041).

The properties under test (the slice's contract):

* ``--non-interactive`` setup produces a config that passes the SAME dry validation ``stack
  validate`` builds on (structural + DAG + capability wiring), and is **byte-identical** across two
  runs with the same flags (the determinism/repro guarantee);
* the ONE PII toggle adds/omits EXACTLY the ``content_screen`` + ``redact`` stages and nothing else;
* the mandatory spine is ALWAYS present (both PII on and off);
* ``process`` loads + runs a saved config (a CPU stub — no heavy corpus/models needed);
* a missing source dir is a clear error; an unset LLM key is a clear, non-crashing warning.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from latence_core.cli import app
from latence_core.pipeline import Pipeline, Stage
from latence_core.setup import (
    FINANCIAL_PII_LABELS,
    SetupAnswers,
    build_stack_config,
    recommend_parser,
    scan_source,
    validate_config_text,
)
from latence_core.setup.wizard import (
    BLESSED_REDACT_PROVIDER,
    INTAKE_MAX_BYTES_CEILING,
    INTAKE_MAX_BYTES_FLOOR,
    PARSER_OCR,
    PARSER_OCR_VLLM,
    PARSER_PDF_TEXT,
    PARSER_UNIVERSAL,
    PROFILE_LITE,
    PROFILE_SOTA,
)
from typer.testing import CliRunner

runner = CliRunner()

# The mandatory spine every generated config must always carry (the quality-baked GOOD stack),
# by (stage name, capability). The two toggled stages (content_screen/redact) are NOT here.
_MANDATORY_SPINE: tuple[tuple[str, str], ...] = (
    ("source", "source"),
    ("intake_screen", "intake_screening"),
    ("parse", "parse"),
    ("chunk", "chunk"),
    ("induce", "schema_induction"),
    ("extract", "fused_entity_relation"),
    ("profiling", "profiling"),
    ("disambiguate", "disambiguation"),
    ("graph", "graph_assembly"),
    ("enrich", "context_enrichment"),
    ("export_kg", "export"),
    ("export_corpus", "export"),
)


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
    parser: str = PARSER_UNIVERSAL,
    extract_on: str = "unmasked",
    profile: str = PROFILE_LITE,
) -> SetupAnswers:
    return SetupAnswers(
        source=_src_dir(tmp_path),
        storage=f"file://{tmp_path}/out",
        parser=parser,
        device="auto",
        pii=pii,
        extract_on=extract_on,
        profile=profile,
        extensions=("pdf", "txt"),
    )


# -- determinism + validity ---------------------------------------------------


def test_non_interactive_setup_is_valid_and_byte_identical(tmp_path: Path) -> None:
    """Same flags → a config that passes dry validation AND is byte-identical across two runs."""
    src = _src_dir(tmp_path)
    out1 = tmp_path / "one.stack.yaml"
    out2 = tmp_path / "two.stack.yaml"
    args = [
        "setup",
        "--non-interactive",
        "--source",
        src,
        "--storage",
        f"file://{tmp_path}/out",
        "--output",
    ]
    r1 = runner.invoke(app, [*args, str(out1)])
    r2 = runner.invoke(app, [*args, str(out2)])
    assert r1.exit_code == 0, r1.output
    assert r2.exit_code == 0, r2.output
    assert out1.read_bytes() == out2.read_bytes()  # determinism
    # And the saved config passes the SAME dry validation the setup command applied.
    validate_config_text(out1.read_text(encoding="utf-8"))  # raises on failure


def test_build_stack_config_is_pure_and_deterministic(tmp_path: Path) -> None:
    """The generator is a pure function of its answers (two builds are byte-identical)."""
    answers = _answers(tmp_path)
    assert build_stack_config(answers) == build_stack_config(answers)


def test_generated_config_validates(tmp_path: Path) -> None:
    """The dry validation (structural + DAG + capability wiring) accepts the generated config."""
    for pii in (True, False):
        validate_config_text(build_stack_config(_answers(tmp_path, pii=pii)))


# -- the enterprise-sota profile ----------------------------------------------


def _stage(text: str, name: str) -> Stage:
    return next(s for s in Pipeline.from_yaml_text(text).stages if s.name == name)


def test_sota_profile_emits_the_validated_composed_stack(tmp_path: Path) -> None:
    """enterprise-sota: parse=lighton_vllm, disambiguate=glinker (Granite embedder), induce present
    at corpus granularity — and the whole config passes the SAME dry validation."""
    text = build_stack_config(_answers(tmp_path, profile=PROFILE_SOTA))
    validate_config_text(text)  # raises on failure

    parse = _stage(text, "parse")
    assert parse.provider == PARSER_OCR_VLLM
    assert parse.config["base_url"] == "http://localhost:8000/v1"
    assert parse.config["model"] == "lightonocr"
    assert parse.config["api_key"] == "EMPTY"  # pragma: allowlist secret  (vLLM placeholder)
    assert parse.config["max_concurrency"] == 8

    induce = _stage(text, "induce")
    assert induce.provider == "label_inducer.llm"
    assert induce.config["granularity"] == "corpus"
    assert induce.config["max_labels"] == 40

    extract = _stage(text, "extract")
    assert extract.provider == "fused_entity_relation.gliner2"
    assert extract.config["threshold"] == 0.3
    assert extract.config["device"] == "cuda"

    disambiguate = _stage(text, "disambiguate")
    assert disambiguate.provider == "disambiguation.glinker"
    assert disambiguate.config["device"] == "cuda"
    assert disambiguate.config["context_window"] == 200
    assert disambiguate.config["use_reranker"] is True
    assert disambiguate.config["use_flash_deberta"] is True
    # The Granite r2 embedder is the resolver's STS rung + the export vectors.
    assert (
        disambiguate.config["embedder"]["config"]["model"]
        == "ibm-granite/granite-embedding-311m-multilingual-r2"
    )
    # The precision-by-default GLinker knobs are LEFT to the provider defaults (not emitted).
    assert "use_embedding_merge" not in disambiguate.config
    assert "min_similarity" not in disambiguate.config

    # The blessed redactor (#197, ADR-0041 §7a: the SAME gliner2 engine `extract` runs, in BOTH
    # profiles) + ULTRA graph completion + Granite corpus vectors.
    assert _stage(text, "redact").provider == BLESSED_REDACT_PROVIDER
    complete = _stage(text, "complete")
    assert complete.provider == "graph_completion.ultra"
    assert complete.config["skip_if_unavailable"] is True
    export_corpus = _stage(text, "export_corpus")
    assert (
        export_corpus.config["embedder"]["config"]["model"]
        == "ibm-granite/granite-embedding-311m-multilingual-r2"
    )


def test_sota_non_interactive_is_byte_identical(tmp_path: Path) -> None:
    """`latence setup --non-interactive --profile enterprise-sota` is byte-identical across runs and
    the saved config passes dry validation (the determinism/repro contract for the SOTA profile)."""
    src = _src_dir(tmp_path)
    out1 = tmp_path / "sota1.stack.yaml"
    out2 = tmp_path / "sota2.stack.yaml"
    args = [
        "setup",
        "--non-interactive",
        "--profile",
        PROFILE_SOTA,
        "--source",
        src,
        "--storage",
        f"file://{tmp_path}/out",
        "--output",
    ]
    r1 = runner.invoke(app, [*args, str(out1)])
    r2 = runner.invoke(app, [*args, str(out2)])
    assert r1.exit_code == 0, r1.output
    assert r2.exit_code == 0, r2.output
    assert out1.read_bytes() == out2.read_bytes()  # determinism
    validate_config_text(out1.read_text(encoding="utf-8"))
    # The SOTA parse + disambiguate Providers actually landed in the saved file.
    saved = out1.read_text(encoding="utf-8")
    assert _stage(saved, "parse").provider == PARSER_OCR_VLLM
    assert _stage(saved, "disambiguate").provider == "disambiguation.glinker"


def test_lite_profile_is_unchanged_and_default(tmp_path: Path) -> None:
    """Regression: the default profile is `lite`, and lite emits the prior (non-SOTA) stack — none of
    the SOTA-only Providers leak into it, and the byte output equals an explicit-lite build."""
    default_text = build_stack_config(_answers(tmp_path))  # no profile arg → default
    lite_text = build_stack_config(_answers(tmp_path, profile=PROFILE_LITE))
    assert default_text == lite_text
    providers = {s.provider for s in Pipeline.from_yaml_text(lite_text).stages}
    assert PARSER_OCR_VLLM not in providers
    assert "disambiguation.glinker" not in providers
    assert "graph_completion.ultra" not in providers
    # the prior lite disambiguator + the blessed redactor (#197: the SAME one sota emits)
    assert "disambiguation.embedding" in providers
    assert BLESSED_REDACT_PROVIDER in providers


# -- the ONE PII toggle -------------------------------------------------------


def _stage_names(text: str) -> list[str]:
    return [s.name for s in Pipeline.from_yaml_text(text).stages]


def test_pii_toggle_adds_or_omits_exactly_content_screen_and_redact(tmp_path: Path) -> None:
    """PII on vs off differ by EXACTLY {content_screen, redact} — nothing else moves."""
    on = set(_stage_names(build_stack_config(_answers(tmp_path, pii=True))))
    off = set(_stage_names(build_stack_config(_answers(tmp_path, pii=False))))
    assert on - off == {"content_screen", "redact"}
    assert off - on == set()


def test_pii_off_reroutes_induce_to_chunk(tmp_path: Path) -> None:
    """With content_screen gone, induce depends on chunk directly (a valid, connected DAG)."""
    pipe = Pipeline.from_yaml_text(build_stack_config(_answers(tmp_path, pii=False)))
    induce = next(s for s in pipe.stages if s.name == "induce")
    assert induce.depends_on == ["chunk"]


def test_pii_on_uses_financial_labels(tmp_path: Path) -> None:
    """PII ON wires the learned redactor with the financial-PII label floor."""
    pipe = Pipeline.from_yaml_text(build_stack_config(_answers(tmp_path, pii=True)))
    redact = next(s for s in pipe.stages if s.name == "redact")
    assert redact.provider == "redaction.gliner2"
    assert redact.config["pii_labels"] == list(FINANCIAL_PII_LABELS)


# -- the mandatory spine is always present ------------------------------------


@pytest.mark.parametrize("pii", [True, False])
def test_mandatory_spine_always_present(tmp_path: Path, pii: bool) -> None:
    """Every mandatory spine Stage (name + capability) is present regardless of the PII toggle."""
    pipe = Pipeline.from_yaml_text(build_stack_config(_answers(tmp_path, pii=pii)))
    present = {(s.name, str(s.capability)) for s in pipe.stages}
    for expected in _MANDATORY_SPINE:
        assert expected in present, f"missing mandatory stage {expected}"


def test_export_corpus_reads_the_enriched_chunk_stream(tmp_path: Path) -> None:
    """The RAG corpus is exported from `enrich` (context columns on the enriched chunks)."""
    pipe = Pipeline.from_yaml_text(build_stack_config(_answers(tmp_path, pii=True)))
    export = next(s for s in pipe.stages if s.name == "export_corpus")
    assert export.depends_on == ["enrich"]
    assert export.config["context_columns"] is True


def test_pii_on_wires_w16_chunk_level_redaction_dataflow(tmp_path: Path) -> None:
    """W16 (ADR-0042): with PII ON, redact consumes the induced CHUNK stream and enrich reads the
    REDACTED chunk stream, so export_corpus is masked AND enriched (`redact → enrich → export`)."""
    pipe = Pipeline.from_yaml_text(build_stack_config(_answers(tmp_path, pii=True)))
    redact = next(s for s in pipe.stages if s.name == "redact")
    enrich = next(s for s in pipe.stages if s.name == "enrich")
    # redact is a chunk→chunk transform over the induced chunk stream (NOT the pre-W16 doc-level
    # [parse, induce] shape) so the induced pii_types union onto the FINANCIAL_PII floor.
    assert redact.depends_on == ["induce"]
    # enrich reads the redacted chunk stream first, then the graph — the blessed context-enrichment
    # dataflow, so the exported corpus row carries masked_content AND the context header.
    assert enrich.depends_on == ["redact", "graph"]


def test_pii_off_enrich_reads_induced_chunk_stream(tmp_path: Path) -> None:
    """With PII OFF there is no redact stage, so enrich falls back to the induced chunk stream."""
    pipe = Pipeline.from_yaml_text(build_stack_config(_answers(tmp_path, pii=False)))
    enrich = next(s for s in pipe.stages if s.name == "enrich")
    assert enrich.depends_on == ["induce", "graph"]


# -- W18 mask-before-extract advanced option ----------------------------------


def test_default_is_unmasked_and_omits_the_switch(tmp_path: Path) -> None:
    """The default advanced option is `unmasked`: no `extract_on` key, extract reads the induced
    chunk stream — byte-identical to pre-W18."""
    text = build_stack_config(_answers(tmp_path))
    pipe = Pipeline.from_yaml_text(text)
    assert pipe.extract_on.value == "unmasked"
    assert "extract_on:" not in text  # the field is omitted when it matches the default
    extract = next(s for s in pipe.stages if s.name == "extract")
    assert extract.depends_on == ["induce"]


def test_masked_mode_wires_extract_after_redact(tmp_path: Path) -> None:
    """`extract_on: masked` sets the pipeline switch AND wires `extract` to depend on `redact`, so
    extraction reads the redacted chunk stream and PII never reaches the KG."""
    pipe = Pipeline.from_yaml_text(build_stack_config(_answers(tmp_path, extract_on="masked")))
    assert pipe.extract_on.value == "masked"
    extract = next(s for s in pipe.stages if s.name == "extract")
    assert extract.depends_on == ["redact"]
    # It still validates (structural + DAG + the masked-mode wiring check).
    validate_config_text(build_stack_config(_answers(tmp_path, extract_on="masked")))


def test_masked_requires_pii_on(tmp_path: Path) -> None:
    """masked-before-extract is meaningless without a redact Stage — a clear error, not a bad config."""
    with pytest.raises(ValueError, match="requires PII redaction ON"):
        build_stack_config(_answers(tmp_path, pii=False, extract_on="masked"))


# -- the intake cap is sized from the scanned corpus --------------------------
#
# The defect these pin: the wizard emitted a FLAT 2 MiB `max_bytes`, so a real corpus (the live
# repro: a 33,898,584-byte German PDF, 16.5x the cap) was quarantined at intake in its ENTIRETY and
# `latence setup && latence process` produced an empty corpus with no visible failure.


def _big_source(tmp_path: Path, size: int) -> tuple[str, int]:
    """A source folder holding one sparse file of ``size`` bytes (no 30 MB of real I/O)."""
    d = tmp_path / "big-corpus"
    d.mkdir()
    big = d / "handbook.pdf"
    with big.open("wb") as fh:
        fh.truncate(size)
    (d / "small.txt").write_text("x", encoding="utf-8")
    return f"file://{d}", big.stat().st_size


def _intake_cap(text: str) -> int:
    return int(_stage(text, "intake_screen").config["max_bytes"])


def test_setup_admits_the_largest_file_in_the_scanned_corpus(tmp_path: Path) -> None:
    """The emitted intake cap ADMITS the biggest real file the wizard scanned.

    The behavioural bar (not a flag): a book-sized document under the source folder is under the
    cap the wizard writes, so `latence process` ingests it instead of quarantining it.
    """
    src, size = _big_source(tmp_path, 33_898_584)  # the live-repro PDF's exact size
    out = tmp_path / "big.stack.yaml"
    result = runner.invoke(
        app,
        [
            "setup",
            "--non-interactive",
            "--source",
            src,
            "--storage",
            f"file://{tmp_path}/out",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    cap = _intake_cap(out.read_text(encoding="utf-8"))
    assert cap >= size, f"intake cap {cap} quarantines the corpus's own {size}-byte file"
    # It is still a REAL guard, not infinity: bounded by the headroom rule and the ceiling.
    assert cap <= INTAKE_MAX_BYTES_CEILING
    assert cap < size * 4
    # And the operator is TOLD, in the summary, what the cap is and where it came from.
    assert "intake size cap" in result.output
    assert "sized from your corpus" in result.output


def test_sized_intake_cap_is_deterministic(tmp_path: Path) -> None:
    """Same corpus → byte-identical config (the sizing must not break the determinism contract)."""
    src, _ = _big_source(tmp_path, 9_000_000)
    args = ["setup", "--non-interactive", "--source", src, "--storage", f"file://{tmp_path}/o", "--output"]
    out1, out2 = tmp_path / "a.yaml", tmp_path / "b.yaml"
    assert runner.invoke(app, [*args, str(out1)]).exit_code == 0
    assert runner.invoke(app, [*args, str(out2)]).exit_code == 0
    assert out1.read_bytes() == out2.read_bytes()


def test_small_corpus_keeps_the_default_floor(tmp_path: Path) -> None:
    """A corpus well under the floor keeps the pre-fix 2 MiB default (no needless cap inflation)."""
    text = build_stack_config(_answers(tmp_path))  # tiny files
    assert _intake_cap(text) == INTAKE_MAX_BYTES_FLOOR


def test_unscannable_source_keeps_the_default_and_warns(tmp_path: Path) -> None:
    """A non-local / unreadable source degrades honestly: default cap + a visible warning, no crash.

    `scan_source` observes nothing for a Storage URI it cannot walk, so the generator falls back to
    the floor — and the operator is warned rather than silently handed a cap that may quarantine
    everything.
    """
    scan = scan_source("s3://bucket/corpus")
    assert scan.scanned is False
    assert scan.largest_bytes == 0
    assert build_stack_config(
        _answers(tmp_path)  # largest_source_bytes defaults to 0 == "nothing observed"
    ).count(f"max_bytes: {INTAKE_MAX_BYTES_FLOOR}") == 1

    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "e.stack.yaml"
    result = runner.invoke(
        app,
        ["setup", "--non-interactive", "--source", f"file://{empty}", "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "could not read any files under the source folder" in result.output
    assert _intake_cap(out.read_text(encoding="utf-8")) == INTAKE_MAX_BYTES_FLOOR


def test_cap_is_ceilinged_and_over_cap_files_warn_loudly(tmp_path: Path) -> None:
    """Past the ceiling the cap stops tracking the corpus — and the operator is LOUDLY warned that
    those files will be quarantined (never a silent empty corpus)."""
    src, size = _big_source(tmp_path, INTAKE_MAX_BYTES_CEILING + 1)  # sparse: no real 2 GiB written
    out = tmp_path / "huge.stack.yaml"
    result = runner.invoke(
        app,
        ["setup", "--non-interactive", "--source", src, "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    cap = _intake_cap(out.read_text(encoding="utf-8"))
    assert cap == INTAKE_MAX_BYTES_CEILING  # the guard's backstop holds
    assert size > cap
    assert "WARNING" in result.output
    assert "OVER the" in result.output and "intake cap" in result.output


def test_sota_profile_sizes_the_cap_too(tmp_path: Path) -> None:
    """The sizing rides on BOTH profiles (the SOTA stack has the same intake screen)."""
    answers = SetupAnswers(
        source=_src_dir(tmp_path),
        storage=f"file://{tmp_path}/out",
        parser=PARSER_UNIVERSAL,
        profile=PROFILE_SOTA,
        largest_source_bytes=33_898_584,
        extensions=("pdf",),
    )
    assert _intake_cap(build_stack_config(answers)) >= 33_898_584


# -- the domain hint reaches schema induction ---------------------------------
#
# The defect this pins: `--domain` was accepted, prompted for, and then only became a YAML comment
# — it never reached the `induce` Stage, so an operator's domain hint changed nothing about the
# induced schema (wired-but-dead).


def test_domain_hint_reaches_the_induce_stage(tmp_path: Path) -> None:
    """`--domain` lands on `induce.config.domain` (where the inducer Provider reads it), not only in
    a comment — so it genuinely steers schema induction."""
    out = tmp_path / "d.stack.yaml"
    hint = "German customs law: tariff classification and export-control rulings"
    result = runner.invoke(
        app,
        [
            "setup",
            "--non-interactive",
            "--source",
            _src_dir(tmp_path),
            "--storage",
            f"file://{tmp_path}/out",
            "--domain",
            hint,
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    text = out.read_text(encoding="utf-8")
    induce = _stage(text, "induce")
    assert induce.config["domain"] == hint  # the Provider's own config key
    assert f"# Domain hint: {hint}" in text  # provenance comment KEPT
    validate_config_text(text)


def test_no_domain_hint_is_byte_identical_to_before(tmp_path: Path) -> None:
    """Backward compatibility: no hint ⇒ no `domain` key and no header note at all."""
    text = build_stack_config(_answers(tmp_path))
    assert "domain" not in _stage(text, "induce").config
    assert "Domain hint" not in text


def test_multiline_domain_hint_stays_one_comment_line(tmp_path: Path) -> None:
    """A hint with newlines cannot break out of the YAML comment (it would corrupt the config)."""
    answers = SetupAnswers(
        source=_src_dir(tmp_path),
        storage=f"file://{tmp_path}/out",
        parser=PARSER_UNIVERSAL,
        domain="customs law\nstages: []\n",
        extensions=("pdf",),
    )
    text = build_stack_config(answers)
    validate_config_text(text)  # would raise if the hint escaped the comment
    assert _stage(text, "induce").config["domain"] == "customs law stages: []"


# -- parser recommendation ----------------------------------------------------


def test_recommend_parser_by_file_types() -> None:
    assert recommend_parser(("pdf",)) == PARSER_PDF_TEXT  # born-digital PDF
    assert recommend_parser(("png", "jpg")) == PARSER_OCR  # scans/images
    assert recommend_parser(("pdf", "docx", "png")) == PARSER_UNIVERSAL  # mixed


# -- error / warning paths ----------------------------------------------------


def test_missing_source_dir_is_a_clear_error(tmp_path: Path) -> None:
    """A non-existent source folder fails loudly (exit 2) with a clear message, no config written."""
    out = tmp_path / "nope.stack.yaml"
    result = runner.invoke(
        app,
        [
            "setup",
            "--non-interactive",
            "--source",
            f"file://{tmp_path}/does-not-exist",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 2
    assert "does not exist" in result.output
    assert not out.exists()


def test_unset_llm_key_warns_clearly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unset LLM key is a clear warning (not a crash); the config is still generated + saved."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = tmp_path / "s.stack.yaml"
    result = runner.invoke(
        app,
        [
            "setup",
            "--non-interactive",
            "--source",
            _src_dir(tmp_path),
            "--storage",
            f"file://{tmp_path}/out",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "OPENROUTER_API_KEY is not set" in result.output
    assert out.exists()


# -- process runs a saved config ----------------------------------------------


def test_process_loads_and_runs_a_saved_config(tmp_path: Path) -> None:
    """`latence process --config <stub>` loads the saved stack and runs it end-to-end (CPU stub).

    A minimal plaintext stack (no heavy providers) proves the process command's load+run plumbing
    — the same execution path as `latence run` — without needing a real corpus or GPU models.
    """
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "hello.txt").write_text("the framework reads the folder\n", encoding="utf-8")
    store = f"file://{tmp_path}/store"
    config = tmp_path / "latence.stack.yaml"
    config.write_text(
        f"""
name: process-stub
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config: {{path: 'file://{docs}'}}
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [parse]
""",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["process", "--config", str(config), "--run-id", "proc"])
    assert result.exit_code == 0, result.output
    assert '"pipeline_name": "process-stub"' in result.output


def test_process_missing_config_is_a_clear_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["process", "--config", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 2
    assert "Config not found" in result.output
