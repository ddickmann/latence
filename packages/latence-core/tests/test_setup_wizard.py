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
from latence_core.pipeline import Pipeline
from latence_core.setup import (
    FINANCIAL_PII_LABELS,
    SetupAnswers,
    build_stack_config,
    recommend_parser,
    validate_config_text,
)
from latence_core.setup.wizard import (
    PARSER_OCR,
    PARSER_PDF_TEXT,
    PARSER_UNIVERSAL,
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
) -> SetupAnswers:
    return SetupAnswers(
        source=_src_dir(tmp_path),
        storage=f"file://{tmp_path}/out",
        parser=parser,
        device="auto",
        pii=pii,
        extract_on=extract_on,
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
