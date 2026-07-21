"""The enterprise-sota profile's PARSE PATH must fit the corpus (dogfood-2-wizard).

The enterprise-sota profile is a GPU-pod stack whose production parse path is the served-vLLM
LightOn OCR endpoint (``parser.lighton_vllm``). But that provider ONLY rasterizes document pages to
images and POSTs them to the OCR server — it has NO plain-text passthrough. So when the corpus is
born-digital plain text (``.txt`` / ``.md`` / ``.jsonl`` — the common enterprise export/log/ticket
dump), forcing OCR is wrong: on a text-corpus pod the OCR box is (correctly) not served, and every
document degrades to ``PARSE_ERROR`` with no content downstream. The wizard — the product's mandated
entry gate — must yield a RUNNABLE stack for that corpus.

Before the fix, ``_build_sota_stages`` hard-wired ``parser.lighton_vllm`` and the CLI suppressed the
parser-choice prompt under ``--profile enterprise-sota``, so a ``SetupAnswers`` carrying any other
parser was ignored — a text corpus could never get a text parser through the wizard. These tests pin
the fix: the SOTA profile routes a detected plaintext corpus to the CPU ``parser.plaintext`` parser
(no OCR endpoint) while keeping every other SOTA signal intact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from latence_core.pipeline import Pipeline
from latence_core.setup import (
    SetupAnswers,
    build_stack_config,
    recommend_sota_parser,
    validate_config_text,
)
from latence_core.setup.wizard import PROFILE_SOTA

if TYPE_CHECKING:
    from pathlib import Path


def _sota_answers(tmp_path: Path, *, parser: str, extensions: tuple[str, ...]) -> SetupAnswers:
    src = tmp_path / "corpus"
    src.mkdir(exist_ok=True)
    (src / "a.txt").write_text("hello world", encoding="utf-8")
    return SetupAnswers(
        source=f"file://{src}",
        storage=f"file://{tmp_path}/out",
        parser=parser,
        device="cuda",
        pii=True,
        extract_on="unmasked",
        profile=PROFILE_SOTA,
        extensions=extensions,
    )


def _stage(text: str, name: str) -> dict[str, Any]:
    stage = next(s for s in Pipeline.from_yaml_text(text).stages if s.name == name)
    return {"provider": stage.provider, "config": dict(stage.config)}


def test_recommend_sota_parser_routes_plaintext_vs_ocr() -> None:
    """A wholly-text corpus routes to the plaintext parser; anything else keeps the served OCR."""
    assert recommend_sota_parser(("txt", "md", "jsonl")) == "parser.plaintext"
    assert recommend_sota_parser(("txt",)) == "parser.plaintext"
    assert recommend_sota_parser(("markdown",)) == "parser.plaintext"
    # A single non-text type present anywhere ⇒ the served-vLLM OCR path (its territory).
    assert recommend_sota_parser(("txt", "pdf")) == "parser.lighton_vllm"
    assert recommend_sota_parser(("png",)) == "parser.lighton_vllm"
    assert recommend_sota_parser(("docx",)) == "parser.lighton_vllm"
    # An empty / undetectable scan is not assumed text — keep the OCR default.
    assert recommend_sota_parser(()) == "parser.lighton_vllm"


def test_sota_plaintext_corpus_emits_a_runnable_text_parser(tmp_path: Path) -> None:
    """The core regression: a plaintext SOTA corpus must parse with `parser.plaintext`, NOT OCR.

    Pre-fix `_build_sota_stages` ignored `answers.parser` and always emitted `parser.lighton_vllm`
    (an OCR-only endpoint with no text passthrough), so this assertion failed — the stack could not
    parse a .txt corpus on a pod where the OCR box is not served.
    """
    text = build_stack_config(
        _sota_answers(tmp_path, parser="parser.plaintext", extensions=("jsonl", "md", "txt"))
    )
    parse = _stage(text, "parse")
    assert parse["provider"] == "parser.plaintext"
    # The plaintext parser needs NO served OCR endpoint — no base_url / api_key leaks into its config.
    assert "base_url" not in parse["config"]
    assert "api_key" not in parse["config"]


def test_sota_plaintext_keeps_every_other_sota_signal(tmp_path: Path) -> None:
    """Switching the parse path must not strip the SOTA spine: pagerank + BM25 + Granite stay."""
    text = build_stack_config(
        _sota_answers(tmp_path, parser="parser.plaintext", extensions=("txt",))
    )
    graph = _stage(text, "graph")
    export = _stage(text, "export_corpus")
    assert graph["config"]["graph_features"]["provider"] == "graph_features.pagerank"
    assert "bm25" in export["config"]
    assert "granite" in str(export["config"]["embedder"]).lower()
    # The neural GLinker disambiguator is still wired (this is the composed pod stack).
    assert _stage(text, "disambiguate")["provider"] == "disambiguation.glinker"


def test_sota_ocr_corpus_still_emits_the_served_vllm_parser(tmp_path: Path) -> None:
    """A scanned/PDF corpus keeps the served-vLLM OCR parse path (no regression to that route)."""
    text = build_stack_config(
        _sota_answers(tmp_path, parser="parser.lighton_vllm", extensions=("pdf", "png"))
    )
    parse = _stage(text, "parse")
    assert parse["provider"] == "parser.lighton_vllm"
    assert parse["config"]["base_url"]  # the served endpoint is wired
    assert parse["config"]["model"]


def test_sota_plaintext_stack_dry_validates(tmp_path: Path) -> None:
    """The emitted plaintext SOTA stack passes the wizard's own dry validation (DAG + wiring)."""
    text = build_stack_config(
        _sota_answers(tmp_path, parser="parser.plaintext", extensions=("txt", "md"))
    )
    validate_config_text(text)  # raises SetupConfigError on a broken/unwired config


def test_sota_config_is_byte_identical_for_identical_answers(tmp_path: Path) -> None:
    """Determinism holds across the new parse branch: same answers ⇒ byte-identical config."""
    a = _sota_answers(tmp_path, parser="parser.plaintext", extensions=("txt", "md"))
    assert build_stack_config(a) == build_stack_config(a)
