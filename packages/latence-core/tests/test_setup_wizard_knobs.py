"""Every wizard knob proven reachable END-TO-END: field → CLI flag → emitted YAML → the provider.

The extended-E2E ticket set (``issues/e2e-extended/TICKETS.md``) records the same defect three
times over: a Provider exposes a tuning knob, the operator has no way to reach it, and the run is
silently pinned to whatever literal the recipe happened to carry. ``rasterize.py``'s ``DEFAULT_DPI``
/ ``DEFAULT_MAX_LONGEST_DIM`` (T-OCR-3), the GLiNER2 micro-batch + thread bounds (T-WIZ-3) and the
chunk token budget (T-WIZ-1) were all unreachable from ``latence setup``.

A knob is only *done* when all four hops hold, so every knob here gets one test per hop and the
fourth hop instantiates the **real Provider class** with the **wizard's own emitted config** — no
fake, no mock, no skip. The heavy Provider packages are installed ``--no-deps`` from source
(``scripts/verify-local.sh``), and every constructor exercised below is pure config validation, so
this stays CPU-only and offline.

The fifth thing these tests carry is the pod measurement that makes T-WIZ-1 dangerous:
``chunk.markdown`` counts in its OWN tokenizer, ~1.2× *fewer* tokens than the mdeberta-v3 encoder
GLiNER2 runs, so a 768-chunker-token budget measured **median 666 / MAX 921 mdeberta tokens** while
``perf.py`` bound the model side at 768 — 2 chunks in 7 silently truncated. Asserting the YAML says
768 proves nothing; these tests assert the **model-side** bound covers the emitted budget, and that
widening it does not widen the padded micro-batch past the pod-safe point.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from latence_core.chunking import ChunkBudget, chunk_document, estimate_tokens
from latence_core.cli import app
from latence_core.providers.perf import (
    DEFAULT_GLINER_MAX_LEN,
    estimate_token_length,
    iter_token_budget_microbatches,
    resolve_intra_op_threads,
    resolve_max_len,
    resolve_microbatch_bounds,
)
from latence_core.setup.recipe import (
    CHUNK_MAX_TOKENS,
    CHUNK_MIN_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    DEFAULT_RASTER_DPI,
    DEFAULT_RASTER_MAX_LONGEST_DIM,
    MDEBERTA_TOKEN_EXPANSION,
    PARSER_OCR,
    PARSER_OCR_VLLM,
    PARSER_UNIVERSAL,
    POD_SAFE_PADDED_POSITIONS,
    PROFILE_LITE,
    PROFILE_SOTA,
    microbatch_max_texts_for,
    model_window_for,
)
from latence_core.setup.wizard import SetupAnswers, build_pipeline_dict, build_stack_config
from latence_core.stages.chunk import MarkdownChunker
from typer.testing import CliRunner

# The real Providers that READ these knobs — the fourth hop. Installed --no-deps from source in
# CI + scripts/verify-local.sh; every constructor below is pure config validation (no weights, no
# torch import, no network), so importing them keeps this suite CPU-only and offline.
from latence_extract_gliner2.provider import Gliner2FusedExtractor
from latence_parser_lighton.provider import LightOnOcrParser
from latence_parser_lighton.rasterize import DEFAULT_DPI, DEFAULT_MAX_LONGEST_DIM
from latence_parser_lighton_vllm.provider import LightOnVllmParser
from latence_parser_render.provider import UniversalRenderParser
from latence_pii_gliner2.provider import Gliner2PIIRedactor

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator

runner = CliRunner()

# tests/ -> latence-core/ -> packages/ -> repo root. Real, committed prose is the chunker input for
# the model-window test: a synthetic string of repeated words would understate the chunker→mdeberta
# expansion the pod measured on real documents.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_REAL_TEXT_FILES = ("README.md", "CONTEXT.md", "docs/DEEP-DIVE.md")


def _src_dir(tmp_path: Path) -> str:
    d = tmp_path / "corpus"
    d.mkdir(exist_ok=True)
    (d / "a.pdf").write_text("a", encoding="utf-8")
    return f"file://{d}"


def _answers(tmp_path: Path, **overrides: Any) -> SetupAnswers:
    base: dict[str, Any] = {
        "source": _src_dir(tmp_path),
        "storage": f"file://{tmp_path}/out",
        "parser": PARSER_UNIVERSAL,
        "device": "auto",
        "extensions": ("pdf",),
    }
    base.update(overrides)
    return SetupAnswers(**base)


def _stage(answers: SetupAnswers, name: str) -> dict[str, Any]:
    stages = build_pipeline_dict(answers)["stages"]
    assert isinstance(stages, list)
    for block in stages:
        assert isinstance(block, dict)
        if block.get("name") == name:
            return block
    msg = f"no {name!r} stage in the emitted stack"
    raise AssertionError(msg)


def _config(answers: SetupAnswers, name: str) -> dict[str, Any]:
    cfg = _stage(answers, name).get("config", {})
    assert isinstance(cfg, dict)
    return cfg


def _real_chunks(budget: ChunkBudget) -> Iterator[str]:
    """Every chunk the REAL chunker produces over real committed prose at ``budget``."""
    for rel in _REAL_TEXT_FILES:
        path = _REPO_ROOT / rel
        assert path.is_file(), f"the model-window test needs real prose at {rel}"
        for chunk in chunk_document(path.read_text(encoding="utf-8"), budget):
            yield chunk.text


# ==============================================================================================
# T-WIZ-1 — the chunk token budget is an ANSWER, and the model window COVERS it
# ==============================================================================================


def test_chunk_budget_default_is_the_gliner2_window(tmp_path: Path) -> None:
    """Hop 3 (default): the emitted chunk Stage carries the 768 budget, not the old 640 literal."""
    assert _config(_answers(tmp_path), "chunk") == {
        "max_tokens": 768,
        "overlap_tokens": 80,
        "min_tokens": 8,
    }
    assert (CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS, CHUNK_MIN_TOKENS) == (768, 80, 8)


def test_chunk_budget_is_an_answer_not_a_literal(tmp_path: Path) -> None:
    """Hops 1→3: a SetupAnswers field reaches the emitted `chunk.config`, on both profiles."""
    for profile in (PROFILE_LITE, PROFILE_SOTA):
        answers = _answers(
            tmp_path,
            profile=profile,
            parser=PARSER_OCR_VLLM if profile == PROFILE_SOTA else PARSER_UNIVERSAL,
            chunk_max_tokens=512,
            chunk_overlap_tokens=64,
            chunk_min_tokens=16,
        )
        assert _config(answers, "chunk") == {
            "max_tokens": 512,
            "overlap_tokens": 64,
            "min_tokens": 16,
        }


def test_cli_flags_thread_the_chunk_budget(tmp_path: Path) -> None:
    """Hop 2: `--chunk-max-tokens` / `--chunk-overlap-tokens` / `--chunk-min-tokens` reach the file."""
    out = tmp_path / "stack.yaml"
    result = runner.invoke(
        app,
        [
            "setup",
            "--non-interactive",
            "--source",
            _src_dir(tmp_path),
            "--storage",
            f"file://{tmp_path}/out",
            "--chunk-max-tokens",
            "512",
            "--chunk-overlap-tokens",
            "64",
            "--chunk-min-tokens",
            "16",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    text = out.read_text(encoding="utf-8")
    assert "max_tokens: 512" in text
    assert "overlap_tokens: 64" in text
    assert "min_tokens: 16" in text


def test_emitted_chunk_budget_is_what_the_chunker_actually_runs(tmp_path: Path) -> None:
    """Hop 4: `chunk.markdown` built from the emitted config carries exactly that budget."""
    answers = _answers(tmp_path, chunk_max_tokens=704, chunk_overlap_tokens=48, chunk_min_tokens=12)
    chunker = MarkdownChunker(_config(answers, "chunk"))
    assert chunker._budget == ChunkBudget(max_tokens=704, overlap_tokens=48, min_tokens=12)  # noqa: SLF001


@pytest.mark.parametrize("profile", [PROFILE_LITE, PROFILE_SOTA])
@pytest.mark.parametrize("budget", [512, 640, 768, 1024])
def test_model_window_covers_the_emitted_chunk_budget(
    tmp_path: Path, profile: str, budget: int
) -> None:
    """The TRAP, arithmetically: the emitted model-side `max_len` covers the emitted chunk budget.

    ``perf.py:700-707`` binds ``max_len`` on the model side and the gliner-family encoder SILENTLY
    truncates past it. The pod measured 768 chunker tokens → max **921** mdeberta tokens, so a
    stack that emits a chunk budget without widening the window loses the tail of ~2 chunks in 7.
    Both Stages that run the encoder are checked, through the Provider's OWN resolver (which
    applies the 768 default), so a stack that merely OMITS `max_len` fails here too.
    """
    answers = _answers(
        tmp_path,
        profile=profile,
        parser=PARSER_OCR_VLLM if profile == PROFILE_SOTA else PARSER_UNIVERSAL,
        chunk_max_tokens=budget,
    )
    needed = math.ceil(budget * MDEBERTA_TOKEN_EXPANSION)
    for stage in ("extract", "redact"):
        window = resolve_max_len(_config(answers, stage))
        assert window >= needed, (
            f"{profile}/{stage}: emitted max_len={window} is below the {needed} mdeberta tokens a "
            f"{budget}-chunker-token chunk can reach (~{MDEBERTA_TOKEN_EXPANSION}× expansion)"
        )


def test_the_pod_measured_921_fits_the_default_window() -> None:
    """The literal pod data point: 768 chunker tokens hit 921 mdeberta tokens. It must fit."""
    assert model_window_for(768) >= 921
    # And the historical pairing still holds: the old 640 budget derives exactly the old 768 window,
    # so the constant reproduces the calibration it generalises rather than replacing it.
    assert model_window_for(640) == DEFAULT_GLINER_MAX_LEN


def test_model_side_tokens_of_real_chunks_stay_inside_the_emitted_window(tmp_path: Path) -> None:
    """The TRAP, empirically: run the REAL chunker on real prose at the emitted budget.

    Two independent model-side estimates must both fit under the emitted window: the pod's measured
    chunker→mdeberta expansion applied to the chunker's own count, and the Provider's own
    characters-per-token proxy (``estimate_token_length``, which CLIPS at ``max_len`` — a clipped
    value is exactly the silent truncation this test exists to catch).
    """
    answers = _answers(tmp_path)
    window = resolve_max_len(_config(answers, "extract"))
    chunk_cfg = _config(answers, "chunk")
    budget = ChunkBudget(
        max_tokens=int(chunk_cfg["max_tokens"]),
        overlap_tokens=int(chunk_cfg["overlap_tokens"]),
        min_tokens=int(chunk_cfg["min_tokens"]),
    )
    seen = 0
    for text in _real_chunks(budget):
        seen += 1
        expanded = math.ceil(estimate_tokens(text) * MDEBERTA_TOKEN_EXPANSION)
        assert expanded <= window, (
            f"a real chunk of {estimate_tokens(text)} chunker tokens reaches ~{expanded} mdeberta "
            f"tokens, past the emitted max_len={window} — silent truncation"
        )
        assert estimate_token_length(text, window) < window, (
            "the provider's own model-side token estimate is CLIPPED at max_len for a real chunk "
            "— the chunk budget outruns the window"
        )
    assert seen >= 20, f"expected a real corpus of chunks to test against, got {seen}"


def test_widening_the_window_does_not_widen_the_padded_microbatch(tmp_path: Path) -> None:
    """Widening `max_len` must not cost VRAM: the padded micro-batch stays at the pod-safe point.

    GLiNER2 pads every text in a micro-batch up to the batch's longest sequence, so the padded
    worst case is ``len(batch) × longest``. The pod measured batch-4 × 768 tokens peaking at
    **21.6 GiB** — and D3 records that on the 24 GB card that is 2.4 GiB of headroom. So a wider
    window has to buy its positions back out of ``microbatch_max_texts``, which is exactly what
    :func:`microbatch_max_texts_for` does.
    """
    answers = _answers(tmp_path)
    extract = _config(answers, "extract")
    window = resolve_max_len(extract)
    max_texts, token_budget = resolve_microbatch_bounds(extract)
    assert max_texts * window <= POD_SAFE_PADDED_POSITIONS, (
        f"padded worst case {max_texts}×{window} exceeds the pod-safe {POD_SAFE_PADDED_POSITIONS} "
        "token positions (batch-4 × 768 peaked at 21.6 GiB on a 32 GB card)"
    )

    chunk_cfg = _config(answers, "chunk")
    texts = list(
        _real_chunks(
            ChunkBudget(
                max_tokens=int(chunk_cfg["max_tokens"]),
                overlap_tokens=int(chunk_cfg["overlap_tokens"]),
                min_tokens=int(chunk_cfg["min_tokens"]),
            )
        )
    )
    for start, end in iter_token_budget_microbatches(
        texts, max_len=window, max_texts=max_texts, token_budget=token_budget
    ):
        batch = texts[start:end]
        longest = max(estimate_token_length(t, window) for t in batch)
        assert len(batch) * longest <= POD_SAFE_PADDED_POSITIONS


# ==============================================================================================
# T-OCR-3 — DPI + resolution policy, reachable on every parser path
# ==============================================================================================


def test_raster_defaults_mirror_the_rasteriser(tmp_path: Path) -> None:
    """The wizard's raster defaults ARE LightOn's published values, read off the rasteriser itself.

    ``latence-core`` cannot depend on the parser package, so the two constants are mirrored — and a
    mirror that nothing checks is the drift ADR-0041 §7a recorded. This is the check.
    """
    assert (DEFAULT_RASTER_DPI, DEFAULT_RASTER_MAX_LONGEST_DIM) == (
        DEFAULT_DPI,
        DEFAULT_MAX_LONGEST_DIM,
    )
    assert (DEFAULT_RASTER_DPI, DEFAULT_RASTER_MAX_LONGEST_DIM) == (200, 1540)


def test_raster_defaults_leave_every_parse_block_byte_identical(tmp_path: Path) -> None:
    """T-OCR-3 is REACHABILITY, not retuning: unset flags emit exactly the pre-ticket blocks."""
    universal = _config(_answers(tmp_path, parser=PARSER_UNIVERSAL), "parse")
    assert universal == {
        "device": "auto",
        "dpi": 200,
        "soffice_timeout_s": 120,
        "ocr": {"provider": PARSER_OCR, "config": {"device": "auto", "max_new_tokens": 4096}},
    }
    assert _config(_answers(tmp_path, parser=PARSER_OCR), "parse") == {
        "device": "auto",
        "max_new_tokens": 4096,
    }
    sota = _answers(tmp_path, profile=PROFILE_SOTA, parser=PARSER_OCR_VLLM, device="cuda")
    assert _config(sota, "parse") == {
        "base_url": "http://localhost:8000/v1",
        "model": "lightonocr",
        "api_key": "EMPTY",  # pragma: allowlist secret
        "max_concurrency": 8,
    }


def test_raster_knobs_reach_every_parser_path_and_their_providers(tmp_path: Path) -> None:
    """Hops 1→4 on all three rasterising parse paths, through the REAL Provider constructors."""
    # parser.lighton — the in-process OCR VLM rasterises PDFs AND downscales images itself.
    direct = _config(_answers(tmp_path, parser=PARSER_OCR, ocr_dpi=300, ocr_max_longest_dim=1120), "parse")
    assert direct["dpi"] == 300
    assert direct["max_longest_dim"] == 1120
    parser = LightOnOcrParser(direct)
    assert parser._dpi == 300  # noqa: SLF001
    assert parser._max_longest_dim == 1120  # noqa: SLF001

    # parser.lighton_vllm — the served-OCR path rasterises locally, POSTs the images.
    sota = _answers(
        tmp_path,
        profile=PROFILE_SOTA,
        parser=PARSER_OCR_VLLM,
        device="cuda",
        ocr_dpi=144,
        ocr_max_longest_dim=1512,
    )
    served = _config(sota, "parse")
    assert served["dpi"] == 144
    assert served["max_longest_dim"] == 1512
    vllm = LightOnVllmParser(served)
    assert vllm._dpi == 144  # noqa: SLF001
    assert vllm._max_longest_dim == 1512  # noqa: SLF001

    # parser.render — the universal renderer owns the PDF DPI; its OCR sub-provider owns the
    # downscale (the renderer hands it page IMAGES, never a PDF), so the knob rides the sub-block.
    universal = _config(
        _answers(tmp_path, parser=PARSER_UNIVERSAL, ocr_dpi=288, ocr_max_longest_dim=1400), "parse"
    )
    assert universal["dpi"] == 288
    render = UniversalRenderParser(universal)
    assert render._dpi == 288  # noqa: SLF001
    sub = universal["ocr"]["config"]
    assert sub["max_longest_dim"] == 1400
    assert LightOnOcrParser(sub)._max_longest_dim == 1400  # noqa: SLF001


def test_cli_flags_thread_the_raster_knobs(tmp_path: Path) -> None:
    """Hop 2 for T-OCR-3: `--ocr-dpi` / `--ocr-max-longest-dim` reach the saved file."""
    out = tmp_path / "stack.yaml"
    result = runner.invoke(
        app,
        [
            "setup",
            "--non-interactive",
            "--source",
            _src_dir(tmp_path),
            "--storage",
            f"file://{tmp_path}/out",
            "--parser",
            PARSER_OCR,
            "--ocr-dpi",
            "300",
            "--ocr-max-longest-dim",
            "1120",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    text = out.read_text(encoding="utf-8")
    assert "dpi: 300" in text
    assert "max_longest_dim: 1120" in text


# ==============================================================================================
# T-WIZ-3 — the GLiNER2 performance knobs
# ==============================================================================================


def test_perf_knobs_are_absent_until_the_operator_sets_them(tmp_path: Path) -> None:
    """Unset ⇒ no key ⇒ the Provider's own default rides (nothing pinned behind the operator)."""
    answers = _answers(tmp_path, chunk_max_tokens=640)  # window 768 ⇒ the provider default bound
    for stage in ("extract", "redact"):
        cfg = _config(answers, stage)
        assert "microbatch_max_texts" not in cfg
        assert "microbatch_token_budget" not in cfg
        assert "intra_op_threads" not in cfg


def test_perf_knobs_reach_both_gliner2_providers(tmp_path: Path) -> None:
    """Hops 1→4 for T-WIZ-3, through the REAL GLiNER2 extractor + redactor constructors."""
    answers = _answers(
        tmp_path,
        microbatch_max_texts=2,
        microbatch_token_budget=1536,
        intra_op_threads=3,
    )
    extract_cfg = _config(answers, "extract")
    redact_cfg = _config(answers, "redact")
    for cfg in (extract_cfg, redact_cfg):
        assert cfg["microbatch_max_texts"] == 2
        assert cfg["microbatch_token_budget"] == 1536
        assert cfg["intra_op_threads"] == 3

    extractor = Gliner2FusedExtractor(extract_cfg)
    assert extractor._microbatch_max_texts == 2  # noqa: SLF001
    assert extractor._microbatch_token_budget == 1536  # noqa: SLF001
    assert resolve_intra_op_threads(extractor._config) == 3  # noqa: SLF001

    redactor = Gliner2PIIRedactor(redact_cfg)
    assert redactor._microbatch_max_texts == 2  # noqa: SLF001
    assert redactor._microbatch_token_budget == 1536  # noqa: SLF001
    assert resolve_intra_op_threads(redactor._config) == 3  # noqa: SLF001


def test_operator_microbatch_cap_wins_over_the_derived_one(tmp_path: Path) -> None:
    """An explicit answer beats the window-derived default — the operator is never overruled."""
    derived = _config(_answers(tmp_path), "extract")
    assert derived["microbatch_max_texts"] == microbatch_max_texts_for(
        resolve_max_len(derived)
    )
    pinned = _config(_answers(tmp_path, microbatch_max_texts=1), "extract")
    assert pinned["microbatch_max_texts"] == 1


def test_cli_flags_thread_the_perf_knobs(tmp_path: Path) -> None:
    """Hop 2 for T-WIZ-3."""
    out = tmp_path / "stack.yaml"
    result = runner.invoke(
        app,
        [
            "setup",
            "--non-interactive",
            "--source",
            _src_dir(tmp_path),
            "--storage",
            f"file://{tmp_path}/out",
            "--microbatch-max-texts",
            "2",
            "--microbatch-token-budget",
            "1536",
            "--intra-op-threads",
            "3",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    text = out.read_text(encoding="utf-8")
    assert "microbatch_max_texts: 2" in text
    assert "microbatch_token_budget: 1536" in text
    assert "intra_op_threads: 3" in text


# ==============================================================================================
# The determinism contract these knobs must not break
# ==============================================================================================


def test_identical_answers_still_yield_byte_identical_yaml(tmp_path: Path) -> None:
    """ADR-0041 §3: the generator stays a pure function of its answers, knobs and all."""
    kwargs: dict[str, Any] = {
        "chunk_max_tokens": 704,
        "chunk_overlap_tokens": 48,
        "chunk_min_tokens": 12,
        "ocr_dpi": 300,
        "ocr_max_longest_dim": 1120,
        "microbatch_max_texts": 2,
        "microbatch_token_budget": 1536,
        "intra_op_threads": 3,
    }
    first = build_stack_config(_answers(tmp_path, **kwargs))
    second = build_stack_config(_answers(tmp_path, **kwargs))
    assert first == second
