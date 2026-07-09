"""Office dispatch tests: soffice→pdf→OCR faithful render vs the MIT text-extraction fallback.

The office readers (python-docx/openpyxl/python-pptx) and ``soffice`` are not in the dev env, so the
soffice branch and the text-extraction branch are driven by patching the seams the Provider itself
owns — proving the DISPATCH logic (soffice present → render path; absent → text path; soffice
failure → graceful degrade to text), which is the contract under test.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from latence_core.contracts import Disposition, ParserInput
from latence_parser_render import provider as prov
from latence_parser_render.provider import UniversalRenderParser

# A minimal spoof-guard-passing office file: a ZIP local-file-header prefix + arbitrary tail.
_FAKE_DOCX = b"PK\x03\x04" + b"fake docx zip body"


def _office_input(make_input: Callable[..., ParserInput], file_type: str) -> ParserInput:
    return make_input(_FAKE_DOCX, file_name=f"doc.{file_type}", file_type=file_type)


def test_soffice_present_routes_through_pdf_render(
    render_config: dict[str, Any],
    patch_renderer: None,
    monkeypatch: pytest.MonkeyPatch,
    make_pdf: Callable[[list[str]], bytes],
    make_input: Callable[..., ParserInput],
) -> None:
    """When soffice is present, an office file is converted to PDF then rasterised+OCR'd (faithful)."""
    monkeypatch.setattr(UniversalRenderParser, "_soffice_path", staticmethod(lambda: "/usr/bin/soffice"))

    def fake_convert(self: UniversalRenderParser, data: bytes, file_type: str) -> bytes:
        # soffice "renders" the office doc to a 2-page PDF the patched rasteriser will split.
        return make_pdf(["# Rendered docx\n\nFrom soffice.", "## Page two"])

    monkeypatch.setattr(UniversalRenderParser, "_soffice_convert_to_pdf", fake_convert)
    provider = UniversalRenderParser(render_config)
    record = next(iter(provider.parse([_office_input(make_input, "docx")])))
    assert record.disposition is Disposition.PARSED
    assert record.page_map is not None
    assert len(record.page_map.pages) == 2
    assert "Rendered docx" in record.content


def test_soffice_failure_degrades_to_text_path(
    render_config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    make_input: Callable[..., ParserInput],
) -> None:
    """A soffice failure is not fatal: the office file falls through to MIT text extraction."""
    monkeypatch.setattr(UniversalRenderParser, "_soffice_path", staticmethod(lambda: "/usr/bin/soffice"))

    def failing_convert(self: UniversalRenderParser, data: bytes, file_type: str) -> bytes:
        raise prov._ParseFailure("soffice conversion exited 1.")

    monkeypatch.setattr(UniversalRenderParser, "_soffice_convert_to_pdf", failing_convert)
    monkeypatch.setattr(
        UniversalRenderParser, "_docx_text", lambda self, data: "Text-extracted docx body."
    )
    provider = UniversalRenderParser(render_config)
    record = next(iter(provider.parse([_office_input(make_input, "docx")])))
    assert record.disposition is Disposition.PARSED
    assert record.content == "Text-extracted docx body."
    assert record.page_map is not None
    assert len(record.page_map.pages) == 1


def test_soffice_absent_uses_text_path_no_ocr(
    monkeypatch: pytest.MonkeyPatch, make_input: Callable[..., ParserInput]
) -> None:
    """No soffice + no ocr config: the office text path still works (it needs no OCR)."""
    monkeypatch.setattr(UniversalRenderParser, "_soffice_path", staticmethod(lambda: None))
    monkeypatch.setattr(
        UniversalRenderParser, "_docx_text", lambda self, data: "Body without soffice or OCR."
    )
    provider = UniversalRenderParser({})  # no ocr configured at all
    record = next(iter(provider.parse([_office_input(make_input, "docx")])))
    assert record.disposition is Disposition.PARSED
    assert record.content == "Body without soffice or OCR."


def test_prefer_soffice_false_forces_text_path(
    monkeypatch: pytest.MonkeyPatch, make_input: Callable[..., ParserInput]
) -> None:
    """prefer_soffice=False uses the text path even when soffice is present."""
    monkeypatch.setattr(UniversalRenderParser, "_soffice_path", staticmethod(lambda: "/usr/bin/soffice"))
    called = {"convert": False}

    def convert(self: UniversalRenderParser, data: bytes, file_type: str) -> bytes:
        called["convert"] = True
        return b"%PDF-x"

    monkeypatch.setattr(UniversalRenderParser, "_soffice_convert_to_pdf", convert)
    monkeypatch.setattr(UniversalRenderParser, "_docx_text", lambda self, data: "text-only")
    provider = UniversalRenderParser({"prefer_soffice": False})
    record = next(iter(provider.parse([_office_input(make_input, "docx")])))
    assert record.content == "text-only"
    assert called["convert"] is False


def test_xlsx_text_path_one_page_per_sheet(
    monkeypatch: pytest.MonkeyPatch, make_input: Callable[..., ParserInput]
) -> None:
    """xlsx text extraction is one markdown page per worksheet."""
    monkeypatch.setattr(UniversalRenderParser, "_soffice_path", staticmethod(lambda: None))
    monkeypatch.setattr(
        UniversalRenderParser,
        "_xlsx_page_texts",
        lambda self, data: ["# Sheet1\n\na\tb", "# Sheet2\n\nc\td"],
    )
    provider = UniversalRenderParser({})
    record = next(iter(provider.parse([_office_input(make_input, "xlsx")])))
    assert record.disposition is Disposition.PARSED
    assert record.page_map is not None
    assert len(record.page_map.pages) == 2
    assert "Sheet1" in record.content and "Sheet2" in record.content


def test_pptx_text_path_one_page_per_slide(
    monkeypatch: pytest.MonkeyPatch, make_input: Callable[..., ParserInput]
) -> None:
    """pptx text extraction is one markdown page per slide."""
    monkeypatch.setattr(UniversalRenderParser, "_soffice_path", staticmethod(lambda: None))
    monkeypatch.setattr(
        UniversalRenderParser,
        "_pptx_page_texts",
        lambda self, data: ["Slide one title", "Slide two body"],
    )
    provider = UniversalRenderParser({})
    record = next(iter(provider.parse([_office_input(make_input, "pptx")])))
    assert record.disposition is Disposition.PARSED
    assert record.page_map is not None
    assert len(record.page_map.pages) == 2


# --- the hardened soffice subprocess (no shell, timeout, no leak) ------------


def test_soffice_convert_timeout_is_graceful(monkeypatch: pytest.MonkeyPatch) -> None:
    """A soffice timeout raises _ParseFailure with no leaked content (degrades to text upstream)."""
    import subprocess

    monkeypatch.setattr(UniversalRenderParser, "_soffice_path", staticmethod(lambda: "/usr/bin/soffice"))

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="soffice", timeout=1)

    monkeypatch.setattr("latence_parser_render.provider.subprocess.run", fake_run)
    provider = UniversalRenderParser({"soffice_timeout_s": 1})
    with pytest.raises(prov._ParseFailure) as excinfo:
        provider._soffice_convert_to_pdf(_FAKE_DOCX, "docx")
    # The message names the timeout, never the input bytes (no leak).
    assert "timed out" in str(excinfo.value)
    assert "fake docx zip body" not in str(excinfo.value)


def test_soffice_convert_nonzero_exit_is_graceful(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-zero soffice exit is a graceful _ParseFailure that does NOT interpolate stderr."""

    monkeypatch.setattr(UniversalRenderParser, "_soffice_path", staticmethod(lambda: "/usr/bin/soffice"))

    class _Completed:
        returncode = 3
        stderr = b"error: /tmp/secret-path/doc.docx failed"
        stdout = b""

    monkeypatch.setattr("latence_parser_render.provider.subprocess.run", lambda *a, **k: _Completed())
    provider = UniversalRenderParser({})
    with pytest.raises(prov._ParseFailure) as excinfo:
        provider._soffice_convert_to_pdf(_FAKE_DOCX, "docx")
    assert "exited 3" in str(excinfo.value)
    assert "secret-path" not in str(excinfo.value)  # stderr not leaked


def test_soffice_subprocess_is_hardened_argv_no_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real subprocess call uses an explicit argv (no shell), a timeout, and check=False.

    The timeout/nonzero tests above verify error HANDLING but replace ``subprocess.run`` with a
    signature-swallowing lambda, so they never inspect HOW it was invoked. This captures the actual
    call arguments and asserts the defense-in-depth properties the security review requires: a list
    argv (not a shell string), ``shell`` never truthy, the operator timeout wired through, and
    ``check=False`` so a non-zero exit degrades gracefully instead of raising an untyped error.
    """
    monkeypatch.setattr(UniversalRenderParser, "_soffice_path", staticmethod(lambda: "/usr/bin/soffice"))
    captured: dict[str, Any] = {}

    class _Completed:
        returncode = 0
        stdout = b""
        stderr = b""

    def fake_run(args: Any, **kwargs: Any) -> Any:
        captured["args"] = args
        captured["kwargs"] = kwargs
        # Materialise the PDF soffice would have produced so the caller returns bytes, not fails.
        outdir = args[args.index("--outdir") + 1]
        from pathlib import Path

        (Path(outdir) / "input.pdf").write_bytes(b"%PDF-1.7 minimal")
        return _Completed()

    monkeypatch.setattr("latence_parser_render.provider.subprocess.run", fake_run)
    provider = UniversalRenderParser({"soffice_timeout_s": 7})
    pdf_bytes = provider._soffice_convert_to_pdf(_FAKE_DOCX, "docx")

    assert pdf_bytes == b"%PDF-1.7 minimal"
    argv = captured["args"]
    assert isinstance(argv, list)  # explicit argument vector, NOT a shell string
    assert argv[0] == "/usr/bin/soffice"
    assert "--headless" in argv and "--convert-to" in argv and "pdf" in argv
    # No shell interpretation (shell defaults to False and is never passed truthy).
    assert captured["kwargs"].get("shell", False) is False
    assert captured["kwargs"]["timeout"] == 7  # the operator timeout is wired through
    assert captured["kwargs"]["check"] is False  # non-zero exit degrades, never raises untyped
