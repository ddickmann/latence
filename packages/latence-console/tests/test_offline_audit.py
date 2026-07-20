"""The offline audit itself is under test (audit R1-console_readonly-12).

``render.external_references`` is *the* proof the console's pages are fully offline — four test
modules assert it returns ``[]`` against rendered bytes. An audit that cannot see a whole class of
off-machine URL makes those assertions hollow, so the audit gets its own adversarial tests: feed it
references it MUST catch, and local forms it must NOT flag.
"""

from __future__ import annotations

import pytest
from latence_console.render import external_references

# Every one of these reaches off this machine when the page is opened. The protocol-relative
# (scheme-less) forms are the regression: they were invisible to the audit.
_OFF_MACHINE = [
    '<img src="//cdn.example.com/a.png">',
    '<a href="//evil.example.com/x">click</a>',
    "@import url(//fonts.example.com/f.css);",
    "background:url(//cdn.example.com/bg.png)",
    "plain text //evil.example.com in a paragraph",
    '<script src="//analytics.example.com/t.js"></script>',
    '<a href="//[2001:db8::1]/x">v6</a>',
    "s3://bucket.example.com/prefix",
    '<img src="https://evil.example.com/a.png">',
    '<img src="http://evil.example.com/a.png">',
]

# None of these leaves the machine: loopback, or same-origin paths the console really emits.
_LOCAL = [
    '<a href="/run/run-0001">run</a>',
    '<a href="/run/run-0001/chunks?page=2">chunks</a>',
    "http://127.0.0.1:8765/run/run-0001",
    '<a href="//127.0.0.1:8765/x">loopback</a>',
    '<a href="//localhost:8765/x">loopback</a>',
    '<a href="//[::1]:8765/x">loopback</a>',
    "file:///tmp/corpus/run-0001",
    "file:///private/var/folders/abc/corpus",
    "file://localhost/tmp/corpus",
    "a fraction 3/4 and a comment // not a url",
    "<!doctype html><html lang=\"en\"><head><meta charset='utf-8'></head></html>",
]


@pytest.mark.parametrize("html", _OFF_MACHINE)
def test_off_machine_reference_is_reported(html: str) -> None:
    """The audit must see every off-machine reference, scheme or not."""
    assert external_references(html), f"audit blind to: {html}"


@pytest.mark.parametrize("html", _LOCAL)
def test_local_reference_is_not_reported(html: str) -> None:
    """Loopback and same-origin paths are not off-machine — no false positives."""
    assert external_references(html) == [], f"audit false-positive on: {html}"


def test_loopback_lookalike_host_is_not_whitelisted() -> None:
    """A prefix match would let ``localhost.evil.example.com`` through as 'loopback'."""
    assert external_references('<img src="//localhost.evil.example.com/a.png">')
    assert external_references('<img src="//127.0.0.1.evil.example.com/a.png">')


def test_reported_reference_is_the_matched_text_in_document_order() -> None:
    """The contract callers rely on: the offending text, in order, so failures are readable."""
    html = '<img src="//cdn.example.com/a.png"><a href="https://evil.example.com/x">x</a>'
    assert external_references(html) == [
        "//cdn.example.com",
        "https://evil.example.com",
    ]
