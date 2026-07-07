"""Redaction recall/precision benchmark — the Hardening-bar metric for #61.

The G1 E2E found the hybrid-rule redactor masked email + phone but MISSED planted
SSN, credit-card, and secret/API-key values (partial recall). Root cause: the built-in
library had no ``aws_key``/``api_key``/``secret`` recogniser, and shipped configs
narrowed ``detectors`` to a partial set. The fix adds those recognisers and makes the
default comprehensive (all built-ins on when ``detectors`` is omitted).

This module is a deterministic, labelled-PII benchmark: a fixed corpus with a known
gold span per PII type, scored as **per-type recall + precision** against the redactor
run with its DEFAULT (comprehensive) config. It doubles as the reverting regression
guard — the per-type recall assertions FAIL the moment any recogniser is removed or the
default is narrowed, and the aggregate gate encodes the Hardening-bar number.

The benchmark is offline, seedless-deterministic (regex over fixed text), and asserts
the *bound* (recall == 1.0 on every planted type; zero false-positive types on clean
controls) rather than printing a number, so CI enforces it.
"""

from __future__ import annotations

from latence_core.contracts import (
    Classification,
    DocumentRecord,
    Provenance,
    RedactionRecord,
)
from latence_core.stages.redaction import HybridPIIRedactor

# -- the labelled PII gold set ------------------------------------------------
# Each case: a document text, and the gold (pii_type, exact substring) planted in it.
# The substrings mirror the G1 planted values (SSN 123-45-6789, CC 4111…, AKIA…,
# 'Password: hunter2') plus the rest of the built-in library, so recall is measured
# across the full comprehensive set — not just the two types G1 happened to catch.
_GOLD: tuple[tuple[str, str, str], ...] = (
    ("email", "Reach us at alice.smith@example.com for details.", "alice.smith@example.com"),
    ("phone", "Call the desk at +1 (415) 555-0132 during hours.", "+1 (415) 555-0132"),
    ("ssn", "Employee SSN on file: 123-45-6789 (confidential).", "123-45-6789"),
    ("credit_card", "Charged to card 4111 1111 1111 1111 last night.", "4111 1111 1111 1111"),
    ("ip", "The gateway responded from 192.168.10.254 at boot.", "192.168.10.254"),
    ("iban", "Wire to DE89370400440532013000 by Friday.", "DE89370400440532013000"),
    ("aws_key", "Rotate the leaked key AKIAIOSFODNN7EXAMPLE now.", "AKIAIOSFODNN7EXAMPLE"),
    ("api_key", "config api_key=sk-abc123XYZ4567890deadbeef in env.", "sk-abc123XYZ4567890deadbeef"),
    ("secret", "The dump had Password: hunter2 in plain text.", "hunter2"),
    # -- #66 extended secret recall -----------------------------------------
    # A secret whose keyword lives INSIDE a compound identifier: the plain
    # ``\bsecret\b`` word boundary misses it because ``_`` is a word char. The gold
    # substring is the credential value only (the ``secret`` recogniser's group 1).
    (
        "secret",
        "env: aws_secret_access_key=wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY set.",
        "wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY",
    ),
    # A bare ``token: <JWT>`` — the label is generic and the value is a JWT. The
    # structural ``jwt`` recogniser fires on the value's own shape (three base64url
    # segments, header starting ``eyJ``) even though ``#61``'s label set never named
    # a bare ``token``. Gold = the whole JWT.
    (
        "jwt",
        "Set header token: "
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ"
        ".dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U before send.",
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ"
        ".dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
    ),
)

# Clean control documents: NO PII of any type. A recogniser that fires here is a false
# positive — the precision half of the benchmark.
_CLEAN_CONTROLS: tuple[str, ...] = (
    "The quarterly report summarises revenue growth across three regions.",
    "Version 1.4 shipped on schedule with no regressions in the suite.",
    "Meeting notes: agenda, action items, and the follow-up date next week.",
)


def _doc(text: str, record_id: str) -> DocumentRecord:
    prov = Provenance(
        source_uri=f"file:///c/{record_id}.txt",
        file_name=f"{record_id}.txt",
        file_type="txt",
        file_size=len(text),
        document_id=f"sha256:{record_id}",
        char_start=0,
        char_end=len(text),
        page_start=1,
        page_end=1,
    )
    return DocumentRecord(
        record_id=record_id,
        provenance=prov,
        classification=Classification(language="en", sensitivity="public"),
        content=text,
    )


def _redact_all(docs: list[DocumentRecord]) -> list[RedactionRecord]:
    # DEFAULT config on purpose: no ``detectors`` key ⇒ comprehensive coverage. If a
    # future change narrows the default, this benchmark's recall drops and fails.
    return list(HybridPIIRedactor().redact(docs))


def test_per_type_recall_is_perfect_on_the_gold_set() -> None:
    """Every planted PII value is detected AND labelled with its correct type.

    Recall per type = (# gold spans of that type whose exact span is redacted with the
    matching label) / (# gold spans of that type). The benchmark requires 1.0 for every
    built-in type — the fix's guarantee. Reverting any recogniser (or mislabelling SSN
    as phone) drops that type's recall below 1.0 and fails here.
    """
    docs = [_doc(text, f"gold-{i}") for i, (_, text, _) in enumerate(_GOLD)]
    records = _redact_all(docs)
    by_id = {r.document_record_id: r for r in records}

    misses: list[str] = []
    for i, (pii_type, text, value) in enumerate(_GOLD):
        rec = by_id[f"gold-{i}"]
        start = text.index(value)
        end = start + len(value)
        hit = any(
            span.pii_type == pii_type
            and span.char_start == start
            and span.char_end == end
            for span in rec.pii_spans
        )
        if not hit:
            found = [
                (s.pii_type, text[s.char_start : s.char_end]) for s in rec.pii_spans
            ]
            misses.append(f"{pii_type}: expected {value!r} at [{start},{end}); got {found}")

    assert not misses, "Redaction recall regressed (per-type gold miss):\n" + "\n".join(misses)


def test_no_false_positive_types_on_clean_controls() -> None:
    """Precision guard: the recognisers fire on NO clean control document.

    A comprehensive default must not turn ordinary prose into a false-positive storm.
    Any span on a clean control is a precision failure and fails the benchmark.
    """
    docs = [_doc(text, f"clean-{i}") for i, text in enumerate(_CLEAN_CONTROLS)]
    records = _redact_all(docs)
    false_positives = [
        (r.document_record_id, s.pii_type, r.masked_content)
        for r in records
        for s in r.pii_spans
    ]
    assert not false_positives, f"False-positive PII on clean prose: {false_positives}"


def test_aggregate_recall_meets_hardening_bar() -> None:
    """The single benchmark number: aggregate recall over the gold set == 1.0.

    This is the Hardening-bar metric the issue asks for (labelled PII set → recall).
    Pinned at 1.0 so any recall regression across the full comprehensive set is a hard
    CI failure, not a silently-degraded number.
    """
    docs = [_doc(text, f"gold-{i}") for i, (_, text, _) in enumerate(_GOLD)]
    records = _redact_all(docs)
    by_id = {r.document_record_id: r for r in records}

    hits = 0
    for i, (pii_type, text, value) in enumerate(_GOLD):
        rec = by_id[f"gold-{i}"]
        start, end = text.index(value), text.index(value) + len(value)
        if any(
            s.pii_type == pii_type and s.char_start == start and s.char_end == end
            for s in rec.pii_spans
        ):
            hits += 1

    recall = hits / len(_GOLD)
    assert recall == 1.0, f"aggregate redaction recall {recall:.3f} < 1.0 Hardening bar"


def test_secret_value_masked_but_label_not_leaked() -> None:
    """A labelled ``password:``/``api_key=`` assignment masks ONLY the credential.

    The span must cover the secret value, never the ``Password:`` / ``api_key=`` label,
    so the masked corpus keeps the (harmless) label context while the secret is gone —
    and the raw secret never appears in the masked output.
    """
    text = "The dump had Password: hunter2 and api_key=sk-DEADbeef01234567890 exposed."
    rec = _redact_all([_doc(text, "secret-doc")])[0]
    assert "hunter2" not in rec.masked_content
    assert "sk-DEADbeef01234567890" not in rec.masked_content
    # The labels are preserved (only the value is redacted).
    assert "Password:" in rec.masked_content
    assert "api_key=" in rec.masked_content
    assert "[SECRET]" in rec.masked_content
    assert "[API_KEY]" in rec.masked_content


def test_compound_secret_key_is_redacted() -> None:
    """#66: a secret keyword inside a COMPOUND identifier is redacted (value only).

    ``aws_secret_access_key=<40-char secret>`` — the ``\\bsecret\\b`` denylist misses it
    because ``_`` is a word char, so ``secret`` is not a standalone word. The extended
    recogniser masks the credential value while keeping the harmless label. Reverting to
    ``\\bsecret\\b`` leaves the AWS secret in the clear and fails this test. Also covers
    the dash-separated and ``client_secret`` forms.
    """
    cases = (
        ("aws_secret_access_key=wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY", "wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY"),
        ("aws-secret-access-key: wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY", "wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY"),
        ("client_secret = s3cr3t-Value-01234567", "s3cr3t-Value-01234567"),
    )
    for i, (text, secret_value) in enumerate(cases):
        rec = _redact_all([_doc(text, f"compound-secret-{i}")])[0]
        assert secret_value not in rec.masked_content, f"leaked secret in {text!r}: {rec.masked_content!r}"
        # The label component is preserved; only the value is masked.
        assert "secret" in rec.masked_content.lower()
        assert any(s.pii_type == "secret" for s in rec.pii_spans), rec.pii_spans


def test_bare_token_jwt_is_redacted() -> None:
    """#66: a bare ``token: <JWT>`` is redacted by the structural JWT recogniser.

    ``#61``'s label set never named a bare ``token``, and a JWT value contains dots. The
    structural ``jwt`` recogniser fires on the value's own shape, so the JWT is masked
    even with no recognised label at all. Reverting the ``jwt`` recogniser (and the
    ``token`` label) leaves the JWT in the clear and fails this test.
    """
    jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ"
        ".dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
    )
    # Bare label, and no label at all (JWT alone on a line) — both must be caught.
    for i, text in enumerate((f"token: {jwt}", f"The captured value was {jwt} in the log.")):
        rec = _redact_all([_doc(text, f"jwt-{i}")])[0]
        assert jwt not in rec.masked_content, f"leaked JWT: {rec.masked_content!r}"
        assert any(s.pii_type == "jwt" for s in rec.pii_spans), rec.pii_spans


def test_extended_recognisers_are_linear_time_on_pathological_input() -> None:
    """#66 ReDoS guard: the compound-secret and JWT recognisers stay linear-time.

    The compound-secret recogniser has an optional leading identifier component and any
    number of trailing ones around the keyword; a naive form backtracks catastrophically
    on a long ``x_x_x…secret…`` run with no closing assignment. The JWT recogniser must
    not blow up on a long ``eyJ…`` run with missing segments. Both finish in
    milliseconds; a ReDoS-prone form would time out here.
    """
    import re
    import time

    from latence_core.stages.redaction import _BUILTIN_PATTERNS

    # Inputs that hammer exactly the constructs #66 added/changed: the compound-secret's
    # optional leading component + unbounded trailing ``_x`` components around the keyword
    # with no closing assignment, and a JWT header run with a missing segment.
    pathological = (
        "a" * 40000 + "_secret_" + "b" * 40000 + " "      # prefix+suffix, no assignment value
        + "secret" + "_a" * 20000 + " "                    # long trailing-component chain
        + "eyJ" + "A" * 60000 + " "                        # jwt header run, no dots
        + "eyJ" + "A" * 30000 + "." + "B" * 30000          # jwt two segments, third missing
    )
    # Assert the bound on the #66 recognisers directly (isolating them from unrelated
    # built-ins), so a ReDoS-prone form of THIS fix times out here.
    for name in ("secret", "jwt", "api_key"):
        compiled = re.compile(_BUILTIN_PATTERNS[name])
        start = time.perf_counter()
        list(compiled.finditer(pathological))
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"{name!r} took {elapsed:.2f}s on pathological input (possible ReDoS)"

    # And end-to-end through the redactor on the same input stays well-bounded.
    doc = _doc(pathological, "extended-redos")
    start = time.perf_counter()
    _redact_all([doc])
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"extended recognisers took {elapsed:.2f}s end-to-end (possible ReDoS)"


def test_email_recogniser_is_linear_time_on_pathological_input() -> None:
    """ReDoS guard: the ``email`` recogniser stays linear on a long dotted run.

    Pre-existing (predates #66): a naive ``\\b[local]+@[domain]+\\.tld`` backtracks
    catastrophically on a long letters-and-dots run with no ``@`` — the local and domain
    classes both contain ``.`` and letters, and ``\\b`` matches at every word/non-word
    transition, so ``finditer`` relaunches a full greedy scan from O(n) starts (measured
    ~19s for this recogniser alone, ~53s end-to-end). The linear form (lookbehind-pinned
    start + possessive local part + non-overlapping ``label.`` domain) finishes in
    milliseconds; the ReDoS-prone form times out here.
    """
    import re
    import time

    from latence_core.stages.redaction import _BUILTIN_PATTERNS

    pathological = "x." * 30000  # long letters-and-dots run, no ``@``
    compiled = re.compile(_BUILTIN_PATTERNS["email"])
    start = time.perf_counter()
    list(compiled.finditer(pathological))
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"email took {elapsed:.2f}s on pathological input (possible ReDoS)"

    # Also guard a long dotted run AFTER an ``@`` (the domain side) and end-to-end.
    after_at = "a@" + "x." * 30000
    start = time.perf_counter()
    list(compiled.finditer(after_at))
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"email took {elapsed:.2f}s on post-@ run (possible ReDoS)"

    doc = _doc(pathological, "email-redos")
    start = time.perf_counter()
    _redact_all([doc])
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"email recogniser took {elapsed:.2f}s end-to-end (possible ReDoS)"


def test_recognisers_are_linear_time_on_pathological_input() -> None:
    """ReDoS guard: a pathological input cannot induce catastrophic backtracking.

    Runs the comprehensive default over adversarial long runs that would blow up a
    naively-nested regex, under a wall-clock bound. Linear-time patterns finish in
    milliseconds; a ReDoS-prone pattern would time out here.
    """
    import time

    pathological = (
        "api_key=" + "A" * 40000 + " "
        + "password: " + "x" * 40000 + " "
        + "+1 " * 20000 + " "
        + "4" * 40000 + " "
        + "9" * 40000
    )
    doc = _doc(pathological, "redos")
    start = time.perf_counter()
    _redact_all([doc])
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"redaction took {elapsed:.2f}s on pathological input (possible ReDoS)"
