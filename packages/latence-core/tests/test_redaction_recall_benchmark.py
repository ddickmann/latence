"""Redaction recall/precision benchmark — the Hardening-bar metric for #61 + H-C1.

The G1 E2E found the hybrid-rule redactor masked email + phone but MISSED planted
SSN, credit-card, and secret/API-key values (partial recall). Root cause: the built-in
library had no ``aws_key``/``api_key``/``secret`` recogniser, and shipped configs
narrowed ``detectors`` to a partial set. The fix adds those recognisers and makes the
default comprehensive (all built-ins on when ``detectors`` is omitted).

H-C1 extends the gold set with EVERY modern secret/token format the hostile audit
found flowing unredacted at defaults — GitHub (``ghp_``/``github_pat_``), Stripe,
OpenAI, Google, Slack, GitLab, PEM private-key blocks, password-embedding connection
strings, the AWS **secret** access key in a labelled context — plus the digits-only /
space-grouped SSN (§3) — so the gated recall assertion FAILs the moment any of them
regresses. It is non-tautological: reverting a recogniser drops its type's recall.

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
    # -- H-C1 §1 modern secret/token formats + §3 SSN-no-dash ----------------
    # Each modern-secret family reports ``pii_type='secret'`` (its family is in the
    # ``detector`` provenance). The gold span is the WHOLE secret value. These are the
    # formats the audit found flowing UNREDACTED into the corpus at defaults.
    (
        "secret",
        "Leaked GitHub token ghp_1234567890abcdefghijklmnopqrstuvwxyz in a gist.",
        "ghp_1234567890abcdefghijklmnopqrstuvwxyz",
    ),
    (
        "secret",
        "Fine-grained PAT github_pat_11ABCDEFG0abcdefghijkl_mnopqrstuvwxyz1234567890 here.",
        "github_pat_11ABCDEFG0abcdefghijkl_mnopqrstuvwxyz1234567890",
    ),
    (
        "secret",
        "Stripe key sk_live_51H8xEXAMPLEabcdef1234567890 was committed.",
        "sk_live_51H8xEXAMPLEabcdef1234567890",
    ),
    (
        "secret",
        "OpenAI key sk-proj-abcDEF1234567890ghijklmnop in the notebook.",
        "sk-proj-abcDEF1234567890ghijklmnop",
    ),
    (
        "secret",
        "Google key AIzaSyD1234567890abcdefghijklmnopqrstuv in config.",
        "AIzaSyD1234567890abcdefghijklmnopqrstuv",
    ),
    (
        "secret",
        "Slack token xoxb-1234567890-ABCDEFGHIJKLMNOPqrstuvwx leaked.",
        "xoxb-1234567890-ABCDEFGHIJKLMNOPqrstuvwx",
    ),
    (
        "secret",
        "GitLab PAT glpat-ABCdef1234567890_xyz in the pipeline log.",
        "glpat-ABCdef1234567890_xyz",
    ),
    (
        "secret",
        "Pasted -----BEGIN RSA PRIVATE KEY----- into the ticket.",
        "-----BEGIN RSA PRIVATE KEY-----",
    ),
    (
        "secret",
        "DB url postgres://admin:s3cr3t!P4ss@db.example.com:5432/prod exposed.",
        "s3cr3t!P4ss",
    ),
    (
        "secret",
        "env AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY set.",
        "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    ),
    # §3: SSN with NO dashes and the space-grouped form — a real ``ssn`` label, not a
    # ``phone`` mislabel. A ``types:["ssn"]`` policy must genuinely cover these.
    ("ssn", "SSN digits-only on file: 123456789 (verify).", "123456789"),
    ("ssn", "Space-grouped SSN 123 45 6789 in the export.", "123 45 6789"),
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


def test_named_families_win_their_own_gold_span_provenance() -> None:
    """Each named §1 family owns its gold span's ``detector`` provenance — not a proxy.

    The per-type recall gate scores on ``pii_type`` alone, which is NOT enough on its
    own: several §1 families report ``pii_type='secret'`` and, on a *labelled-assignment*
    shape, the generic ``secret`` recogniser matches the SAME span with the SAME label.
    So a named family whose gold case is a labelled assignment (``aws_secret_key`` on
    ``AWS_SECRET_ACCESS_KEY=…``) is invisibly cross-covered: reverting its pattern leaves
    the span redacted by ``regex:secret`` and the recall benchmark stays green, defeating
    the issue's "each family a dedicated named recogniser so the per-type recall metric is
    meaningful / the benchmark FAILs if a recogniser is reverted" guarantee.

    This pins the winning span's ``detector`` to the named family, so reverting that
    family's pattern (its span disappears and the generic ``secret`` wins with
    ``regex:secret``) fails HERE — closing the cross-cover hole. Only families that are
    NOT uniquely covered by their own value-shape need pinning; the whole-value families
    (github_token, stripe_key, …) already bite via recall because nothing else matches
    their prefix-anchored value.
    """
    # (gold-set text, gold value, the named family that MUST own the span's provenance).
    # A labelled-assignment family whose value shape is also matched by the generic
    # ``secret`` capture at the identical span — the cross-cover the recall gate can't see.
    pinned: tuple[tuple[str, str, str], ...] = (
        (
            "env AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY set.",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "regex:aws_secret_key",
        ),
    )
    misses: list[str] = []
    for i, (text, value, want_detector) in enumerate(pinned):
        rec = _redact_all([_doc(text, f"provenance-{i}")])[0]
        start, end = text.index(value), text.index(value) + len(value)
        owning = [
            s for s in rec.pii_spans if s.char_start == start and s.char_end == end
        ]
        if not owning:
            misses.append(f"{want_detector}: gold span {value!r} not redacted at all")
        elif owning[0].detector != want_detector:
            misses.append(
                f"{want_detector}: span redacted by {owning[0].detector!r} instead — "
                "the named family was reverted and is cross-covered by a generic recogniser"
            )
    assert not misses, "Named-family provenance regressed (cross-cover hole):\n" + "\n".join(
        misses
    )


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


# -- H-C1 §2 compound-identifier api_key/token ---------------------------------


def test_compound_identifier_token_labels_are_redacted() -> None:
    """§2: a token keyword inside a larger UPPER_SNAKE identifier is caught.

    ``GITHUB_TOKEN=…``, ``SESSION_TOKEN=…``, ``X_AUTH_TOKEN=…``, ``MY_API_KEY=…`` all
    evaded the old ``api_key`` recogniser because its plain ``\\btoken\\b`` boundary
    cannot match inside ``GITHUB_TOKEN`` (``_`` is a word char). The optional-affix
    construction (identical to the ``secret`` fix) now catches them. Reverting to the
    plain boundary leaves the credential in the clear and fails here.
    """
    cases = (
        "GITHUB_TOKEN=ghp_ABCdef1234567890ABCdef1234567890abcd",
        "SESSION_TOKEN=abcdefghij0123456789ABCDEF",
        "X_AUTH_TOKEN: abcdefghij0123456789ABCDEF",
        "MY_API_KEY=abcdefghij0123456789ABCDEF",
    )
    for i, text in enumerate(cases):
        rec = _redact_all([_doc(text, f"compound-token-{i}")])[0]
        value = text.split("=" if "=" in text else ":", 1)[1].strip()
        assert value not in rec.masked_content, f"leaked token in {text!r}: {rec.masked_content!r}"
        assert any(
            s.pii_type in {"api_key", "secret"} for s in rec.pii_spans
        ), rec.pii_spans


def test_github_token_type_policy_covers_modern_secrets() -> None:
    """A ``types:["secret"]`` policy genuinely masks the modern secret families.

    The modern-secret families report ``pii_type='secret'`` (family in ``detector``),
    so an operator restricting to ``types:["secret"]`` still redacts a GitHub/Stripe/
    OpenAI/Google/Slack/GitLab token, a private-key header, and a conn-string password —
    the per-type metric is honest and the restriction is meaningful.
    """
    text = (
        "gh ghp_ABCdef1234567890ABCdef1234567890abcd "
        "stripe sk_live_ABCdef1234567890abcdef "
        "pem -----BEGIN OPENSSH PRIVATE KEY-----"
    )
    config = {"policies": {"public": {"types": ["secret"], "action": "mask"}}}
    rec = list(HybridPIIRedactor(config).redact([_doc(text, "secret-only")]))[0]
    assert "ghp_ABCdef1234567890ABCdef1234567890abcd" not in rec.masked_content
    assert "sk_live_ABCdef1234567890abcdef" not in rec.masked_content
    assert "-----BEGIN OPENSSH PRIVATE KEY-----" not in rec.masked_content
    assert all(s.pii_type == "secret" for s in rec.pii_spans), rec.pii_spans


def test_private_key_masks_the_whole_block_including_the_body() -> None:
    """A full PEM block is masked in its entirety — the key *body* is the secret.

    Masking only the ``-----BEGIN … -----`` header would leave the base64 key material
    (the actual secret) in the corpus. The recogniser spans header→body→footer, so no
    line of the key body survives into ``masked_content``. A truncated paste with no
    footer still masks the header line (the fallback), covered by the gold-set case.
    """
    body = "\n".join("MIIEowIBAAKCAQEA" + "b64Key0123456789" * 3 for _ in range(6))
    pem = f"-----BEGIN RSA PRIVATE KEY-----\n{body}\n-----END RSA PRIVATE KEY-----"
    text = f"Customer pasted their key:\n{pem}\nplease rotate it."
    rec = list(HybridPIIRedactor().redact([_doc(text, "pem-block")]))[0]
    # No fragment of the key body or the anchors survives.
    assert "-----BEGIN RSA PRIVATE KEY-----" not in rec.masked_content
    assert "-----END RSA PRIVATE KEY-----" not in rec.masked_content
    for line in body.splitlines():
        assert line not in rec.masked_content, f"key body line leaked: {line!r}"
    assert any(s.pii_type == "secret" for s in rec.pii_spans), rec.pii_spans


# -- H-C1 §3 SSN-no-dash label + false-positive guard --------------------------


def test_ssn_nodash_is_labelled_ssn_not_phone() -> None:
    """§3: a digits-only / space-grouped SSN is labelled ``ssn``, never ``phone``.

    The audit found ``123456789`` masked only by accidental ``phone`` overlap and
    mislabelled — so ``types:["ssn"]`` dropped it and the per-type metric was corrupted.
    The dedicated recogniser fixes the label; the specificity rank makes ``ssn`` win the
    tie over ``phone`` on the same span.
    """
    for i, (text, value) in enumerate(
        (("SSN 123456789 recorded.", "123456789"), ("SSN 123 45 6789 filed.", "123 45 6789"))
    ):
        rec = _redact_all([_doc(text, f"ssn-nodash-{i}")])[0]
        start, end = text.index(value), text.index(value) + len(value)
        matched = [s for s in rec.pii_spans if s.char_start == start and s.char_end == end]
        assert matched, f"SSN {value!r} not masked in {text!r}: {rec.pii_spans}"
        assert matched[0].pii_type == "ssn", f"mislabelled {matched[0].pii_type} (want ssn)"


def test_ssn_nodash_types_policy_covers_it() -> None:
    """§3: a ``types:["ssn"]`` policy genuinely masks the digits-only SSN.

    Before the dedicated recogniser this silently dropped (only ``phone`` matched it and
    ``phone`` is excluded). Now ``ssn`` covers both dashed and digits-only forms.
    """
    text = "Dashed 123-45-6789 and bare 123456789 both here."
    config = {"policies": {"public": {"types": ["ssn"], "action": "mask"}}}
    rec = list(HybridPIIRedactor(config).redact([_doc(text, "ssn-typed")]))[0]
    assert "123-45-6789" not in rec.masked_content
    assert "123456789" not in rec.masked_content
    assert all(s.pii_type == "ssn" for s in rec.pii_spans), rec.pii_spans


def test_ssn_nodash_plausible_guard_rejects_invalid_ranges() -> None:
    """§3 precision: the plausible-SSN guard rejects SSA-invalid area/group/serial.

    Area ``000``/``666``/``9xx``, group ``00``, serial ``0000`` are never issued, so a
    9-digit run in those ranges is NOT masked as an SSN — keeping recall gains from
    drowning the corpus in mask noise on ordinary 9-digit numbers.
    """
    invalid = ("000112222", "666445555", "900445555", "123005555", "123450000")
    for i, digits in enumerate(invalid):
        text = f"Reference number {digits} in the ledger."
        rec = _redact_all([_doc(text, f"ssn-invalid-{i}")])[0]
        assert not any(
            s.pii_type == "ssn" for s in rec.pii_spans
        ), f"{digits} wrongly masked as SSN: {rec.pii_spans}"


# -- H-C1 §1 per-family ReDoS probes ------------------------------------------


def test_modern_secret_families_are_linear_time() -> None:
    """§1 ReDoS probe: each new prefix-anchored family stays linear-time.

    Every §1 family is a literal prefix + a single bounded character class (no nested or
    overlapping quantifier). This hammers each with a long pathological run of its value
    class and asserts a wall-clock bound — a ReDoS-prone form would time out.
    """
    import re
    import time

    from latence_core.stages.redaction import _BUILTIN_PATTERNS

    probes = {
        "github_token": "ghp_" + "A" * 60000,
        "github_pat": "github_pat_" + "A" * 60000,
        "stripe_key": "sk_live_" + "A" * 60000,
        "openai_key": "sk-proj-" + "A" * 60000,
        "google_api_key": "AIza" + "A" * 60000,
        "slack_token": "xoxb-" + "A" * 60000,
        "gitlab_token": "glpat-" + "A" * 60000,
        "private_key": "-----BEGIN " + "A" * 60000,
        "aws_secret_key": "aws_secret_access_key=" + "A" * 60000,
        "conn_string": "postgres://u:" + "A" * 60000,
        "ssn_nodash": "1" * 60000 + " " * 60000,
    }
    for name, probe in probes.items():
        compiled = re.compile(_BUILTIN_PATTERNS[name], re.IGNORECASE)
        start = time.perf_counter()
        list(compiled.finditer(probe))
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"{name!r} took {elapsed:.2f}s on pathological input (possible ReDoS)"
