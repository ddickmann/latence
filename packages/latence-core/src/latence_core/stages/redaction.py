"""Built-in reference PIIDetector Provider (deterministic, hybrid rule, zero-dep).

The S6 Redaction reference Provider (ADR-0004; ships in core like the S3
Chunker/Screeners, the S4 gazetteer and the S5 trigger extractor so the demo runs
end to end, ADR-0007 CPU-first, ADR-0016 no extra deps). It is the **rule half** of
the "hybrid rule+model" reference the issue calls for.

W16 (ADR-0042): Redaction is a **chunk→chunk transform** — :meth:`~HybridPIIRedactor.redact`
consumes :class:`~latence_core.contracts.ChunkRecord`s and yields the SAME chunks with
``masked_content`` + ``pii_spans`` populated (each chunk ≤ the model window, so no PII is
truncated). The deprecated document-level path (one :class:`~latence_core.contracts.RedactionRecord`
per document, described below) survives as :meth:`~HybridPIIRedactor.redact_documents`.

**Coverage (accurate, not "comprehensive" — H-C1).** The rule library recognises,
enabled by default:

* well-known PII: ``email``, ``phone``, ``ssn`` (both ``NNN-NN-NNNN`` and the
  digits-only / space-grouped ``NNNNNNNNN`` / ``NNN NN NNNN`` form, with a
  plausible-SSN guard), ``credit_card``, ``ip``, ``iban``;
* the AWS access-key **id** (``aws_key``, the ``AKIA…`` public identifier);
* modern secret/token formats, each a high-precision prefix-anchored recogniser
  (reported under ``pii_type='secret'`` with a per-family ``detector`` provenance):
  GitHub (``ghp_``/``gho_``/``ghu_``/``ghs_``/``ghr_``, ``github_pat_``), Stripe
  (``sk_live_``/``sk_test_``/``rk_live_``/``pk_live_``), OpenAI (``sk-``/``sk-proj-``),
  Google API key (``AIza…``), Slack (``xox[baprs]-…``), GitLab (``glpat-…``), PEM
  private-key block headers, and a password-embedding connection string
  (``scheme://user:PASSWORD@host``, password only masked);
* labelled-assignment secrets: a generic ``api_key``/``token`` (also inside a compound
  ``UPPER_SNAKE`` identifier such as ``GITHUB_TOKEN``), a structural bare-``JWT``, the
  AWS **secret** access key in a labelled ``aws_secret_access_key=…`` context, and a
  ``password``/``secret`` assignment (also inside a compound identifier).

**Documented boundary (honest, not over-claimed).** The AWS **secret** access key is
matched only in a *labelled* context (``aws_secret_access_key=<40-char>``); the **bare**
40-char base64 form is deliberately NOT recognised because a bare high-entropy blob is
high-false-positive. This Provider recognises the shapes above; it is not an exhaustive
secret scanner, and the recall it guarantees is exactly what the gated recall benchmark
(``test_redaction_recall_benchmark``) proves — nothing beyond it is claimed. The
**model half** is the drop-in,
separately-packaged ``latence-pii-gliner`` (a permissively-licensed GLiNER-PII
checkpoint via plain ``transformers``, ADR-0012/0016) behind this same
:class:`~latence_core.capability.PIIDetector` seam — shipping the deterministic
rule reference in core keeps first contact (``git clone`` → working redaction) free
of torch and byte-stable for the Baseline bar.

Redaction is **metadata-aware** (S6 AC): the Provider keys its policy off each
document's ``Classification.sensitivity``. A ``policies`` map says, per sensitivity
value, which :class:`~latence_core.contracts.RedactionAction` to take (mask / replace
/ hash / tag) and, optionally, which PII types to act on — so a ``public`` document
can be masked hard while an ``internal`` one is only tagged for review, from the same
Pipeline with no code change.

Each emitted :class:`~latence_core.contracts.RedactionRecord` carries:

* ``masked_content`` — the document's assembled markdown with every non-``TAG`` PII
  span replaced by its deterministic placeholder;
* ``pii_spans`` — the structured, auditable spans (type, **document offset**, action,
  placeholder, resolved source page) — never the raw PII value, so the audit trail is
  not itself a PII leak (S6 AC: no raw PII written);
* ``policy`` — the applied policy name (the sensitivity value, or ``default``);
* Provenance covering ``[0, len)`` of the document, the parent's Classification
  inherited unchanged.

A document with no detected PII yields a RedactionRecord whose masked text equals the
source and whose span list is empty (never an error), so the corpus always has one
PII-handled variant per document.

Determinism: within a document, overlapping matches are resolved by (start, longer
span, type) and non-``TAG`` replacements are applied right-to-left so earlier offsets
never shift; documents are emitted in first-seen order. A ``replace``/``hash`` action
derives its surrogate from a config ``salt`` + the value, so a seeded run is
byte-identical without ever persisting the value.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from latence_core.contracts import Disposition, PIISpan, RedactionRecord
from latence_core.page_index import PageIndexResolver
from latence_core.pagemap import PageOffsetIndex
from latence_core.providers.profile import (
    REPO_LICENSE_CODE,
    REPO_LICENSE_SOURCE,
    ProviderProfile,
)
from latence_core.redaction_policy import (
    RedactionPolicy,
    apply_masking,
    build_span,
    finalize_redacted_chunk,
    parse_action,
    plan_chunk_redaction,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import ChunkRecord, DocumentRecord

# The built-in rule library: well-known PII/secret types with conservative,
# whole-value recognisers. Selectable by name via config ``detectors`` (default:
# all — comprehensive coverage). These are literal, offline, dependency-free — the
# deterministic rule half of the hybrid. Every pattern is linear-time (no nested
# quantifier over an overlapping class), so a pathological input cannot induce
# catastrophic backtracking (ReDoS).
_BUILTIN_PATTERNS: dict[str, str] = {
    # RFC-5322-ish email (deliberately conservative, not exhaustive). Linear-time:
    # a naive ``\b[local]+@[domain]+\.tld`` backtracks catastrophically on a long
    # dotted run because the local class ``[A-Za-z0-9._%+-]`` and the domain class
    # ``[A-Za-z0-9.-]`` both contain ``.`` and letters — and ``\b`` matches at every
    # word/non-word transition in ``x.x.x…``, so ``finditer`` relaunches a full
    # greedy scan from each of O(n) starts → O(n²). Fixed structurally: a negative
    # lookbehind pins the local part to its true start (no re-entry mid-run, since the
    # preceding char is never a local-part char), a possessive ``++`` local part never
    # backtracks, and the domain is a run of ``label.`` units whose label class
    # ``[A-Za-z0-9-]`` excludes ``.`` (no overlap). The single greedy outer ``+`` on
    # the domain backtracks only to peel a trailing sentence dot off the last label,
    # which is O(labels) once per (unique) ``@`` — linear overall.
    "email": (
        r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]++@(?:[A-Za-z0-9-]++\.)+[A-Za-z]{2,}\b"
    ),
    # US SSN NNN-NN-NNNN.
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    # US SSN with NO dashes (or single-space-grouped): a contiguous ``NNNNNNNNN`` or
    # ``NNN NN NNNN`` (§3, H-C1). Today such a value is masked only by accidental
    # ``phone`` overlap and MISLABELLED ``phone`` — so a ``types:["ssn"]`` policy or a
    # ``detectors`` list excluding ``phone`` silently drops it. A dedicated recogniser
    # with a **plausible-SSN guard** fixes both the coverage and the label: a leading
    # negative lookahead rejects the SSA-invalid area/group/serial ranges (area
    # ``000``/``666``/``9xx``, group ``00``, serial ``0000``) to cut the false positives
    # a bare 9-digit run would otherwise cause. A **letter-glue guard** rejects a run
    # fused to a letter on either side. It came from enterprise-gold T21 finding 2,
    # where the chunker welded ``| Industrial drives | 410 | 455 | 470 |`` into
    # ``Industrial drives410455470`` and this recogniser masked the innocent numeric row
    # as an SSN. **That weld is gone** — ``chunking._MARKUP_PATTERNS`` pattern 13 now
    # leaves a separator between cells, so the row arrives as
    # ``Industrial drives 410 455 470`` — and the guard was re-measured against the fix
    # rather than deleted along with its original motivation. It stays, on two measured
    # grounds: it costs no recall now that cells are separated (a table cell's SSN is
    # masked exactly as the same value in prose — pinned end-to-end by
    # ``test_table_row_pii_is_masked_identically_to_prose``), and it still buys precision
    # on letter-glued digit runs that have nothing to do with tables — part and lot codes
    # (``Lot 410455470A``), a label ending in a non-ASCII letter (``Menü410455470``).
    # ``[^\W\d_]`` is "any unicode letter" (word char minus digit minus ``_``), so
    # a German cell label ending in ``ä``/``ü`` is rejected too; the one letter-glued
    # form that IS a real SSN presentation — the literal label fused to the digits,
    # ``SSN123456789`` — is kept by the fixed-width ``(?<=[sS][sS][nN])`` exception
    # (case-classes, so it holds under ``case_sensitive: true`` as well). Linear-time:
    # every guard is a bounded fixed-width lookaround and the body is a fixed count of
    # digit classes with an optional single space between groups (no unbounded
    # quantifier over an overlapping class). Emitted with ``pii_type='ssn'`` (see
    # ``_DETECTOR_TYPE``) so the label is correct and ``types:["ssn"]`` genuinely
    # covers it.
    "ssn_nodash": (
        r"(?:(?<=[sS][sS][nN])|(?<![^\W\d_]))(?<!\d)"
        r"(?!000|666|9\d\d)\d{3} ?(?!00)\d{2} ?(?!0000)\d{4}"
        r"(?!\d)(?![^\W\d_])"
    ),
    # 13–16 digit card number, optional single space/dash between digits.
    "credit_card": r"\b\d(?:[ -]?\d){12,15}\b",
    # International-ish phone: optional +, 7–15 digits with separators.
    "phone": r"(?<!\w)\+?\d[\d\s().-]{6,}\d(?!\w)",
    # IPv4.
    "ip": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    # IBAN: 2 letters, 2 check digits, up to 30 alnum.
    "iban": r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b",
    # AWS access key id: literal AKIA/ASIA/AGPA/… prefix + 16 uppercase alnum.
    "aws_key": r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|A3T[A-Z0-9])[A-Z0-9]{16}\b",
    # Generic long high-entropy API token: a labelled key/token (including a bare
    # ``token:``/``jwt:`` and the session/refresh variants) followed by a 20+ char
    # base64/hex-ish credential (bounded, linear-time). The keyword may be a
    # **component of a larger UPPER_SNAKE identifier** (``GITHUB_TOKEN``,
    # ``X_AUTH_TOKEN``, ``MY_API_KEY``, ``SESSION_TOKEN``): the plain ``\btoken\b``
    # word boundary fails inside ``GITHUB_TOKEN`` because ``_`` is a word char, so —
    # exactly as the ``secret`` recogniser already does — match an optional single
    # leading identifier component and any number of trailing ones around the keyword
    # (§2, H-C1). Linear-time: the leading component ends in a separator and each
    # trailing component starts with one (both disjoint from the alnum run's edge), so
    # there is no catastrophic backtracking on a pathological ``x_x_x…token…`` run;
    # the value class ``[A-Za-z0-9/+_.=-]`` sits under a single bounded ``{20,}``.
    "api_key": (
        r"(?i:\b(?:[a-z0-9]+[_-])?"
        r"(?:api[_-]?key|access[_-]?token|auth[_-]?token|session[_-]?token"
        r"|refresh[_-]?token|bearer|token|jwt|secret[_-]?key)"
        r"(?:[_-][a-z0-9]+)*\b)"
        r"['\"]?\s*[:=]\s*['\"]?([A-Za-z0-9/+_.=-]{20,})"
    ),
    # A structural JSON Web Token: three base64url segments joined by dots, the header
    # segment starting ``eyJ`` (base64 of ``{"``). Catches a **bare** ``token: <JWT>``
    # even when the label is unrecognised, absent, or on another line — the value's own
    # shape is the recogniser. Linear-time: each segment is a single character class
    # under a fixed `+`, the segments separated by literal dots (no overlap).
    "jwt": r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\b",
    # -- §1 modern secret/token formats (H-C1) ------------------------------
    # Each family is a **high-precision, prefix-anchored** recogniser matching the
    # whole secret value as the span (group 0), not a greedy high-entropy blob — so
    # the per-type recall metric is meaningful and the false-positive rate is low.
    # Every pattern is a literal prefix followed by a single bounded character class
    # (no nested/overlapping quantifier), hence linear-time / ReDoS-safe. All emit
    # ``pii_type='secret'`` via ``_DETECTOR_TYPE`` so they roll up under one honest
    # per-type count while the ``detector`` field keeps the family provenance
    # (``regex:github_token`` …) for the audit.
    #
    # GitHub: ``ghp_``/``gho_``/``ghu_``/``ghs_``/``ghr_`` + 36 base62; and the
    # fine-grained ``github_pat_`` PAT (base62 with ``_``, bounded run).
    "github_token": r"\bgh[pousr]_[A-Za-z0-9]{36}\b",
    "github_pat": r"\bgithub_pat_[A-Za-z0-9_]{22,}\b",
    # Stripe secret/restricted/publishable live+test keys: ``sk_live_``/``sk_test_``/
    # ``rk_live_``/``pk_live_`` + a base62 run.
    "stripe_key": r"\b[rsp]k_(?:live|test)_[A-Za-z0-9]{16,}\b",
    # OpenAI: ``sk-`` and project ``sk-proj-`` keys (base62 with ``_``/``-``). Anchored
    # on the literal ``sk-`` prefix so it does not fire on an arbitrary hyphenated word;
    # the ``(?<![A-Za-z0-9])`` pin keeps it off a mid-identifier ``…sk-`` and there is a
    # 20+ length floor to avoid short false positives.
    "openai_key": r"(?<![A-Za-z0-9])sk-(?:proj-)?[A-Za-z0-9_-]{20,}\b",
    # Google API key: literal ``AIza`` + exactly 35 of ``[0-9A-Za-z_-]`` (the documented
    # Google key shape).
    "google_api_key": r"\bAIza[0-9A-Za-z_\-]{35}\b",
    # Slack token: ``xox`` + one of ``baprs`` + ``-`` + hyphen-joined digit/hex groups.
    # The trailing run is a single class (``[A-Za-z0-9-]``) under one bounded ``+``.
    "slack_token": r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
    # GitLab personal access token: ``glpat-`` + 20 base62-with-``_``/``-``.
    "gitlab_token": r"\bglpat-[A-Za-z0-9_-]{20}\b",
    # PEM private-key block — masks the WHOLE block (header + base64 body + footer), not
    # just the header: the key *body* is the secret, so masking only ``-----BEGIN…-----``
    # would leave the key material in the corpus. The body is a bounded lazy run pinned by
    # the ``-----END…-----`` footer; the bound (16 KiB, > an RSA-4096/OpenSSH PEM) keeps it
    # linear-time (no catastrophic backtracking). If the footer is absent (a truncated
    # paste), the optional group falls back to masking the header line alone. Fixed
    # alternation of literal words on both anchors — ReDoS-safe.
    "private_key": (
        r"-----BEGIN (?:RSA |EC |OPENSSH |PGP |DSA )?PRIVATE KEY-----"
        r"(?:[\s\S]{0,16384}?-----END (?:RSA |EC |OPENSSH |PGP |DSA )?PRIVATE KEY-----)?"
    ),
    # AWS **secret** access key — the bare 40-char base64 form is high-false-positive,
    # so this recogniser is deliberately **labelled-context only**: an
    # ``aws_secret_access_key``/``aws-secret-access-key`` assignment followed by the
    # 40-char ``[A-Za-z0-9/+]`` secret (captured as group 1, value only). The bare form
    # is intentionally NOT matched here — that boundary is documented in the docstring
    # rather than over-claimed (the labelled ``secret`` recogniser below also covers the
    # assignment shape; this named family makes the AWS-secret recall metric explicit).
    "aws_secret_key": (
        r"(?i:\baws[_-]?secret[_-]?access[_-]?key\b)"
        r"['\"]?\s*[:=]\s*['\"]?([A-Za-z0-9/+]{40})"
    ),
    # A connection string embedding a password: ``scheme://user:PASSWORD@host`` for the
    # common schemes (postgres/mysql/redis/amqp/mongodb/…). The captured span (group 1)
    # is the **password only** — between the ``:`` and the ``@`` — so the masked corpus
    # keeps the harmless scheme/user/host while the credential is gone. Linear-time: the
    # password class ``[^:@/\s]`` excludes the ``@`` delimiter, so the single ``+`` has
    # no overlapping alternative to backtrack into.
    "conn_string": (
        r"(?i:\b(?:postgres(?:ql)?|mysql|redis|amqp|mongodb(?:\+srv)?|rediss|ftp)"
        r"://[^:@/\s]+:)([^:@/\s]+)(?=@)"
    ),
    # A ``password``/``passwd``/``pwd``/``secret`` assignment — including when the
    # keyword is a **component of a compound identifier** (``aws_secret_access_key``,
    # ``client_secret``, ``db_password``): the plain ``\bsecret\b`` word boundary fails
    # inside ``…_secret_…`` because ``_`` is a word char, so match an optional single
    # leading identifier component and any number of trailing ones around the keyword.
    # Linear-time: the leading component ends in a separator and each trailing component
    # starts with one (both disjoint from the alnum run's edge), so there is no
    # catastrophic backtracking on a pathological run.
    "secret": (
        r"(?i:\b(?:[a-z0-9]+[_.-])?(?:password|passwd|pwd|secret)(?:[_.-][a-z0-9]+)*)"
        r"['\"]?\s*[:=]\s*['\"]?(\S{3,})"
    ),
}
# Recognisers whose credential value is captured in group(1) (a labelled
# assignment or a scheme://user:PASSWORD@host connection string): the span must
# cover only the secret, not the surrounding label/context. All other recognisers
# match the whole PII value as group(0).
_CAPTURE_GROUP_DETECTORS: frozenset[str] = frozenset(
    {"api_key", "secret", "aws_secret_key", "conn_string"}
)

# A recogniser NAME that emits a DIFFERENT ``pii_type`` label than its own name.
# Multiple named families roll up under one honest per-type count while each keeps a
# distinct ``detector`` provenance (``regex:github_token`` …) and its own linear-time
# ReDoS probe. The modern-secret families (§1) all report as ``secret``; the digits-
# only SSN recogniser (§3) reports as ``ssn`` so ``types:["ssn"]`` covers it and the
# per-type metric is honest (SSN never mislabelled ``phone``). Names not listed here
# use their own name as the ``pii_type`` (unchanged).
_DETECTOR_TYPE: dict[str, str] = {
    "ssn_nodash": "ssn",
    "github_token": "secret",
    "github_pat": "secret",
    "stripe_key": "secret",
    "openai_key": "secret",
    "google_api_key": "secret",
    "slack_token": "secret",
    "gitlab_token": "secret",
    "private_key": "secret",
    "aws_secret_key": "secret",
    "conn_string": "secret",
}

# Specificity rank used only to break ties when two recognisers claim the SAME
# span (same start, same length): the more specific, structured recogniser wins its
# label so a per-type recall/precision benchmark is not polluted by the loose phone
# recogniser mislabelling an SSN. Lower rank = higher priority. Config-supplied
# gazetteer/pattern recognisers and unlisted types fall back to the default rank, so
# they never lose their own exact spans to a built-in (they only ever tie on an
# identical span, and a stable, type-alphabetical order then decides — unchanged).
_SPECIFICITY_RANK: dict[str, int] = {
    "ssn": 0,
    "credit_card": 0,
    "iban": 0,
    "aws_key": 0,
    "jwt": 0,  # structural, whole-value token — wins a tie over a loose label capture
    "email": 1,
    "api_key": 1,
    "secret": 1,
    "ip": 1,
    "phone": 5,  # loosest built-in: loses a tie to any structured recogniser above
}
_DEFAULT_RANK = 3


class _Detector:
    """One compiled PII recogniser: a type label + a pattern + which engine fired.

    ``group`` is the capture group whose span is masked: 0 for a whole-value
    recogniser, 1 for a labelled-assignment recogniser (``password: <value>``)
    where only the credential — not the label — must be redacted.
    """

    __slots__ = ("pii_type", "pattern", "detector", "confidence", "group")

    def __init__(
        self,
        pii_type: str,
        pattern: re.Pattern[str],
        detector: str,
        confidence: float,
        group: int = 0,
    ) -> None:
        self.pii_type = pii_type
        self.pattern = pattern
        self.detector = detector
        self.confidence = confidence
        self.group = group


# A raw match before it becomes a PIISpan: type, document-offset span, the surface
# value (used only transiently to derive a surrogate — never persisted), the
# detector name, and the confidence.
_Match = tuple[str, int, int, str, str, float]


class HybridPIIRedactor:
    """Zero-dep hybrid-rule PIIDetector Provider, policy-keyed off Classification.

    The PII type set and how to recognise each type come from the built-in rule
    library plus ``config`` — nothing about *which* document gets *what* treatment is
    hardcoded: that is the sensitivity-keyed ``policies`` map.

    Config:
        detectors: list[str]  — which built-in recognisers to enable, by NAME (subset
            of ``email``/``ssn``/``ssn_nodash``/``credit_card``/``phone``/``ip``/
            ``iban``/``aws_key``/``api_key``/``jwt``/``secret`` plus the modern-secret
            families ``github_token``/``github_pat``/``stripe_key``/``openai_key``/
            ``google_api_key``/``slack_token``/``gitlab_token``/``private_key``/
            ``aws_secret_key``/``conn_string``). Default: **all** — the full coverage the
            recall benchmark proves, so a corpus is PII-handled by default (do not
            silently narrow this to a partial set in production configs; on a
            sensitive-classified document a narrowing that runs no detectors is surfaced
            LOUDLY via the §4 no-op floor, below).
        sensitive_set: list[str]  — the ``Classification.sensitivity`` values treated as
            SENSITIVE by the no-op floor (default ``["confidential", "restricted"]``).
            When a document in this set has the redaction control run NO detectors
            (``skip:true``, ``types: []``, or a ``detectors``/``types`` combination that
            resolves to nothing), the emitted RedactionRecord's
            ``redaction_disabled_for_sensitive`` flag is set and the Quality Report's
            ``redaction_disabled_for_sensitive`` count increments — so a disabled
            control is auditable and loud, never masquerading as "nothing to redact".
            The floor does NOT force redaction on (an operator may intentionally skip);
            it only makes the disposition first-class (H-C1 §4).
        gazetteer: dict[str, list[str]]  — extra literal terms to redact, keyed by PII
            type (e.g. ``{"name": ["Alice", "Bob"]}``) — whole-word, case-insensitive
            by default. The config-supplied half of the rule set.
        patterns: dict[str, list[str]]  — extra custom regexes keyed by PII type.
        default_action: str  — action when a policy names none: ``mask`` (default),
            ``replace``, ``hash``, or ``tag``.
        policies: dict[str, dict]  — the sensitivity-keyed policy map. Each key is a
            ``Classification.sensitivity`` value (``public``/``internal``/… or the
            literal ``default`` fallback); each value may set:
              * ``action``: str  — the action for this sensitivity (overrides
                ``default_action``).
              * ``types``: list[str]  — only act on these PII types (default: all
                detected). Types not listed are ignored (not even tagged).
              * ``skip``: bool  — if true, this sensitivity is passed through with an
                empty span list (redaction disabled for that class).
        salt: str  — salt mixed into ``replace``/``hash`` surrogates so equal values
            map to equal (but non-reversible) placeholders deterministically.
        case_sensitive: bool  — match gazetteer terms case-sensitively (default False).
        min_chars: int  — a document whose assembled text is shorter than this is
            emitted as a clean (no-PII) RedactionRecord (default 1).
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic,
        Apache-2.0.

        The hybrid-rule PII Redactor is a linear-time regex/gazetteer detector.
        A pure-Python part of ``latence-core`` (Apache-2.0, verified per ADR-0012
        — the package's own code, weights-and-code trivially the same), it ships no model weights
        (``model_id=None``), is deterministic, CPU-only, does not batch, and has no per-call cost,
        so a bake-off/conformance run reads honest, non-fabricated columns off it.
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=32,
            model_id=None,
            license=REPO_LICENSE_CODE,
            license_verified=True,
            license_code=REPO_LICENSE_CODE,
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-06",
            deterministic=True,
            batch=False,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._case_sensitive = bool(config.get("case_sensitive", False))
        self._min_chars = int(config.get("min_chars", 1))
        self._salt = str(config.get("salt", "latence"))
        default_action = parse_action(config.get("default_action", "mask"))
        self._policy = RedactionPolicy(config.get("policies"), default_action)
        self._detectors = self._compile_detectors(config)
        # The set of pii_types some enabled detector can produce — the coverage the
        # applied policy narrows. Used by the §4 no-op floor to tell "control ran no
        # detectors" (empty intersection with the policy allow-list) apart from
        # "genuinely no PII found".
        self._detector_types = {d.pii_type for d in self._detectors}
        # The Classification.sensitivity values considered SENSITIVE for the §4 no-op
        # floor. Configurable; default confidential/restricted. A disabled control on
        # one of these surfaces as a loud, first-class signal rather than a silent no-op.
        sensitive = config.get("sensitive_set", ("confidential", "restricted"))
        if not isinstance(sensitive, (list, tuple, set)):
            msg = "'sensitive_set' config must be a list of Classification.sensitivity values."
            raise TypeError(msg)
        self._sensitive_set = {str(s) for s in sensitive}

    # -- config compilation -------------------------------------------------

    def _compile_detectors(self, config: dict[str, Any]) -> list[_Detector]:
        """Compile the enabled built-in rules + config gazetteer/patterns into recognisers."""
        flags = 0 if self._case_sensitive else re.IGNORECASE
        enabled = config.get("detectors")
        if enabled is not None and not isinstance(enabled, list):
            msg = "'detectors' config must be a list of built-in recogniser names."
            raise TypeError(msg)
        names = list(enabled) if enabled is not None else list(_BUILTIN_PATTERNS)
        detectors: list[_Detector] = []
        for name in names:
            builtin = _BUILTIN_PATTERNS.get(str(name))
            if builtin is None:
                available = ", ".join(sorted(_BUILTIN_PATTERNS))
                msg = f"Unknown built-in PII detector {name!r}. Available: {available}."
                raise ValueError(msg)
            # Built-in patterns are ASCII-cased already; IGNORECASE is harmless.
            # ``name`` is the recogniser (→ ``detector`` provenance ``regex:<name>``);
            # ``pii_type`` is the label it reports (``_DETECTOR_TYPE`` remaps a family
            # like ``github_token``→``secret`` / ``ssn_nodash``→``ssn`` so the per-type
            # count is honest, else it is the name itself).
            name_s = str(name)
            group = 1 if name_s in _CAPTURE_GROUP_DETECTORS else 0
            pii_type = _DETECTOR_TYPE.get(name_s, name_s)
            detectors.append(
                _Detector(pii_type, re.compile(builtin, flags), f"regex:{name_s}", 0.9, group)
            )
        gazetteer = config.get("gazetteer", {})
        if not isinstance(gazetteer, dict):
            msg = "'gazetteer' config must be a mapping of pii_type -> list[str]."
            raise TypeError(msg)
        for pii_type, terms in gazetteer.items():
            alternatives = [rf"\b{re.escape(str(term))}\b" for term in terms]
            if not alternatives:
                continue
            pattern = re.compile("|".join(f"(?:{alt})" for alt in alternatives), flags)
            detectors.append(
                _Detector(str(pii_type), pattern, f"gazetteer:{pii_type}", 0.99)
            )
        patterns = config.get("patterns", {})
        if not isinstance(patterns, dict):
            msg = "'patterns' config must be a mapping of pii_type -> list[str]."
            raise TypeError(msg)
        for pii_type, raw_patterns in patterns.items():
            alternatives = [str(pat) for pat in raw_patterns]
            if not alternatives:
                continue
            pattern = re.compile("|".join(f"(?:{alt})" for alt in alternatives), flags)
            detectors.append(
                _Detector(str(pii_type), pattern, f"pattern:{pii_type}", 0.9)
            )
        return detectors

    # -- detection ----------------------------------------------------------

    def redact(self, chunks: Iterable[ChunkRecord]) -> Iterator[ChunkRecord]:
        """Detect PII per chunk; yield the SAME chunks with ``masked_content`` + ``pii_spans`` set.

        W16 (ADR-0042): the chunk-level Redactor. Each chunk's ``content`` (≤ the model window) is
        scanned in full — no truncation — so PII anywhere in the chunk is masked. The clean
        ``content`` is UNCHANGED; ``masked_content`` is the PII-safe variant and ``pii_spans`` the
        chunk-local, counts-only spans (source pages resolved through the chunk's offset_map + the
        document-scoped page index). The shared :func:`plan_chunk_redaction` /
        :func:`finalize_redacted_chunk` machinery owns policy + no-op-floor + masking; this
        Provider owns only detection. Chunks are threaded 1:1 in first-seen order for determinism.
        """
        # ONE resolver for the whole call — the document's page map rides its first chunk
        # (v18), so a per-chunk resolver would only resolve pages for that chunk.
        resolver = PageIndexResolver()
        for chunk in chunks:
            plan = plan_chunk_redaction(
                chunk,
                self._policy,
                detector_types=self._detector_types,
                sensitive_set=self._sensitive_set,
                min_chars=self._min_chars,
                resolver=resolver,
            )
            accepted: list[tuple[int, int, str, str, str, float]] = []
            if plan.needs_detection:
                accepted = [
                    (start, end, pii_type, value, detector, confidence)
                    for pii_type, start, end, value, detector, confidence in self._find_matches(
                        chunk.content, plan.allowed
                    )
                ]
            yield finalize_redacted_chunk(chunk, plan, accepted, self._salt)

    def redact_documents(self, documents: Iterable[DocumentRecord]) -> Iterator[RedactionRecord]:
        """DEPRECATED (W16, ADR-0042): the document-level path. Prefer chunk-level :meth:`redact`.

        Scans the whole document's assembled markdown (``record.content``) and emits one
        :class:`RedactionRecord` per document. Retained for the superseded doc-level path; the
        blessed Capability truncates nothing because it runs per chunk (:meth:`redact`). A
        ``PARSE_ERROR`` document (no usable content) is skipped, mirroring Chunk.
        """
        for document in documents:
            if document.disposition is Disposition.PARSE_ERROR:
                continue
            yield self._redact_one_document(document)

    def _redact_one_document(self, document: DocumentRecord) -> RedactionRecord:
        doc_id = document.record_id
        text = document.content
        page_map = document.page_map
        classification = document.classification
        base_prov = document.provenance
        action, allowed, skip, policy_name = self._policy.resolve(classification.sensitivity)

        # §4 no-op floor: would the applied policy run ANY detector on this document?
        # No, if the policy skips, or restricts to a type set that intersects none of
        # the enabled detectors' types (including an empty ``types: []``), or the
        # Provider has no detectors enabled at all. When the document is also
        # classified sensitive, that disposition is a LOUD, first-class signal — an
        # empty ``pii_spans`` there is "control OFF", not "clean".
        would_run_detectors = (not skip) and bool(
            self._detector_types
            if allowed is None
            else self._detector_types & allowed
        )
        disabled_for_sensitive = (
            not would_run_detectors
            and classification.sensitivity in self._sensitive_set
        )

        spans: list[PIISpan] = []
        masked = text
        if not skip and len(text.strip()) >= self._min_chars:
            matches = self._find_matches(text, allowed)
            page_index = PageOffsetIndex(page_map) if page_map is not None else None
            spans = [
                build_span(
                    pii_type=pii_type,
                    detector=detector,
                    start=start,
                    end=end,
                    value=value,
                    action=action,
                    salt=self._salt,
                    page_index=page_index,
                    confidence=confidence,
                )
                for pii_type, start, end, value, detector, confidence in matches
            ]
            masked = apply_masking(text, spans)

        # A RedactionRecord's Provenance covers the whole document [0, len). Source
        # identity is inherited from the DocumentRecord's Provenance; the offset span is
        # set to the document extent.
        provenance = base_prov.model_copy(
            update={
                "char_start": 0,
                "char_end": len(text),
                "page_start": None,
                "page_end": None,
            }
        )
        return RedactionRecord(
            record_id=f"{doc_id}#redaction",
            document_record_id=doc_id,
            masked_content=masked,
            media_type="text/markdown",
            policy=policy_name,
            pii_spans=spans,
            redaction_disabled_for_sensitive=disabled_for_sensitive,
            provenance=provenance,
            classification=classification,
        )

    def _find_matches(self, text: str, allowed: set[str] | None) -> list[_Match]:
        """All non-overlapping PII matches across enabled detectors, deterministically.

        Each detector scans independently; a type not in ``allowed`` (when a policy
        restricts types) is skipped entirely. Overlaps between detectors are resolved
        by (earlier start, longer span, more specific type, type name) so a stable
        non-overlapping set results — the specificity rank ensures a structured
        recogniser (SSN, credit card, …) claims a span the loose phone recogniser also
        matched, so the per-type label is correct.
        """
        raw: list[_Match] = []
        for detector in self._detectors:
            if allowed is not None and detector.pii_type not in allowed:
                continue
            for m in detector.pattern.finditer(text):
                # For a labelled-assignment recogniser, redact only the captured
                # credential (group 1) — never the ``password:`` label; otherwise
                # the whole matched value (group 0).
                group = detector.group
                start, end = m.start(group), m.end(group)
                if start < 0 or end <= start:
                    continue
                raw.append(
                    (detector.pii_type, start, end, m.group(group), detector.detector,
                     detector.confidence)
                )
        raw.sort(
            key=lambda t: (
                t[1],
                -(t[2] - t[1]),
                _SPECIFICITY_RANK.get(t[0], _DEFAULT_RANK),
                t[0],
            )
        )
        accepted: list[_Match] = []
        last_end = -1
        for match in raw:
            _, start, end, _, _, _ = match
            if start >= last_end:
                accepted.append(match)
                last_end = end
        return accepted
