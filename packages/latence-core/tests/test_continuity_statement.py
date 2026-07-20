"""Regression guards for the continuity & security-process statement (issue #179).

``docs/CONTINUITY.md`` is a *buyer-facing* document: procurement reads it and takes
it at face value. The failure mode that matters is therefore not "the file is
missing" but "the file says something that is no longer true" — an aspirational
security posture that the repository does not actually implement.

These tests make the statement self-verifying, offline and dependency-free:

* the document exists and is reachable from both front doors (``README.md`` and the
  mkdocs navigation), so it cannot be published-but-unfindable;
* every repository path the document cites (in backticks or as a markdown link)
  actually exists — a claim that points at a missing artifact fails the gate;
* the load-bearing *posture* claims are checked against the artifacts themselves,
  in both directions: the signing claim must match what ``release.yml`` really
  does, the supported-versions claim must match ``SECURITY.md``, and the
  disabled-PyPI-publish claim must match the workflow.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import yaml

# tests/  ->  latence-core/  ->  packages/  ->  repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOC_RELPATH = "docs/CONTINUITY.md"
_DOC = _REPO_ROOT / _DOC_RELPATH

# Tokens that would indicate a real signing/attestation step in the release workflow.
_SIGNING_TOKENS = ("cosign", "sigstore", "attestations:", "gpg --sign", "minisign")

# Ways the document can *assert an unsigned posture*. Keyed on the claim, not on a single
# keyword: the whole failure mode here was §6/§7 asserting "no release signing" while the
# word "unsigned" appeared nowhere, so a `"unsigned" in text` guard stayed green on a lie.
_NEGATIVE_SIGNING_CLAIMS: tuple[tuple[str, str], ...] = (
    ("unsigned releases", r"\bunsigned\b"),
    ("no release signing", r"\bno\s+release\s+signing\b"),
    ("releases are not signed", r"\b(?:releases?|artifacts?)\b[^.]{0,40}?\bnot\s+signed\b"),
    ("no signing step in the workflow", r"absence of any[^.]{0,60}?(?:cosign|sigstore|attestation)"),
    (
        "a signed release chain listed as not provided",
        r"(?:does not give you|do(?:es)? not provide|you must cover yourself)[^.]{0,200}?"
        r"signed release chain",
    ),
)

# When the workflow really signs, the statement must name the concrete signing artifact
# rather than gesturing at "integrity" — this is what an evaluator is told to go verify.
_POSITIVE_SIGNING_ARTIFACTS = ("sigstore", "scripts/release-integrity.sh")


def _negative_signing_claims(text: str) -> list[str]:
    """Every 'releases are not signed' assertion the document makes, by label."""
    lowered = text.lower()
    return [
        label
        for label, pattern in _NEGATIVE_SIGNING_CLAIMS
        if re.search(pattern, lowered, flags=re.DOTALL)
    ]

# A backticked token is treated as a repo-path claim when it looks like one: it
# contains a "/" or a known-file suffix, and no shell/YAML noise.
_BACKTICK = re.compile(r"`([^`\n]+)`")
_MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_PATH_SUFFIXES = (".md", ".py", ".sh", ".yml", ".yaml", ".toml", ".lock", ".baseline")


def _doc_text() -> str:
    assert _DOC.is_file(), f"missing {_DOC_RELPATH} (the continuity statement, issue #179)"
    text = _DOC.read_text()
    assert text.strip(), f"{_DOC_RELPATH} is empty"
    return text


def _claimed_repo_paths(text: str) -> set[str]:
    """Repo-relative paths the document asserts exist (backticks + markdown links)."""
    found: set[str] = set()

    for raw in _BACKTICK.findall(text):
        cand = raw.strip()
        if " " in cand or cand.startswith(("$", "-", "#")):
            continue  # a command or a flag, not a path claim
        if "*" in cand:
            continue  # a glob (e.g. .github/workflows/*.yml) — covered by its dir
        if not (cand.endswith(_PATH_SUFFIXES) or cand.endswith("/")):
            continue
        found.add(cand.rstrip("/"))

    # Markdown links inside docs/ are relative to docs/; skip URLs and anchors.
    for raw in _MD_LINK.findall(text):
        if raw.startswith(("http://", "https://", "#", "mailto:")):
            continue
        target = raw.split("#", 1)[0]
        if not target:
            continue
        found.add(f"docs/{target}")

    assert found, "no repo-path claims parsed out of the continuity statement"
    return found


def test_continuity_statement_exists_and_covers_every_required_topic() -> None:
    """The statement answers all four procurement questions the issue names."""
    text = _doc_text()
    lowered = text.lower()
    required = {
        "maintainer model / bus factor": "bus factor",
        "license & fork posture": "apache",
        "docs-as-handover": "handover",
        "support lifetime": "support-lifetime statement",
        "CVD policy": "coordinated",
        "patch/release policy": "release policy",
        "SBOM posture": "sbom",
        "release-signing posture": "signing",
        "supply-chain controls": "supply-chain controls",
        "CRA readiness": "cyber resilience act",
    }
    missing = [topic for topic, token in required.items() if token not in lowered]
    assert not missing, f"{_DOC_RELPATH} does not cover: {missing}"


def test_continuity_statement_is_reachable_from_both_front_doors() -> None:
    """Linked from the README and present in the mkdocs navigation."""
    readme = (_REPO_ROOT / "README.md").read_text()
    assert "docs/CONTINUITY.md" in readme, "README.md does not link docs/CONTINUITY.md"

    mkdocs = (_REPO_ROOT / "mkdocs.yml").read_text()
    nav = mkdocs.split("nav:", 1)
    assert len(nav) == 2, "mkdocs.yml has no nav: section"
    assert "CONTINUITY.md" in nav[1], "mkdocs.yml nav does not include CONTINUITY.md"


def test_every_artifact_the_statement_cites_actually_exists() -> None:
    """Anti-false-green for prose: a cited artifact that is gone fails the gate."""
    missing = sorted(p for p in _claimed_repo_paths(_doc_text()) if not (_REPO_ROOT / p).exists())
    assert not missing, (
        f"{_DOC_RELPATH} cites artifacts that do not exist: {missing}\n"
        "Either restore the artifact or correct the statement — it must mirror reality."
    )


def test_negative_signing_claim_detector_is_not_keyword_hollow() -> None:
    """The detector must catch the *claim*, however it is phrased — not one keyword.

    Guards the guard: the original check was ``"unsigned" in text``, which a document
    asserting "No release signing" or listing "a signed release chain" under *what it does
    not give you* passes trivially. Each phrasing below is one that really appeared (or
    plausibly appears) in a buyer-facing statement and means "we do not sign releases".
    """
    for phrasing in (
        "Release artifacts are currently unsigned.",
        "| No release signing | absence of any cosign/sigstore/attestation step in `x.yml` |",
        "Releases are not signed today.",
        "**What it does not give you**, and you must cover yourself: a signed release chain\n(§5.4), automated dependency-CVE gating (§5.5).",
    ):
        assert _negative_signing_claims(phrasing), f"detector missed a denial of signing: {phrasing!r}"

    # ...and must not fire on the *affirmative* posture, or §5.4 itself would be unwritable.
    for phrasing in (
        "Release artifacts are signed with keyless Sigstore, producing SHA256SUMS.sigstore.json.",
        "a missing `id-token: write` permission fails the job loudly rather than quietly "
        "shipping an unverifiable release.",
        "| Signed release chain | `scripts/release-integrity.sh`, the `release-integrity` job |",
    ):
        assert not _negative_signing_claims(phrasing), f"detector false-positived on: {phrasing!r}"


def test_signing_claim_matches_the_release_workflow() -> None:
    """The signing posture is checked against release.yml, both ways.

    If someone adds cosign/sigstore/PEP-740 attestations to the release workflow, every
    'releases are unsigned' paragraph becomes a false *understatement* and must be updated —
    including the ones that never use the word "unsigned" (§6's "what it does not give you"
    list and §7's verification appendix). If someone claims releases are signed while the
    workflow has no signing step, that is the false-green this test exists to stop.
    """
    workflow = (_REPO_ROOT / ".github" / "workflows" / "release.yml").read_text().lower()
    workflow_signs = any(tok in workflow for tok in _SIGNING_TOKENS)
    doc = _doc_text()
    denials = _negative_signing_claims(doc)

    if workflow_signs:
        assert not denials, (
            f"release.yml has a signing/attestation step, but {_DOC_RELPATH} still asserts "
            f"an unsigned posture: {denials}. Every section must agree with §5.4 — a "
            "buyer-facing statement that contradicts itself on release integrity is a defect."
        )
        lowered = doc.lower()
        missing = [a for a in _POSITIVE_SIGNING_ARTIFACTS if a not in lowered]
        assert not missing, (
            f"{_DOC_RELPATH} must name the artifacts an adopter verifies against when the "
            f"release chain is signed; missing: {missing}"
        )
    else:
        assert denials, (
            "release.yml has no signing/attestation step, so docs/CONTINUITY.md must say "
            "release artifacts are unsigned rather than implying a signed release chain."
        )


def _pypi_autopublish_gate_defect(workflow_text: str) -> str | None:
    """Return why `pypi-publish` is not hard-gated off, or None when it is.

    Anchored to the *executable* condition, never to prose: `yaml.safe_load` drops `#`
    comments, so a workflow whose only `if: false` occurrences are in the ENABLE CHECKLIST
    cannot satisfy this check. The gate counts as closed only when the job's condition
    unconditionally short-circuits — a bare `false`, or a `false &&` prefix inside the
    `${{ … }}` expression. Anything else (including a condition that merely mentions
    `false` somewhere downstream of an `||`) is reported as a defect.
    """
    loaded = yaml.safe_load(workflow_text)
    jobs = (loaded or {}).get("jobs") or {}
    if "pypi-publish" not in jobs:
        return "release.yml has no `pypi-publish` job at all"
    job = jobs["pypi-publish"] or {}
    if "if" not in job:
        return "the `pypi-publish` job carries no `if:` condition — it runs unconditionally"

    condition = str(job["if"]).strip()
    # Unwrap `${{ … }}` so the check sees the expression itself, not the delimiters.
    expression = condition
    if expression.startswith("${{") and expression.endswith("}}"):
        expression = expression[3:-2].strip()

    if expression.lower() == "false":
        return None
    if re.match(r"^false\s*&&", expression, flags=re.IGNORECASE):
        return None
    return (
        f"the `pypi-publish` condition {condition!r} does not unconditionally "
        "short-circuit to false — a tag push can reach PyPI"
    )


def test_pypi_publish_disabled_claim_matches_the_workflow() -> None:
    """The statement says auto-publish is off; release.yml must actually gate it off."""
    workflow = (_REPO_ROOT / ".github" / "workflows" / "release.yml").read_text()
    defect = _pypi_autopublish_gate_defect(workflow)
    assert defect is None, (
        "docs/CONTINUITY.md states PyPI auto-publish is deliberately disabled, but "
        f"{defect}."
    )


def test_pypi_gate_check_rejects_the_documented_enable_mutation() -> None:
    """The guard must catch the exact edit release.yml's ENABLE CHECKLIST prescribes.

    This is the anti-false-green half. The old guard asserted the literal string
    `if: false`, which lives only inside `#` comment blocks — so removing the real
    `false &&` short-circuit (i.e. enabling auto-publish on every `v*` tag) left the
    comment prose, and the guard, untouched and green while docs/CONTINUITY.md's
    "deliberately disabled" claim became false.
    """
    workflow = (_REPO_ROOT / ".github" / "workflows" / "release.yml").read_text()
    assert _pypi_autopublish_gate_defect(workflow) is None, "precondition: gate is closed"

    enabled = workflow.replace("${{ false && github.event.inputs.dry_run", "${{ github.event.inputs.dry_run")
    assert enabled != workflow, (
        "the enable-mutation no longer applies — release.yml's `pypi-publish` condition "
        "changed shape, so this guard must be re-anchored to the new one."
    )
    # Prose asserting the job is disabled survives the mutation untouched — which is exactly
    # why a check anchored to comments (rather than to the condition) cannot be trusted.
    assert "SCAFFOLDED, DISABLED" in enabled, (
        "expected the disabled-claiming ENABLE CHECKLIST comments to survive the mutation"
    )

    defect = _pypi_autopublish_gate_defect(enabled)
    assert defect is not None, (
        "removing the `false &&` short-circuit enables auto-publish to PyPI on every tag "
        "push, but the guard still reports the gate as closed — docs/CONTINUITY.md would "
        "silently become false."
    )


def test_docs_do_not_cite_a_gate_literal_that_release_yml_lacks() -> None:
    """A buyer greps for the cited gate; it must be findable outside the comments.

    The statement used to name an `if: false` gate that exists nowhere in release.yml
    except its own ENABLE CHECKLIST prose.
    """
    workflow = (_REPO_ROOT / ".github" / "workflows" / "release.yml").read_text()
    executable = str(yaml.safe_load(workflow)["jobs"]["pypi-publish"]["if"])
    for relpath in ("docs/CONTINUITY.md", "docs/GO-LIVE-RUNBOOK.md", "docs/RELEASING.md"):
        text = (_REPO_ROOT / relpath).read_text()
        for cited in re.findall(r"`(if: [^`\n]+)`", text):
            assert cited in workflow or cited.rstrip(" …") in executable, (
                f"{relpath} cites the gate as {cited!r}, which does not appear in the "
                f"executable condition {executable!r} of the pypi-publish job."
            )


def test_pypi_gate_check_is_not_vacuous_when_the_job_disappears() -> None:
    """Deleting the job or its condition must fail loudly, not pass by absence."""
    workflow = (_REPO_ROOT / ".github" / "workflows" / "release.yml").read_text()
    loaded = yaml.safe_load(workflow)

    without_job = copy.deepcopy(loaded)
    del without_job["jobs"]["pypi-publish"]
    assert _pypi_autopublish_gate_defect(yaml.safe_dump(without_job)) is not None

    without_condition = copy.deepcopy(loaded)
    del without_condition["jobs"]["pypi-publish"]["if"]
    assert _pypi_autopublish_gate_defect(yaml.safe_dump(without_condition)) is not None


def test_pypi_gate_check_rejects_a_condition_that_merely_mentions_false() -> None:
    """`false` reachable behind an `||` is not a gate; the checker must say so."""
    template = yaml.safe_load(
        (_REPO_ROOT / ".github" / "workflows" / "release.yml").read_text()
    )
    for condition in (
        "${{ false || github.event.inputs.dry_run != 'true' }}",
        "${{ github.event.inputs.dry_run != 'true' && false }}",
        "${{ github.ref_type == 'tag' }}",
    ):
        mutated = copy.deepcopy(template)
        mutated["jobs"]["pypi-publish"]["if"] = condition
        assert _pypi_autopublish_gate_defect(yaml.safe_dump(mutated)) is not None, (
            f"condition {condition!r} lets a tag push reach PyPI but was accepted"
        )


def test_supported_versions_claim_matches_security_md() -> None:
    """The 'no backports, no maintained release branches' claim mirrors SECURITY.md."""
    security = (_REPO_ROOT / "SECURITY.md").read_text().lower()
    assert "no maintained release branches" in security, (
        "docs/CONTINUITY.md §4 claims there are no maintained release branches; "
        "SECURITY.md no longer says so."
    )
    doc = _doc_text().lower()
    assert "no backports" in doc and "no lts" in doc, (
        "the support-lifetime statement must state plainly that there are no backports "
        "and no LTS line"
    )


def test_statement_names_the_sole_maintainer_and_the_disclosure_contact() -> None:
    """Continuity is meaningless without naming who, and where to report."""
    text = _doc_text()
    assert "ddickmann81@gmail.com" in text, "no vulnerability-reporting contact in the statement"
    assert "@ddickmann" in text, "the statement must name the CODEOWNERS-of-record"
    codeowners = (_REPO_ROOT / ".github" / "CODEOWNERS").read_text()
    owners = {tok for tok in codeowners.split() if tok.startswith("@")}
    assert owners == {"@ddickmann"}, (
        f"CODEOWNERS now lists {sorted(owners)} — docs/CONTINUITY.md §1 claims a single "
        "owner on every path (bus factor 1). Update the statement."
    )


def test_cra_section_carries_the_dates_and_the_not_legal_advice_framing() -> None:
    """CRA readiness must be dated and scoped, not a vague compliance gesture."""
    text = _doc_text()
    assert "2024/2847" in text, "the CRA section must cite Regulation (EU) 2024/2847"
    assert "11 September 2026" in text, "the CRA section must state the Art. 14 reporting date"
    assert "not legal advice" in text.lower(), "the CRA section must disclaim legal advice"


# --- §5.5: "GitHub Actions pinned to commit SHAs" (issue R3-docs_truth-6) ---------------
#
# The §5.5 supply-chain table scopes this claim to `.github/workflows/*.yml` — every
# workflow, not the three that happened to be pinned. It was false for codeql.yml,
# scorecard.yml and docs.yml, i.e. for both of the security workflows the same table
# cites as the SAST and Scorecard controls. `_claimed_repo_paths` cannot catch it: it
# skips any backticked token containing `*`, so the glob in that row is never read.
#
# A `uses:` value is either a *local* composite action (`./.github/actions/...`, which has
# no SHA to pin), a Docker reference (`docker://...`), or an external `owner/repo[/path]@ref`
# — and every external ref must be a 40-hex commit SHA.

_USES = re.compile(r"^\s*(?:-\s*)?uses:\s*(?P<ref>\S+)", re.MULTILINE)
_SHA_PIN = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")


def _action_definition_files() -> list[Path]:
    """Every file in the repo that can carry a `uses:` step."""
    github = _REPO_ROOT / ".github"
    workflows = sorted((github / "workflows").glob("*.yml"))
    composites = sorted(github.glob("actions/*/action.yml"))
    return workflows + composites


def test_every_external_github_action_is_pinned_to_a_commit_sha() -> None:
    """§5.5's "a moved tag cannot change what runs" must hold for *every* workflow."""
    files = _action_definition_files()
    # Self-guarding: deleting the workflows must not make this test vacuously green.
    assert len(files) >= 4, f"expected the workflow set to be discoverable, found {files}"

    external = 0
    unpinned: list[str] = []
    for path in files:
        for match in _USES.finditer(path.read_text()):
            ref = match.group("ref").strip("\"'")
            if ref.startswith("./") or ref.startswith("docker://"):
                continue  # local composite action / Docker ref — no upstream tag to move
            external += 1
            if not _SHA_PIN.match(ref):
                line = path.read_text()[: match.start()].count("\n") + 1
                unpinned.append(f"{path.relative_to(_REPO_ROOT)}:{line} uses: {ref}")

    assert external >= 10, (
        f"only {external} external action references found — the guard has lost its subject"
    )
    assert not unpinned, (
        "docs/CONTINUITY.md §5.5 claims every action in `.github/workflows/*.yml` is pinned "
        "to a commit SHA so 'a moved tag cannot change what runs'. These are pinned to a "
        "MUTABLE TAG:\n  " + "\n  ".join(unpinned)
    )


def test_every_sha_pinned_action_records_the_version_it_pins() -> None:
    """A bare 40-hex SHA is unreviewable; each pin carries a `# vX.Y.Z` comment."""
    missing: list[str] = []
    for path in _action_definition_files():
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            match = _USES.match(line)
            if match is None or not _SHA_PIN.match(match.group("ref").strip("\"'")):
                continue
            if not re.search(r"#\s*v?\d", line):
                missing.append(f"{path.relative_to(_REPO_ROOT)}:{lineno} {line.strip()}")
    assert not missing, (
        "SHA-pinned actions must record the version they pin in a trailing comment "
        "(so a reviewer can tell what is being upgraded):\n  " + "\n  ".join(missing)
    )
