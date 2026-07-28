"""Regression guards for the OSS publish-readiness batch (issue #70).

Deterministic, offline, dependency-free repo-hygiene tests. Each one fails if the
publish-readiness fix is reverted:

* the Apache-2.0 ``LICENSE`` + ``NOTICE`` and the governance/security docs exist at
  the repo root with the expected, load-bearing content;
* every publishable workspace package's ``pyproject`` carries the PyPI metadata a
  release needs — Apache-2.0 license, the OSI Apache classifier, Python-version
  classifiers, ``keywords``, and ``project.urls`` — and none of it silently
  displaced the package's ``dependencies`` (the ``[project.urls]`` sub-table must
  come after every plain ``[project]`` key or TOML re-parents them);
* ``.gitignore`` denies the secret patterns a credential leak would ride in on;
* the audited ``.secrets.baseline`` exists and every entry is audited
  ``is_secret == false`` (a genuine secret would not be pre-audited, so the CI
  ``detect-secrets`` step would surface it).
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

# tests/  ->  latence-core/  ->  packages/  ->  repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PACKAGES = _REPO_ROOT / "packages"

# Every workspace package is publishable to PyPI and must carry release metadata.
# PEP 639: the license is a bare SPDX *expression* plus a license-files glob. The legacy
# ``License :: OSI Approved ::`` classifier is deprecated and, combined with an SPDX
# expression, trips strict twine/PyPI validators — so its absence is asserted, not its
# presence (issue #72, item 2).
_DEPRECATED_LICENSE_CLASSIFIER_PREFIX = "License ::"
_PY_VERSION_CLASSIFIERS = {
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
}


def _package_pyprojects() -> list[Path]:
    out = [
        pkg / "pyproject.toml"
        for pkg in sorted(_PACKAGES.iterdir())
        if (pkg / "pyproject.toml").is_file() and (pkg / "src").is_dir()
    ]
    assert out, "no publishable packages discovered under packages/"
    return out


def test_license_is_apache_2_0() -> None:
    """The repo-root LICENSE is the full Apache-2.0 text (ADR-0001)."""
    text = (_REPO_ROOT / "LICENSE").read_text()
    assert "Apache License" in text
    assert "Version 2.0, January 2004" in text
    assert "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" in text
    assert "END OF TERMS AND CONDITIONS" in text


def test_notice_asserts_copyright_and_gemma_optin() -> None:
    """NOTICE carries the project copyright and the honest EmbeddingGemma carve-out (ADR-0012)."""
    text = (_REPO_ROOT / "NOTICE").read_text()
    assert "Latence Framework" in text
    assert "Apache License" in text
    # EmbeddingGemma must be flagged NOT-default with flow-down, never silently listed as permissive.
    assert "embeddinggemma" in text.lower()
    assert "NOT a default" in text or "not a default" in text.lower()
    assert "flow-down" in text.lower()


def test_governance_and_third_party_docs_exist() -> None:
    """The OSS front-door governance set is present."""
    for name in (
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "THIRD-PARTY-LICENSES.md",
    ):
        p = _REPO_ROOT / name
        assert p.is_file() and p.read_text().strip(), f"missing or empty: {name}"


def test_security_md_documents_the_real_controls() -> None:
    """SECURITY.md names the actually-implemented controls, not aspirations."""
    text = (_REPO_ROOT / "SECURITY.md").read_text()
    for token in ("Screening", "allowlist", "SSRF", "scrub", "disclosure"):
        assert token.lower() in text.lower(), f"SECURITY.md missing: {token}"
    # A responsible-disclosure contact must be present.
    assert "ddickmann81@gmail.com" in text


def test_third_party_licenses_are_all_permissive() -> None:
    """THIRD-PARTY-LICENSES declares runtime deps permissive and never lists a copyleft runtime dep."""
    text = (_REPO_ROOT / "THIRD-PARTY-LICENSES.md").read_text()
    assert "permissive" in text.lower()
    # The core runtime deps and their verified licenses must appear.
    for dep in ("pydantic", "fsspec", "pyarrow", "pyyaml", "typer", "pypdf"):
        assert dep in text, f"THIRD-PARTY-LICENSES missing runtime dep: {dep}"


def test_gitignore_denies_secret_patterns() -> None:
    """.gitignore blocks the file patterns a credential leak would ride in on."""
    lines = {ln.strip() for ln in (_REPO_ROOT / ".gitignore").read_text().splitlines()}
    for pat in (".env", "*.pem", "*.key", ".pypirc"):
        assert pat in lines, f".gitignore missing secret pattern: {pat}"


def test_every_package_has_pypi_release_metadata() -> None:
    """Each publishable package advertises SPDX Apache-2.0 + classifiers + keywords + urls.

    The license assertion is the PEP 639 form the flip-public rehearsal validated: a bare
    SPDX expression (``license = "Apache-2.0"``, which hatchling emits as
    ``License-Expression``), a ``license-files`` glob that resolves to a real file inside
    the distribution root, and NO deprecated ``License ::`` classifier alongside it.
    """
    problems: list[str] = []
    for pp in _package_pyprojects():
        proj = tomllib.loads(pp.read_text())["project"]
        name = pp.parent.name

        lic = proj.get("license")
        if lic != "Apache-2.0":
            problems.append(
                f"{name}: license must be the bare SPDX expression 'Apache-2.0' "
                f"(PEP 639), got {lic!r}"
            )

        license_files = proj.get("license-files")
        if not license_files:
            problems.append(f"{name}: no license-files glob — the wheel would ship no LICENSE")
        else:
            for glob in license_files:
                if not list(pp.parent.glob(glob)):
                    problems.append(f"{name}: license-files glob {glob!r} matches nothing")

        classifiers = set(proj.get("classifiers", []))
        deprecated = sorted(
            c for c in classifiers if c.startswith(_DEPRECATED_LICENSE_CLASSIFIER_PREFIX)
        )
        if deprecated:
            problems.append(
                f"{name}: deprecated license classifier(s) {deprecated} alongside an SPDX "
                "License-Expression — strict validators reject the combination"
            )
        if not classifiers >= _PY_VERSION_CLASSIFIERS:
            problems.append(f"{name}: missing Python-version classifiers")

        if not proj.get("keywords"):
            problems.append(f"{name}: no keywords")
        if not proj.get("description"):
            problems.append(f"{name}: no description")
        if proj.get("readme") != "README.md":
            problems.append(f"{name}: readme != README.md")

        urls = proj.get("urls", {})
        for key in ("Homepage", "Repository", "Issues"):
            if key not in urls:
                problems.append(f"{name}: project.urls missing {key}")

    assert not problems, "PyPI metadata problems:\n" + "\n".join(problems)


def test_project_urls_did_not_swallow_dependencies() -> None:
    """A [project.urls] sub-table placed mid-[project] silently re-parents later plain keys.

    This is the exact TOML footgun the fix had to avoid: if [project.urls] is emitted
    before `dependencies`, `dependencies` is parsed as a child of `project.urls` and
    vanishes from `project`. Assert every package still exposes its declared
    dependencies (all depend on latence-core except the standalone root libraries).
    """
    # The two workspace ROOTS that intentionally declare no latence-core dependency:
    # latence-core itself, and latence-retrieval — the SEPARATE stateless downstream
    # query-time library (ADR-0048/0049), a thin pure-Python root (only pydantic) that an
    # adopter installs à la carte without pulling the pipeline. Both still must expose a
    # non-empty [project.dependencies] (the footgun guard) — they are exempt only from the
    # latence-core-edge proxy, not from the "dependencies survived parsing" check.
    _standalone_roots = {"latence-core", "latence-retrieval"}
    # Packages that are DELIBERATELY dependency-free, where `dependencies = []` is the
    # declared design and not a parsing casualty: latence-label-profiler is a build-time
    # corpus analysis tool whose whole claim (zero network, zero cost, zero model,
    # reproducible anywhere) a spurious dependency would make unverifiable. Non-emptiness
    # is only a PROXY for "the key survived parsing", so for these the real invariant is
    # asserted directly — the KEY must be present in [project]. If [project.urls] ever
    # re-parented it, `dependencies` would vanish from `proj` entirely and this still fails.
    _zero_dependency_packages = {"latence-label-profiler"}
    for pp in _package_pyprojects():
        proj = tomllib.loads(pp.read_text())["project"]
        name = pp.parent.name
        deps = proj.get("dependencies")
        if name in _zero_dependency_packages:
            assert "dependencies" in proj, f"{name}: dependencies lost from [project]"
            assert deps == [], (
                f"{name} is registered as deliberately dependency-free but declares {deps} — "
                "either drop the dependency or drop it from _zero_dependency_packages"
            )
            continue
        assert deps, f"{name}: dependencies lost from [project]"
        if name not in _standalone_roots:
            assert any(d.startswith("latence-core") for d in deps), (
                f"{name}: expected a latence-core dependency, got {deps}"
            )


def test_secrets_baseline_is_audited_clean() -> None:
    """.secrets.baseline exists and every finding is audited is_secret == false."""
    baseline = _REPO_ROOT / ".secrets.baseline"
    assert baseline.is_file(), "missing .secrets.baseline (CI secret-scan anchor)"
    data = json.loads(baseline.read_text())
    unaudited = [
        (f, item.get("line_number"))
        for f, items in data.get("results", {}).items()
        for item in items
        if item.get("is_secret") is not False
    ]
    assert not unaudited, f"un-audited or true-positive secret findings: {unaudited}"


def test_ci_secret_scan_uses_a_baseline_tolerant_gate() -> None:
    """The CI secret gate must not fail on a pristine tree with no new secret.

    ``detect-secrets scan --baseline`` (and ``detect-secrets-hook --baseline``) rewrite
    volatile fields — the ``generated_at`` timestamp and per-finding ``line_number``s —
    into the baseline on every run, so a ``git diff --exit-code .secrets.baseline`` gate
    is *guaranteed* red on a clean tree: noise that both breaks CI and makes a genuinely
    new secret indistinguishable from no change. The gate must instead be the set-based
    ``scripts/secret_scan_gate.py`` (fresh scan, compare (file, hashed_secret) pairs
    against the audited baseline, ignore line numbers + timestamp, never mutate). This
    locks that fix in.
    """
    ci = (_REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "scripts/secret_scan_gate.py" in ci, (
        "CI secret gate must run scripts/secret_scan_gate.py (the set-based gate that "
        "tolerates the baseline's volatile generated_at/line_number fields)"
    )
    # The fragile pattern must be gone: a raw git-diff on the always-rewritten baseline.
    assert "git diff --exit-code .secrets.baseline" not in ci, (
        "CI must not gate on `git diff --exit-code .secrets.baseline` — the baseline is "
        "rewritten with a fresh timestamp on every `scan --baseline`, so that check is "
        "always red on a clean tree"
    )


def test_secret_scan_gate_script_exists_and_is_wired() -> None:
    """The set-based gate script exists and CI actually invokes it."""
    gate = _REPO_ROOT / "scripts" / "secret_scan_gate.py"
    assert gate.is_file(), "missing scripts/secret_scan_gate.py"
    body = gate.read_text()
    # It must compare hashed_secret sets (immune to line-number/timestamp drift), and
    # only pass findings a human audited is_secret==false.
    assert "hashed_secret" in body
    assert "is_secret" in body


def test_secrets_baseline_line_numbers_are_in_range() -> None:
    """Every baseline finding's line_number is within its (still-present) tracked file.

    A grossly stale baseline points past a file's end — the symptom of the offsets that
    drifted when the secret-scan job was inserted above the audited ci.yml MinIO creds.
    (detect-secrets attributes secrets inside a YAML ``run: |`` block to the block's
    header line, so we only assert the line is in range, not that it is non-blank; the
    CI gate itself now uses ``detect-secrets-hook``, which tolerates line-number drift.)
    """
    data = json.loads((_REPO_ROOT / ".secrets.baseline").read_text())
    problems: list[str] = []
    for rel, items in data.get("results", {}).items():
        path = _REPO_ROOT / rel
        if not path.is_file():
            continue  # generated/ignored fixture; the audited-clean test covers content
        n_lines = len(path.read_text().splitlines())
        for item in items:
            ln = item.get("line_number")
            if not isinstance(ln, int) or not (1 <= ln <= n_lines):
                problems.append(f"{rel}: line_number {ln} out of range (1..{n_lines})")
    assert not problems, "stale .secrets.baseline offsets:\n" + "\n".join(problems)


def test_readme_status_is_not_stale_design_phase() -> None:
    """The front-door README reflects the built + G1-passed reality, not 'design phase'."""
    text = (_REPO_ROOT / "README.md").read_text()
    assert "design phase" not in text.lower(), "README still claims 'design phase' (stale)"
    assert "G1" in text, "README should reference the G1 gate status"
    # The honest 'not yet 1.0' caveat the issue asks for.
    assert "1.0" in text
