"""Regression guards for the OSS publish-readiness batch (issue #70) and the licence.

Deterministic, offline, dependency-free repo-hygiene tests. Each one fails if the
publish-readiness fix is reverted:

* the ``LICENSE`` really is the PolyForm Noncommercial License 1.0.0, reproduced
  VERBATIM (checked by digest against the licence steward's official plain text,
  not by keyword), it carries the never-published provenance record, and it is
  mirrored byte-identically into every package that ships a wheel;
* ``NOTICE``, ``COMMERCIAL.md`` and the governance/security docs exist at the repo
  root with the expected, load-bearing content — including the fact that
  commercial terms are BY AGREEMENT and are not invented here;
* every publishable workspace package's ``pyproject`` carries the PyPI metadata a
  release needs — the ``PolyForm-Noncommercial-1.0.0`` SPDX expression, no
  deprecated ``License ::`` classifier, Python-version classifiers, ``keywords``,
  and ``project.urls`` — and none of it silently displaced the package's
  ``dependencies`` (the ``[project.urls]`` sub-table must come after every plain
  ``[project]`` key or TOML re-parents them);
* ``.gitignore`` denies the secret patterns a credential leak would ride in on;
* the audited ``.secrets.baseline`` exists and every entry is audited
  ``is_secret == false`` (a genuine secret would not be pre-audited, so the CI
  ``detect-secrets`` step would surface it).

LICENCE HISTORY. The project was Apache-2.0 internally until 2026-07-29 and was
never published under it (private repository, nothing on PyPI, no release cut,
sole copyright holder). These gates are written so that a drift BACK to
Apache-2.0 — in the LICENSE text, in the NOTICE, or in any single package's
metadata — fails the build rather than shipping a distribution whose declared
licence and shipped licence text disagree.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path

import pytest

# tests/  ->  latence-core/  ->  packages/  ->  repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PACKAGES = _REPO_ROOT / "packages"

# The project's outbound licence, as the SPDX License List spells it.
_SPDX_LICENSE = "PolyForm-Noncommercial-1.0.0"
# The licence this project used to carry. Asserting its ABSENCE is what makes these
# gates catch a partial revert (e.g. one pyproject quietly flipped back).
_SUPERSEDED_SPDX_LICENSE = "Apache-2.0"

# The verbatim licence body lives between these markers in LICENSE. Everything outside
# them is this project's own framing (provenance record, Required Notice, pointers) and
# must not be able to alter the terms.
_LICENSE_BEGIN = "----- BEGIN POLYFORM NONCOMMERCIAL LICENSE 1.0.0 -----"
_LICENSE_END = "----- END POLYFORM NONCOMMERCIAL LICENSE 1.0.0 -----"
# SHA-256 of https://polyformproject.org/licenses/noncommercial/1.0.0.txt as retrieved
# on 2026-07-29 — the licence steward's OFFICIAL plain text. Pinning the digest is the
# only way to assert "verbatim": a keyword check would pass on a licence someone had
# quietly edited a clause out of, which is precisely the failure that matters here.
_POLYFORM_TEXT_SHA256 = "ffcca38841adb694b6f380647e15f17c446a4d1656fed51a1e2041d064c94cc8"

# Every workspace package is publishable to PyPI and must carry release metadata.
# PEP 639: the license is a bare SPDX *expression* plus a license-files glob. The legacy
# ``License :: OSI Approved ::`` classifier is deprecated and, combined with an SPDX
# expression, trips strict twine/PyPI validators — so its absence is asserted, not its
# presence (issue #72, item 2). It is doubly wrong now: PolyForm Noncommercial is not
# OSI-approved, so there is no correct ``License :: OSI Approved ::`` classifier to use.
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


def _license_body(text: str) -> str:
    """The verbatim licence text between the BEGIN/END markers, or fail saying why."""
    match = re.search(
        re.escape(_LICENSE_BEGIN) + r"\n(.*)" + re.escape(_LICENSE_END), text, re.DOTALL
    )
    assert match, (
        "LICENSE no longer delimits the licence body with the BEGIN/END markers. Without "
        "them there is no way to tell the licence terms apart from this project's own "
        "framing around them, and no way to prove the terms are unmodified."
    )
    return match.group(1)


def test_license_is_polyform_noncommercial_verbatim() -> None:
    """LICENSE embeds the PolyForm Noncommercial 1.0.0 text UNMODIFIED, byte for byte.

    Checked by digest against the licence steward's official plain text, because the
    whole value of using a professionally drafted licence is lost the moment a clause is
    silently edited — and a keyword assertion ("PolyForm" in text) would happily pass on
    a licence with its Violations or No Liability section removed.
    """
    text = (_REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    body = _license_body(text)

    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert digest == _POLYFORM_TEXT_SHA256, (
        "the licence text between the markers is NOT the official PolyForm Noncommercial "
        f"License 1.0.0 plain text.\n  expected sha256: {_POLYFORM_TEXT_SHA256}\n"
        f"  actual sha256:   {digest}\n"
        "Restore it verbatim from https://polyformproject.org/licenses/noncommercial/1.0.0.txt "
        "— a hand-edited licence is an unenforceable licence."
    )

    # Sanity anchors that the digest alone would not make legible in a failure report.
    assert "# PolyForm Noncommercial License 1.0.0" in body
    assert "Any noncommercial purpose is a permitted purpose." in body
    # ...and the superseded licence must not be lurking in the terms.
    assert "Apache License" not in body


def test_license_carries_the_relicence_provenance_record() -> None:
    """The provenance note is what a future reader, acquirer or lawyer looks for FIRST.

    The relicence is only clean because the Apache-2.0 grant never reached anyone. That
    claim has to be IN the licence file, stated as checkable facts, or it is folklore
    that dies with the person who remembers it.
    """
    text = (_REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    head, _, _ = text.partition(_LICENSE_BEGIN)

    assert "PROVENANCE" in head, "LICENSE has no provenance section above the licence terms"
    assert "NOT PART OF THE LICENCE TERMS" in head, (
        "the provenance note must say, in the file, that it is not part of the terms — "
        "otherwise it reads as an amendment to a licence that must stay verbatim"
    )
    lowered = head.lower()
    for fact in ("apache-2.0", "private", "pypi", "sole copyright holder"):
        assert fact in lowered, f"the provenance record does not state: {fact}"
    # The Notices section of the licence is only operative if the notice actually exists.
    assert "Required Notice: Copyright 2026 Dennis Dickmann" in head, (
        "LICENSE carries no `Required Notice:` line, so the licence's Notices section "
        "points at nothing and no downstream copy can carry the copyright attribution"
    )
    assert f"SPDX-License-Identifier: {_SPDX_LICENSE}" in head
    # Commercial use must be routed somewhere, not left as an open question.
    assert "COMMERCIAL.md" in head


def test_every_shipped_package_license_matches_the_root_license() -> None:
    """The wheels ship a LICENSE file — it must be the licence the metadata declares.

    ``license-files = ["LICENSE"]`` puts each package's own copy inside the wheel. A
    relicence that updates the root file and 33 ``pyproject.toml`` entries but forgets
    the 33 copies produces distributions that DECLARE PolyForm and SHIP Apache-2.0 —
    the most dangerous shape of this failure, because the shipped text is the one a
    court reads.
    """
    root = (_REPO_ROOT / "LICENSE").read_bytes()
    drifted = [
        pkg.name
        for pkg in sorted(_PACKAGES.iterdir())
        if (pkg / "pyproject.toml").is_file() and (pkg / "src").is_dir()
        if not (pkg / "LICENSE").is_file() or (pkg / "LICENSE").read_bytes() != root
    ]
    assert not drifted, (
        "these packages ship a LICENSE that is missing or differs from the repo-root "
        f"LICENSE: {drifted}"
    )


def test_notice_asserts_copyright_licence_and_gemma_optin() -> None:
    """NOTICE carries the copyright, the CURRENT licence, and the EmbeddingGemma carve-out."""
    text = (_REPO_ROOT / "NOTICE").read_text()
    assert "Latence Framework" in text
    assert _SPDX_LICENSE in text, "NOTICE does not name the project's actual licence"
    assert "PolyForm Noncommercial License 1.0.0" in text
    # A NOTICE that names the licence but not the commercial route sends a commercial
    # user away believing they may use it.
    assert "COMMERCIAL.md" in text
    assert "Required Notice:" in text
    assert "Apache License, Version 2.0" not in text, (
        "NOTICE still grants the superseded Apache-2.0 licence"
    )
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
        "COMMERCIAL.md",
    ):
        p = _REPO_ROOT / name
        assert p.is_file() and p.read_text().strip(), f"missing or empty: {name}"


def test_commercial_md_routes_commercial_use_without_inventing_terms() -> None:
    """COMMERCIAL.md must name a contact and a boundary — and must NOT invent terms.

    A noncommercial licence is only workable if the commercial route is discoverable.
    But this repository does not get to invent pricing, tiers, or contract language on
    the owner's behalf: those are the owner's to set, in writing, per agreement.
    """
    text = (_REPO_ROOT / "COMMERCIAL.md").read_text()

    assert _SPDX_LICENSE in text
    assert "ddickmann81@gmail.com" in text, "no contact for obtaining a commercial licence"
    assert "by agreement" in text.lower(), (
        "COMMERCIAL.md must state that commercial terms are settled by agreement"
    )
    # The provenance record has to be reachable from here too — this is the document a
    # prospective commercial licensee reads first.
    lowered = text.lower()
    assert "apache-2.0" in lowered and "never published" in lowered

    # No invented commercial terms. Anything resembling a price list is a claim the
    # copyright holder never made.
    assert not re.search(r"[$€£]\s?\d", text), "COMMERCIAL.md quotes a price"
    for invented in ("per seat", "per-seat", "annual fee", "subscription fee", "free trial"):
        assert invented not in lowered, f"COMMERCIAL.md invents a commercial term: {invented}"

    # It must not claim to be legal advice, and must not overreach into third-party terms.
    assert "not legal advice" in lowered
    assert "easydict" in lowered, (
        "COMMERCIAL.md must tell a commercial licensee that third-party terms — the "
        "LGPL-3.0 easydict extra above all — are not covered by an agreement about THIS code"
    )


def test_no_owned_licence_document_still_grants_apache_2_0() -> None:
    """A partial revert is the realistic failure: one file left behind on the old licence.

    Scoped to the files this relicence owns, so it is a real gate and not a proxy: the
    root licence documents, and every package's shipped LICENSE + declared metadata.
    """
    offenders: list[str] = []
    for path in (
        _REPO_ROOT / "LICENSE",
        _REPO_ROOT / "NOTICE",
        _REPO_ROOT / "COMMERCIAL.md",
        *sorted(_PACKAGES.glob("*/LICENSE")),
    ):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            # "was Apache-2.0", "superseded", the provenance record: those are history,
            # not a grant. A GRANT is a line that licenses the work under it.
            if re.search(r"[Ll]icensed under the Apache", line):
                offenders.append(f"{path.relative_to(_REPO_ROOT)}: {line.strip()}")
    assert not offenders, (
        "these files still grant the superseded Apache-2.0 licence:\n" + "\n".join(offenders)
    )


def test_security_md_documents_the_real_controls() -> None:
    """SECURITY.md names the actually-implemented controls, not aspirations."""
    text = (_REPO_ROOT / "SECURITY.md").read_text()
    for token in ("Screening", "allowlist", "SSRF", "scrub", "disclosure"):
        assert token.lower() in text.lower(), f"SECURITY.md missing: {token}"
    # A responsible-disclosure contact must be present.
    assert "ddickmann81@gmail.com" in text


def test_third_party_licenses_are_all_permissive() -> None:
    """THIRD-PARTY-LICENSES declares default-path runtime deps permissive, and lists them."""
    text = (_REPO_ROOT / "THIRD-PARTY-LICENSES.md").read_text()
    assert "permissive" in text.lower()
    # The core runtime deps and their verified licenses must appear.
    for dep in ("pydantic", "fsspec", "pyarrow", "pyyaml", "typer", "pypdf"):
        assert dep in text, f"THIRD-PARTY-LICENSES missing runtime dep: {dep}"


def test_third_party_licenses_framing_matches_the_current_outbound_licence() -> None:
    """The file's FRAMING has to track the relicence, not just its facts.

    It used to argue "all permissive, therefore compatible with our Apache-2.0 posture".
    That sentence is now false in its premise, and a reader who trusts it concludes the
    wrong thing about what they may do — which is exactly the drift the doc-truth gates
    in this repo exist to stop.
    """
    text = (_REPO_ROOT / "THIRD-PARTY-LICENSES.md").read_text()
    lowered = text.lower()

    assert _SPDX_LICENSE in text, (
        "THIRD-PARTY-LICENSES does not name the project's current outbound licence"
    )
    assert "COMMERCIAL.md" in text
    assert not re.search(r"the latence framework is apache-2\.0", lowered), (
        "THIRD-PARTY-LICENSES still opens by declaring the project Apache-2.0"
    )
    assert not re.search(r"our apache-2\.0 posture", lowered), (
        "THIRD-PARTY-LICENSES still frames compatibility against the superseded posture"
    )
    # The honest asymmetry: an outbound licence is not a shield against inbound terms.
    assert "commercial licensee" in lowered or "binds the" in lowered


def test_lgpl_obligation_is_documented_precisely_not_waved_away() -> None:
    """`easydict` is LGPL-3.0. The conditions must be stated, with their section numbers.

    "LGPL is compatible with proprietary distribution" is true and useless on its own:
    it is compatible *subject to conditions*, and the conditions bite when a binary or
    container image is shipped. A commercial licensee acting on the hand-wave would
    convey a Combined Work without meeting §4.
    """
    text = (_REPO_ROOT / "THIRD-PARTY-LICENSES.md").read_text()
    lowered = text.lower()

    assert "easydict" in lowered
    assert "lgpl-3.0" in lowered
    assert "latence-linkpred-ultra[ultra]" in text, (
        "the LGPL dependency must be named with the exact opt-in extra that pulls it"
    )
    # The operative section, its lettered obligations, and the GPL-3.0 §6 cross-reference.
    assert "§4" in text, "LGPL-3.0 §4 (Combined Works) is not cited"
    assert "combined work" in lowered
    for clause in ("§4(a)", "§4(b)", "§4(c)", "§4(d)", "§4(e)"):
        assert clause in text, f"LGPL-3.0 {clause} is not documented"
    assert "§6" in text, (
        "the §4(e) Installation Information requirement cross-references GPL-3.0 §6; "
        "citing §4 without it understates the obligation for a User Product"
    )
    # The trigger has to be named, or a reader cannot tell whether it applies to them.
    assert "container image" in lowered and ("binary" in lowered or "frozen" in lowered), (
        "the document does not say WHEN the obligation is triggered (shipping a binary/image)"
    )
    # And it must be honest that today nothing is conveyed.
    assert "do not redistribute it" in lowered or "not conveyed" in lowered


def test_gitignore_denies_secret_patterns() -> None:
    """.gitignore blocks the file patterns a credential leak would ride in on."""
    lines = {ln.strip() for ln in (_REPO_ROOT / ".gitignore").read_text().splitlines()}
    for pat in (".env", "*.pem", "*.key", ".pypirc"):
        assert pat in lines, f".gitignore missing secret pattern: {pat}"


def test_every_package_has_pypi_release_metadata() -> None:
    """Each publishable package advertises the SPDX licence + classifiers + keywords + urls.

    The license assertion is the PEP 639 form the flip-public rehearsal validated: a bare
    SPDX expression (``license = "PolyForm-Noncommercial-1.0.0"``, which hatchling emits
    as ``License-Expression``), a ``license-files`` glob that resolves to a real file
    inside the distribution root, and NO deprecated ``License ::`` classifier alongside
    it. One package left on the old identifier would publish a wheel that declares
    Apache-2.0 while shipping the PolyForm text, so this is asserted per package and the
    failure names every offender at once.
    """
    problems: list[str] = []
    for pp in _package_pyprojects():
        proj = tomllib.loads(pp.read_text())["project"]
        name = pp.parent.name

        lic = proj.get("license")
        if lic != _SPDX_LICENSE:
            problems.append(
                f"{name}: license must be the bare SPDX expression {_SPDX_LICENSE!r} "
                f"(PEP 639), got {lic!r}"
                + (
                    " — this is the SUPERSEDED licence; see the provenance record in LICENSE"
                    if lic == _SUPERSEDED_SPDX_LICENSE
                    else ""
                )
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


def test_the_declared_licence_is_a_real_spdx_expression() -> None:
    """PyPI rejects a non-SPDX ``License-Expression``, and a typo is invisible by eye.

    ``PolyForm-Noncommercial-1.0.0`` is on the SPDX License List (non-OSI, not
    deprecated), so PEP 639 accepts it — but only spelled exactly that way. This runs the
    same normalisation the build backend and twine run, so a plausible-looking
    ``PolyForm-NonCommercial-1.0`` fails here rather than at upload time.
    """
    licenses = pytest.importorskip(
        "packaging.licenses", reason="packaging is needed to validate an SPDX expression"
    )
    canonical = licenses.canonicalize_license_expression(_SPDX_LICENSE)
    assert canonical == _SPDX_LICENSE

    for pp in _package_pyprojects():
        declared = tomllib.loads(pp.read_text())["project"]["license"]
        assert licenses.canonicalize_license_expression(declared) == _SPDX_LICENSE, (
            f"{pp.parent.name} declares a licence that does not canonicalise to "
            f"{_SPDX_LICENSE}: {declared!r}"
        )


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
