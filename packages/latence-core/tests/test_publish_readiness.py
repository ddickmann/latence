"""Regression guards for the OSS publish-readiness batch (issue #70) and the licence.

Deterministic, offline, dependency-free repo-hygiene tests. Each one fails if the
publish-readiness fix is reverted:

* the ``LICENSE`` really is the Apache License 2.0, reproduced VERBATIM (checked by
  digest over the WHOLE file against the canonical text, not by keyword), and it is
  mirrored byte-identically into every package that ships a wheel;
* ``NOTICE`` carries the copyright, the current grant, and the relicence provenance
  record — under Apache-2.0 the ``LICENSE`` file has to stay byte-canonical, so
  NOTICE (the file Apache-2.0 §4(d) gives operative force to) is where this project's
  own statements about its licence live;
* ``COMMERCIAL.md`` and the governance/security docs exist at the repo root with the
  expected, load-bearing content — including the fact that commercial use is GRANTED
  BY THE LICENCE and needs no separate agreement;
* every publishable workspace package's ``pyproject`` carries the PyPI metadata a
  release needs — the ``Apache-2.0`` SPDX expression, no deprecated ``License ::``
  classifier, Python-version classifiers, ``keywords``, and ``project.urls`` — and
  none of it silently displaced the package's ``dependencies`` (the
  ``[project.urls]`` sub-table must come after every plain ``[project]`` key or TOML
  re-parents them);
* ``.gitignore`` denies the secret patterns a credential leak would ride in on;
* the audited ``.secrets.baseline`` exists and every entry is audited
  ``is_secret == false`` (a genuine secret would not be pre-audited, so the CI
  ``detect-secrets`` step would surface it);
* **every** source file that declares an SPDX identifier declares the SAME one the
  packages declare. The licence topology is now uniform — there is no carve-out, no
  subset and no exception — so the gate is a universal rather than a pinned list.

LICENCE HISTORY. The project was Apache-2.0 internally from its first commit, was
relicensed to PolyForm Noncommercial 1.0.0 on 2026-07-29, and was relicensed BACK to
Apache-2.0 on 2026-08-15 (ADR-0063, superseding ADR-0055) — still without ever having
been published under either: private repository, nothing on PyPI, no release cut, sole
copyright holder, no CLA signatory. These gates are written so that a drift to
PolyForm — in the LICENSE text, in the NOTICE, in a package's metadata, or in a single
source file's SPDX header — fails the build rather than shipping a distribution whose
declared licence and shipped licence text disagree. They are deliberately as strict
about Apache-2.0 as their predecessors were about PolyForm: a licence claim is only
worth anything if it is falsifiable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

# tests/  ->  latence-core/  ->  packages/  ->  repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PACKAGES = _REPO_ROOT / "packages"


def _mkdocs_site_url() -> str:
    """The docs site URL mkdocs actually publishes, read from ``mkdocs.yml``.

    Read by line rather than by ``yaml.safe_load`` on purpose: ``mkdocs.yml`` carries
    mkdocs-material's ``!!python/name:`` tags, which ``safe_load`` refuses, and pulling in
    the full mkdocs loader to read one scalar would make this gate depend on the docs extra.
    """
    for line in (_REPO_ROOT / "mkdocs.yml").read_text().splitlines():
        if line.startswith("site_url:"):
            return line.split(":", 1)[1].strip().strip("\"'")
    raise AssertionError("mkdocs.yml declares no site_url — the docs site has no canonical URL")


_MKDOCS_SITE_URL = _mkdocs_site_url()

# The project's outbound licence, as the SPDX License List spells it.
_SPDX_LICENSE = "Apache-2.0"
# The licence this project carried between 2026-07-29 and 2026-08-15. Asserting its
# ABSENCE is what makes these gates catch a partial revert (e.g. one pyproject, one
# README paragraph or one source header quietly left behind on the old identifier).
_SUPERSEDED_SPDX_LICENSE = "PolyForm-Noncommercial-1.0.0"

# SHA-256 of the canonical Apache License 2.0 plain text as published by the Apache
# Software Foundation (https://www.apache.org/licenses/LICENSE-2.0.txt) — the same bytes
# `git show ab110b4^:LICENSE` restores, and the file whose MD5 is the widely-recognised
# 3b83ef96387f14655fc854ddc3c6bd57 that licence scanners key on.
#
# Under PolyForm this project wrapped the licence body in BEGIN/END markers so its own
# framing (provenance, Required Notice, pointers) could sit in the same file without
# being mistaken for terms; the digest was taken over the body between them. Apache-2.0
# has no such markers, and adding any would make the file stop matching what every
# scanner and every `licensecheck`-style tool expects. So the digest is now taken over
# the WHOLE file — which is STRICTER, not weaker: not only may no clause be edited, no
# preamble, footer or "helpful note" may be added either. Everything this project wants
# to say ABOUT its licence goes in NOTICE, which Apache-2.0 §4(d) already carries
# downstream. A keyword check ("Apache" in text) is not an acceptable substitute: it
# would pass happily on a licence with its patent grant or its warranty disclaimer
# quietly removed, which is precisely the failure that matters here.
_APACHE_TEXT_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
_APACHE_TEXT_MD5 = "3b83ef96387f14655fc854ddc3c6bd57"

# A file declares its own outbound grant with an SPDX identifier on a COMMENT line
# in its header. Only source files are scanned: in Markdown a leading ``#`` is a
# heading and LICENSE's own SPDX line is prose, so widening the scan would trade a
# real gate for false positives on files that merely *mention* an identifier.
_SPDX_SOURCE_SUFFIXES = (".py", ".pyi", ".rs", ".sh")
_SPDX_COMMENT = re.compile(r"^\s*(?:#|//)\s*SPDX-License-Identifier:\s*(\S+)\s*$")
_SPDX_HEADER_LINES = 24
_SPDX_SCAN_SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        # Parallel agent sessions materialise full sibling checkouts under .claude/worktrees;
        # scanning them would report SPDX headers on files that are not this checkout's.
        # Tooling state, never repo content.
        ".claude",
        "site",
        "build",
        "dist",
    }
)

# ---------------------------------------------------------------------------
# What a GRANT of the superseded licence looks like, as opposed to a mention of it.
# ---------------------------------------------------------------------------
# "was PolyForm until 2026-08-15", "supersedes ADR-0055", the provenance record and the
# CHANGELOG entry are HISTORY, and history must stay readable — the never-published
# record is the evidence the relicence rests on, so a gate that forbade the word would
# push the project into erasing its own audit trail. What is forbidden is a live CLAIM:
# a line that licenses the work under PolyForm, declares the identifier as this project's
# own, or tells the reader that commercial use needs permission it no longer needs.
_POLYFORM_GRANT_PATTERNS = (
    # "Licensed under PolyForm Noncommercial License 1.0.0"
    re.compile(r"[Ll]icensed under (?:the )?PolyForm"),
    # A machine-readable self-declaration, in any file that carries one.
    re.compile(r"SPDX-License-Identifier:\s*PolyForm"),
    # The licence document's own title, i.e. someone pasted the terms back in.
    re.compile(r"^#?\s*PolyForm Noncommercial License 1\.0\.0\s*$"),
)
_NONCOMMERCIAL_ONLY_CLAIM_PATTERNS = (
    # The operative sentence of the old model, in the phrasings this repo actually used.
    re.compile(r"[Cc]ommercial use requires"),
    re.compile(r"[Cc]ommercial (?:use|terms|licen[cs]e[sd]?) (?:are |is )?by agreement"),
    re.compile(r"commercial terms by agreement", re.IGNORECASE),
    re.compile(r"[Nn]oncommercial use is free"),
)

# Every workspace package is publishable to PyPI and must carry release metadata.
# PEP 639: the license is a bare SPDX *expression* plus a license-files glob. The legacy
# ``License :: OSI Approved ::`` classifier is deprecated and, combined with an SPDX
# expression, trips strict twine/PyPI validators — so its absence is asserted, not its
# presence (issue #72, item 2). The relicence back to Apache-2.0 does NOT make it correct
# again: ``License :: OSI Approved :: Apache Software License`` exists and would now be
# accurate, which is exactly why it has to stay asserted-absent — the tempting-but-wrong
# move is to "restore" it alongside the SPDX expression and re-break the upload.
_DEPRECATED_LICENSE_CLASSIFIER_PREFIX = "License ::"

# A trove classifier is a CLAIM, and the only evidence behind it is the CI matrix. So both
# classifier sets below are DERIVED from `.github/workflows/ci.yml` rather than written out:
# adding a matrix leg without widening the classifiers, or widening the classifiers without a
# lane to back them, fails here. That is the defect this pair of assertions exists for — the
# packages claimed `Operating System :: OS Independent` while every `runs-on:` in every
# workflow was `ubuntu-latest`, and `requires-python = ">=3.11"` while only 3.11/3.12 ran.
#
# `requires-python` is deliberately NOT derived from the matrix. It states what is *permitted*
# (nothing is known to break on a newer interpreter), while the classifiers state what is
# *tested*. Narrowing `requires-python` to the matrix would assert that 3.14 is broken, which
# nobody has measured — a different false claim, not a fix.
_CI_WORKFLOW = Path(__file__).resolve().parents[3] / ".github" / "workflows" / "ci.yml"

# GitHub runner label -> the trove classifier that runner is evidence for.
_RUNNER_OS_CLASSIFIERS = {
    "ubuntu": "Operating System :: POSIX :: Linux",
    "macos": "Operating System :: MacOS",
    "windows": "Operating System :: Microsoft :: Windows",
}


def _ci_test_matrix() -> dict[str, Any]:
    workflow = yaml.safe_load(_CI_WORKFLOW.read_text())
    matrix = workflow["jobs"]["test"]["strategy"]["matrix"]
    assert isinstance(matrix, dict), "ci.yml job `test` declares no matrix"
    return matrix


def _tested_python_versions() -> set[str]:
    matrix = _ci_test_matrix()
    versions = {str(v) for v in matrix.get("python-version", [])}
    for entry in matrix.get("include", []):
        if "python-version" in entry:
            versions.add(str(entry["python-version"]))
    assert versions, "ci.yml job `test` exercises no interpreter"
    return versions


def _tested_operating_systems() -> set[str]:
    matrix = _ci_test_matrix()
    runners = {str(v) for v in matrix.get("os", [])}
    for entry in matrix.get("include", []):
        if "os" in entry:
            runners.add(str(entry["os"]))
    assert runners, "ci.yml job `test` names no runner OS"

    classifiers = set()
    for runner in runners:
        family = runner.split("-")[0]
        assert family in _RUNNER_OS_CLASSIFIERS, (
            f"ci.yml runs on {runner!r}, which maps to no known trove OS classifier — add it "
            "to _RUNNER_OS_CLASSIFIERS so the packages can honestly claim that platform."
        )
        classifiers.add(_RUNNER_OS_CLASSIFIERS[family])
    return classifiers


_PY_VERSION_CLASSIFIERS = {
    f"Programming Language :: Python :: {v}" for v in _tested_python_versions()
}
_OS_CLASSIFIERS = _tested_operating_systems()


def _package_pyprojects() -> list[Path]:
    out = [
        pkg / "pyproject.toml"
        for pkg in sorted(_PACKAGES.iterdir())
        if (pkg / "pyproject.toml").is_file() and (pkg / "src").is_dir()
    ]
    assert out, "no publishable packages discovered under packages/"
    return out


def _declared_file_licences() -> dict[str, str]:
    """``repo-relative path -> SPDX identifier`` for every source file that declares one."""
    found: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(_REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in _SPDX_SCAN_SKIP_DIRS]
        for name in filenames:
            if not name.endswith(_SPDX_SOURCE_SUFFIXES):
                continue
            path = Path(dirpath) / name
            try:
                lines = path.read_text(encoding="utf-8").splitlines()[:_SPDX_HEADER_LINES]
            except (OSError, UnicodeDecodeError):  # pragma: no cover - defensive
                continue
            for line in lines:
                match = _SPDX_COMMENT.match(line)
                if match:
                    found[path.relative_to(_REPO_ROOT).as_posix()] = match.group(1)
                    break
    return found


def test_license_is_apache_2_0_verbatim() -> None:
    """LICENSE is the canonical Apache License 2.0 text, byte for byte, and NOTHING else.

    Checked by digest over the WHOLE file, because the whole value of using a
    professionally drafted licence is lost the moment a clause is silently edited — and a
    keyword assertion ("Apache" in text) would happily pass on a licence with its §3
    patent grant or its §7 warranty disclaimer removed.

    Whole-file rather than a delimited body (which is what the PolyForm version of this
    gate checked) because Apache-2.0 is recognised by scanners on exactly these bytes: an
    added preamble, a "friendly note" or a trailing pointer would leave the file matching
    nothing, and every SCA tool in an enterprise procurement pipeline reads this file
    before a human does. Anything this project wants to SAY about its licence belongs in
    ``NOTICE`` — the file Apache-2.0 §4(d) already carries downstream — and the test
    below asserts it is there.
    """
    raw = (_REPO_ROOT / "LICENSE").read_bytes()

    sha256 = hashlib.sha256(raw).hexdigest()
    md5 = hashlib.md5(raw, usedforsecurity=False).hexdigest()
    assert sha256 == _APACHE_TEXT_SHA256 and md5 == _APACHE_TEXT_MD5, (
        "LICENSE is NOT the canonical Apache License 2.0 plain text.\n"
        f"  expected sha256: {_APACHE_TEXT_SHA256}\n  actual sha256:   {sha256}\n"
        f"  expected md5:    {_APACHE_TEXT_MD5}\n  actual md5:      {md5}\n"
        "Restore it verbatim from https://www.apache.org/licenses/LICENSE-2.0.txt. Note "
        "that ADDING text is a failure here too, by design: the copyright line, the "
        "provenance record and every pointer belong in NOTICE, not in LICENSE."
    )

    # Sanity anchors that a digest alone would not make legible in a failure report.
    text = raw.decode("utf-8")
    assert "Apache License" in text and "Version 2.0, January 2004" in text
    assert "3. Grant of Patent License." in text, "the patent grant is missing"
    # ...and the superseded licence must not be lurking in the terms.
    assert "PolyForm" not in text


def test_notice_carries_the_relicence_provenance_record() -> None:
    """The provenance note is what a future reader, acquirer or lawyer looks for FIRST.

    Both relicences are clean only because no grant ever reached anyone. That claim has
    to be written down as CHECKABLE FACTS, or it is folklore that dies with the person
    who remembers it — and it matters more now, not less: the tree previously carried a
    PolyForm identifier in 129 file headers and 33 package manifests, so a future reader
    WILL find PolyForm in the history and needs the record that says why it is not the
    licence today.

    It lives in ``NOTICE`` rather than ``LICENSE`` because ``LICENSE`` must stay
    byte-canonical (the test above). That is a relocation, not a relaxation: the same
    facts are asserted, and ``NOTICE`` is the file Apache-2.0 §4(d) requires downstream
    redistributors to carry, so the record travels with the code exactly as it did.
    """
    text = (_REPO_ROOT / "NOTICE").read_text(encoding="utf-8")

    assert "PROVENANCE" in text, "NOTICE has no provenance section"
    assert "not part of the licence terms" in text.lower(), (
        "the provenance note must say, in the file, that it is not part of the terms — "
        "otherwise it reads as an amendment to a licence that must stay verbatim"
    )
    lowered = text.lower()
    for fact in (
        "polyform",  # the interlude it exists to explain
        "2026-07-29",  # relicensed away from Apache-2.0
        "2026-08-15",  # ...and back
        "never published",
        "private",
        "pypi",
        "sole copyright holder",
    ):
        assert fact in lowered, f"the provenance record does not state: {fact}"
    # The reasoning has to be reachable, not just the fact.
    assert "adr-0063" in lowered or "0063-" in lowered, (
        "the provenance record does not point at the ADR that decided the relicence"
    )
    # Apache-2.0 §4(d) attribution: the notice a downstream copy has to carry.
    assert "Required Notice: Copyright 2026 Dennis Dickmann" in text, (
        "NOTICE carries no `Required Notice:` line, so no downstream copy can carry the "
        "copyright attribution Apache-2.0 §4(d) propagates"
    )
    assert f"SPDX identifier {_SPDX_LICENSE}" in text or f"SPDX: {_SPDX_LICENSE}" in text


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
    assert "Apache License, Version 2.0" in text
    # A NOTICE that names the licence but leaves commercial use ambiguous is the exact
    # doubt an enterprise reader is here to resolve — say it, in the file. Compared on
    # whitespace-collapsed text because NOTICE is hard-wrapped at 79 columns and a
    # sentence that reflows across a line break is the same sentence.
    assert "Commercial use is granted by that licence" in " ".join(text.split()), (
        "NOTICE does not state that the licence itself grants commercial use, so a "
        "reader who remembers the PolyForm period is left guessing"
    )
    assert "Required Notice:" in text
    assert not any(p.search(text) for p in _POLYFORM_GRANT_PATTERNS), (
        "NOTICE still grants the superseded PolyForm Noncommercial licence"
    )
    # EmbeddingGemma must be flagged NOT-default with flow-down, never silently listed as permissive.
    assert "embeddinggemma" in text.lower()
    assert "NOT a default" in text or "not a default" in text.lower()
    assert "flow-down" in text.lower()


def test_governance_and_third_party_docs_exist() -> None:
    """The OSS front-door governance set is present."""
    for name in (
        "CONTRIBUTING.md",
        "CLA.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "THIRD-PARTY-LICENSES.md",
        "COMMERCIAL.md",
        "SUPPORT.md",
        "CITATION.cff",
    ):
        p = _REPO_ROOT / name
        assert p.is_file() and p.read_text().strip(), f"missing or empty: {name}"


def test_the_cla_requirement_is_satisfiable() -> None:
    """A mandatory CLA needs an agreement a contributor can actually sign.

    ``CONTRIBUTING.md`` makes a signed CLA a precondition of every merge and used to say the
    text was "maintained separately from this page" — while nothing in the tree was that
    document and no CLA automation existed. On day one of a public repository the first pull
    request could not be merged by anyone, because the stated precondition was unmeetable.
    That is the defect this asserts against: not that a file exists, but that the four things
    which make it usable rather than decorative are all true at once.
    """
    text = (_REPO_ROOT / "CLA.md").read_text(encoding="utf-8")
    contributing = (_REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")

    assert "](CLA.md)" in contributing or "](./CLA.md)" in contributing, (
        "CONTRIBUTING.md requires a signed CLA but never points at the agreement"
    )
    # Signing has to route somewhere real. This is the contact SECURITY.md, COMMERCIAL.md and
    # CODE_OF_CONDUCT.md already use; a procedure naming an invented address is not a procedure.
    assert "ddickmann81@gmail.com" in text, "CLA.md names no address to send a signature to"
    assert "ddickmann81@gmail.com" in contributing, (
        "CONTRIBUTING.md states the CLA requirement but not where a signature goes"
    )
    # The grant that makes this a CLA and not a DCO. Under ADR-0055's noncommercial model
    # this was load-bearing: without the right to license a contribution onward on other
    # terms, every commercial licence granted after that contribution was partly
    # unauthorised. ADR-0063 removed that need — Apache-2.0 is inbound=outbound, so a DCO
    # would now suffice — but retiring the CLA is the owner's call, not this gate's, and
    # while CLA.md is the instrument CONTRIBUTING.md requires, it must still contain the
    # grant that makes it one rather than a decorative signature ritual.
    assert "Grant of Copyright Licence" in text and "sublicense" in text
    assert "commercial and proprietary terms" in text, (
        "CLA.md does not grant the relicensing right that is the entire reason it exists"
    )
    # And it must not pass itself off as reviewed. The review is Gate 3's, and it is open.
    assert "not been reviewed by a lawyer" in text
    assert "RELEASE-GATES.md" in text, (
        "CLA.md must point at the gate that carries its outstanding legal review"
    )


def test_citation_cff_is_parseable_and_claims_nothing_unreleased() -> None:
    """``CITATION.cff`` renders GitHub's "Cite this repository" button — if it parses.

    A malformed CFF is not a silent no-op: GitHub surfaces an error banner on the repository
    landing page, so a typo here is a front-door defect. The two substantive assertions are
    that it points at the same docs site and repository everything else points at, and that
    it does NOT carry ``version``/``date-released`` — no release has been cut, and a version
    in the citation block is exactly the class of claim CHANGELOG.md's ``0.1.0`` entry had to
    be corrected for.
    """
    data = yaml.safe_load((_REPO_ROOT / "CITATION.cff").read_text())

    assert data["cff-version"] == "1.2.0"
    for required in ("message", "title", "authors", "repository-code"):
        assert data.get(required), f"CITATION.cff is missing {required!r}"

    assert data["url"] == _MKDOCS_SITE_URL, (
        f"CITATION.cff points at {data['url']!r} but mkdocs publishes {_MKDOCS_SITE_URL!r}"
    )
    core_urls = tomllib.loads((_PACKAGES / "latence-core" / "pyproject.toml").read_text())[
        "project"
    ]["urls"]
    assert data["repository-code"] == core_urls["Repository"], (
        "CITATION.cff and latence-core's [project.urls] name different repositories"
    )

    for forbidden in ("version", "date-released", "doi"):
        assert forbidden not in data, (
            f"CITATION.cff declares {forbidden!r}, but nothing has been released "
            "(no v* tag, nothing on PyPI). Add it with the first real tag, not before."
        )


def test_commercial_md_says_commercial_use_is_granted_and_gates_nothing() -> None:
    """COMMERCIAL.md must now REMOVE a gate, not describe one.

    The page kept its name and its inbound links — README, NOTICE, CONTRIBUTING, the docs
    nav, `mkdocs.yml`, ADR-0055 and ADR-0062 all reach it, and ADR-0055's pointer is
    exactly the route by which a reader arrives believing they need to buy something. So
    deleting it would have left every one of those readers at a 404 with their last
    information being "commercial use requires a separate written agreement".

    Under Apache-2.0 that sentence is false. This gate is therefore the mirror image of
    the one it replaces, and just as strict: the old version asserted the page routed a
    commercial request and invented no terms; this one asserts the page GRANTS, in so many
    words, and that no gate has crept back in — no agreement requirement, no price, no
    tier, no "contact us before you deploy". The one thing it may still say no about is
    third-party terms, which an outbound licence never covered and still does not.
    """
    text = (_REPO_ROOT / "COMMERCIAL.md").read_text()
    lowered = text.lower()
    # Hard-wrapped prose: a sentence reflowed across a line break is the same sentence.
    flat = " ".join(lowered.split())

    assert _SPDX_LICENSE in text
    # The headline answer, unambiguous enough that a reader can stop reading here.
    assert "no separate agreement is required" in flat, (
        "COMMERCIAL.md does not state plainly that commercial use needs no agreement — "
        "which is the single fact a reader arriving from ADR-0055 or an old README is "
        "here to have corrected"
    )
    assert "ddickmann81@gmail.com" in text, "no contact for the maintainer"

    # No gate may reappear, in any of the phrasings the PolyForm model actually used.
    for pattern in (*_POLYFORM_GRANT_PATTERNS, *_NONCOMMERCIAL_ONLY_CLAIM_PATTERNS):
        offending = [ln.strip() for ln in text.splitlines() if pattern.search(ln)]
        assert not offending, (
            "COMMERCIAL.md re-imposes a commercial gate the licence does not impose:\n  "
            + "\n  ".join(offending)
        )
    assert not re.search(r"[$€£]\s?\d", text), "COMMERCIAL.md quotes a price"
    for invented in ("per seat", "per-seat", "annual fee", "subscription fee", "free trial"):
        assert invented not in lowered, f"COMMERCIAL.md invents a commercial term: {invented}"

    # It must not claim to be legal advice, and must still be honest about what an
    # outbound grant does NOT reach — that asymmetry survives the relicence untouched.
    assert "not legal advice" in lowered
    assert "easydict" in lowered, (
        "COMMERCIAL.md must tell a commercial user that third-party terms — the LGPL-3.0 "
        "easydict extra above all — are not reached by this project's outbound licence"
    )


def test_no_owned_licence_document_still_grants_polyform_noncommercial() -> None:
    """A partial revert is the realistic failure: one file left behind on the old licence.

    Scoped to the files this relicence owns, so it is a real gate and not a proxy: the
    root licence documents, and every package's shipped LICENSE.

    There is now NO exception to exclude. Under PolyForm this gate had to carve out the
    retained-modules section by its own delimiters; the carve-out is gone with the model
    it existed inside, so the rule is a flat universal — which is strictly harder to
    satisfy than the version it replaces, because no region of any owned document is
    exempt from it any more.

    Two distinct failures are caught, because a stale page can be wrong in two ways: it
    can still GRANT PolyForm, or it can still tell the reader that commercial use needs
    permission. The second is the more likely and the more damaging — an enterprise reader
    who hits "commercial use requires a separate written agreement" stops there and never
    reaches the Apache grant that is now true.
    """
    grant_offenders: list[str] = []
    claim_offenders: list[str] = []
    for path in (
        _REPO_ROOT / "LICENSE",
        _REPO_ROOT / "NOTICE",
        _REPO_ROOT / "COMMERCIAL.md",
        *sorted(_PACKAGES.glob("*/LICENSE")),
    ):
        for line in path.read_text(encoding="utf-8").splitlines():
            # "was PolyForm until 2026-08-15", "supersedes ADR-0055", the provenance
            # record: those are history, not a grant. A GRANT is a line that licenses the
            # work under it; a CLAIM is a line that tells the reader what they may do.
            rel = path.relative_to(_REPO_ROOT)
            if any(p.search(line) for p in _POLYFORM_GRANT_PATTERNS):
                grant_offenders.append(f"{rel}: {line.strip()}")
            elif any(p.search(line) for p in _NONCOMMERCIAL_ONLY_CLAIM_PATTERNS):
                claim_offenders.append(f"{rel}: {line.strip()}")
    assert not grant_offenders and not claim_offenders, (
        "the outbound licence is Apache-2.0, which grants commercial use outright.\n"
        + ("  STILL GRANTS PolyForm:\n    " + "\n    ".join(grant_offenders) + "\n" if grant_offenders else "")
        + (
            "  STILL CLAIMS commercial use is restricted:\n    " + "\n    ".join(claim_offenders)
            if claim_offenders
            else ""
        )
    )


def test_no_package_readme_claims_its_own_code_is_polyform_or_noncommercial() -> None:
    """A package README stating the FRAMEWORK's own code licence must state Apache-2.0.

    The 2026-07-29 relicence updated LICENSE/NOTICE/pyprojects but left sixteen README
    prose claims behind, because the grant-line gate above only matches a grant sentence.
    The 2026-08-15 relicence back has exactly the same hole in exactly the same sixteen
    places, so this gate is kept and inverted rather than deleted: a claim that the
    package's or the framework's OWN code is PolyForm-licensed, or that using it
    commercially needs an agreement.

    Third-party statements stay legal, and that is what keeps this gate honest rather
    than a blanket word-ban: `latence-benchmark` documents research-only corpora
    (OmniDocBench), and those really are
    noncommercial — the framework's outbound licence does not change a third party's
    terms. So the patterns match claims about *this* code (framework code, this package,
    the repository-wide licence), never the word "noncommercial" on its own.
    """
    own_code_polyform = (
        re.compile(r"[Ff]ramework code,? (is )?\*{0,2}PolyForm"),
        re.compile(r"[Ll]icensed under (the )?PolyForm"),
        re.compile(r"PolyForm[- ]Noncommercial[^\n]{0,40}\((this|the) package"),
        re.compile(r"PolyForm[^\n]{0,60}\(framework code"),
        re.compile(r"this framework code:? \*{0,2}PolyForm"),
        re.compile(r"PolyForm[^\n]{0,80}repository-wide licence"),
        # The old model's operative sentence, wherever a README restated it.
        re.compile(r"commercial terms by agreement", re.IGNORECASE),
    )
    offenders: list[str] = []
    for path in sorted(_PACKAGES.glob("*/README.md")):
        for line in path.read_text(encoding="utf-8").splitlines():
            for pattern in own_code_polyform:
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(_REPO_ROOT)}: {line.strip()}")
                    break
    assert not offenders, (
        "package READMEs still claim the framework's own code is PolyForm-licensed or "
        "commercially restricted; the repository licence is Apache-2.0 (LICENSE), with no "
        "exception and no carve-out:\n" + "\n".join(offenders)
    )


def test_every_declared_file_licence_is_the_outbound_licence() -> None:
    """The file-level SPDX topology is UNIFORM: every declaring file says Apache-2.0.

    This replaces the pinned seven-module carve-out that the PolyForm model needed, and it
    is a strictly stronger statement than that pin was: the old gate allowed a bounded set
    of files to disagree with the packages' declared licence, this one allows none. A
    single stray ``SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0`` left behind by
    a rebase, or a well-meaning ``MIT`` header copied in from elsewhere, is a
    file-versus-metadata contradiction that a licence scanner reports and a human then has
    to adjudicate — which is the cost this gate exists to prevent.

    The scan is over the whole tree, not a list, so a NEW file cannot escape it by not
    being enumerated anywhere.
    """
    declared = _declared_file_licences()
    assert declared, "the SPDX scan found no source file at all — the walk is broken"

    divergent = sorted(
        f"{path} -> {spdx}" for path, spdx in declared.items() if spdx != _SPDX_LICENSE
    )
    assert not divergent, (
        f"these source files declare a licence other than the repository's {_SPDX_LICENSE}:\n  "
        + "\n  ".join(divergent)
        + "\nThere is no carve-out any more (ADR-0063). If one of these is "
        + f"{_SUPERSEDED_SPDX_LICENSE}, it is a leftover from the 2026-07-29 relicence; flip "
        "it. If a genuine exception is ever wanted, it must be decided in an ADR, stated in "
        "the file, and pinned here — not left to be discovered by a scanner."
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

    It has now been wrong in both directions. Under Apache-2.0 it argued "all permissive,
    therefore compatible with our Apache-2.0 posture"; the 2026-07-29 relicence made that
    premise false and it was rewritten to a PolyForm framing; the 2026-08-15 relicence
    back makes THAT false in turn. A reader who trusts a stale framing concludes the wrong
    thing about what they may do — exactly the drift the doc-truth gates in this repo
    exist to stop — so the gate is inverted rather than retired.
    """
    text = (_REPO_ROOT / "THIRD-PARTY-LICENSES.md").read_text()
    lowered = text.lower()

    assert _SPDX_LICENSE in text, (
        "THIRD-PARTY-LICENSES does not name the project's current outbound licence"
    )
    assert not any(p.search(text) for p in _POLYFORM_GRANT_PATTERNS), (
        "THIRD-PARTY-LICENSES still declares the project PolyForm-licensed"
    )
    assert not re.search(r"the latence framework is polyform", lowered), (
        "THIRD-PARTY-LICENSES still opens by declaring the project PolyForm"
    )
    assert not re.search(r"commercial use (?:of this repository )?(?:is |requires)", lowered), (
        "THIRD-PARTY-LICENSES still frames the reader's rights against the superseded "
        "noncommercial posture"
    )
    # The honest asymmetry, which the relicence does NOT change: an outbound licence is
    # never a shield against inbound terms. Apache-2.0 grants commercial use of THIS code
    # and reaches nothing in this document.
    assert "binds the" in lowered or "bind the" in lowered or "binds you" in lowered, (
        "the document no longer says that third-party terms bind the user directly, which "
        "is the one thing it exists to say"
    )


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
    SPDX expression (``license = "Apache-2.0"``, which hatchling emits as
    ``License-Expression``), a ``license-files`` glob that resolves to a real file inside
    the distribution root, and NO deprecated ``License ::`` classifier alongside it. One
    package left on the old identifier would publish a wheel that declares PolyForm while
    shipping the Apache-2.0 text — and would be hard-blocked by the SCA scanners the
    relicence exists to get past — so this is asserted per package and the failure names
    every offender at once.
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
        # Both classifier sets are derived from the CI matrix (see above), so these two
        # assertions are "the package claims exactly what CI runs", in both directions.
        # The bare ``Programming Language :: Python :: 3`` is a family marker, not a version
        # claim; only the dotted ones say "this interpreter is supported".
        declared_py = {
            c for c in classifiers if c.startswith("Programming Language :: Python :: 3.")
        }
        if declared_py != _PY_VERSION_CLASSIFIERS:
            problems.append(
                f"{name}: Python-version classifiers {sorted(declared_py)} do not match the "
                f"interpreters ci.yml tests, {sorted(_PY_VERSION_CLASSIFIERS)}"
            )

        declared_os = {c for c in classifiers if c.startswith("Operating System ::")}
        if "Operating System :: OS Independent" in declared_os:
            problems.append(
                f"{name}: claims 'Operating System :: OS Independent'. Nothing tests that — "
                "name the platforms ci.yml actually runs on: "
                f"{sorted(_OS_CLASSIFIERS)}"
            )
        elif declared_os != _OS_CLASSIFIERS:
            problems.append(
                f"{name}: OS classifiers {sorted(declared_os)} do not match the platforms "
                f"ci.yml tests, {sorted(_OS_CLASSIFIERS)}"
            )

        if not proj.get("keywords"):
            problems.append(f"{name}: no keywords")
        if not proj.get("description"):
            problems.append(f"{name}: no description")
        if proj.get("readme") != "README.md":
            problems.append(f"{name}: readme != README.md")

        urls = proj.get("urls", {})
        # ``Documentation`` is required, not optional: PyPI renders it as the "Documentation"
        # sidebar link, and a package page with no route to the docs site sends the reader
        # straight back to a bare repository. The docs site is built and deployed by
        # `.github/workflows/docs.yml`, so the URL is not aspirational.
        for key in ("Homepage", "Repository", "Issues", "Documentation"):
            if key not in urls:
                problems.append(f"{name}: project.urls missing {key}")

        # And it must be the SAME site mkdocs publishes. Two spellings of the docs URL is how
        # one of them silently rots after a rename (the org/repo name is not settled — every
        # URL in the tree moves together when it is).
        documentation = urls.get("Documentation")
        if documentation is not None and documentation != _MKDOCS_SITE_URL:
            problems.append(
                f"{name}: project.urls Documentation is {documentation!r}, but mkdocs.yml "
                f"publishes {_MKDOCS_SITE_URL!r} — they must be the same site"
            )

    assert not problems, "PyPI metadata problems:\n" + "\n".join(problems)


def test_the_declared_licence_is_a_real_spdx_expression() -> None:
    """PyPI rejects a non-SPDX ``License-Expression``, and a typo is invisible by eye.

    ``Apache-2.0`` is on the SPDX License List and PEP 639 accepts it — but only spelled
    exactly that way. This runs the same normalisation the build backend and twine run, so
    a plausible-looking ``Apache 2.0`` or ``Apache-2`` fails here rather than at upload
    time. It is not redundant with the equality check above: ``canonicalize`` also accepts
    deprecated spellings and compound expressions that a bare string comparison would
    reject with a less useful message.
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
    # The workspace ROOT that intentionally declares no latence-core dependency:
    # latence-core itself. It still must expose a
    # non-empty [project.dependencies] (the footgun guard) — it is exempt only from the
    # latence-core-edge proxy, not from the "dependencies survived parsing" check.
    _standalone_roots = {"latence-core"}
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


def test_pre_commit_hooks_run_the_same_commands_as_ci() -> None:
    """A commit hook that has drifted from CI is worse than no commit hook.

    ``.pre-commit-config.yaml`` exists to run the two fastest CI gates earlier, not to be a
    second, subtly different gate: the moment the ruff scope or the secret-gate invocation
    differs, a green commit stops predicting a green build and people learn to distrust the
    hook. So each hook's ``entry:`` must appear VERBATIM as a command in ``ci.yml``.

    The comparison is on the command string with the workflow's ``uv run`` prefix intact,
    because that prefix is what makes both run inside the synced workspace rather than
    against whatever interpreter happens to be first on PATH.
    """
    config = _REPO_ROOT / ".pre-commit-config.yaml"
    assert config.is_file(), (
        "missing .pre-commit-config.yaml — the commit-time mirror of the CI lint + secret gates"
    )
    data = yaml.safe_load(config.read_text())

    ci = (_REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    # The workflow wraps long commands, so compare on whitespace-collapsed text.
    ci_flat = " ".join(ci.split())

    entries = [
        hook["entry"] for repo in data["repos"] for hook in repo["hooks"] if repo["repo"] == "local"
    ]
    assert entries, ".pre-commit-config.yaml declares no local hooks"

    for entry in entries:
        assert " ".join(entry.split()) in ci_flat, (
            f"pre-commit hook runs {entry!r}, which is not a command in ci.yml. Either the "
            "hook or the workflow drifted; make them identical again."
        )

    # And the two gates that MUST be mirrored are actually present, so the check above
    # cannot be satisfied by an empty-but-well-formed config.
    joined = " ".join(entries)
    assert "ruff check" in joined, "pre-commit does not mirror the CI ruff lint step"
    assert "secret_scan_gate.py" in joined, "pre-commit does not mirror the CI secret-scan gate"


def test_verify_local_mirrors_ci_lint_scope() -> None:
    """``scripts/verify-local.sh`` lints exactly what CI lints — same paths, same order.

    The script's entire promise is "what CI will say, before you push". A narrower scope is
    worse than no script, because it reports green on a tree CI will reject — which is what
    happened: `verify-local.sh` checked only ``packages`` while CI checked the benchmark
    harnesses, the campaign scripts and ``scripts/`` too, and two real violations sat in that
    gap. Pinning the two argument lists to each other is what stops it recurring; a comment
    asking the next person to remember would not.
    """
    ci = " ".join((_REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text().split())
    script = " ".join((_REPO_ROOT / "scripts" / "verify-local.sh").read_text().split())

    match = re.search(r"uv run ruff check ([^\n&|;]+?) ok ", script)
    assert match, "verify-local.sh no longer has a recognisable `uv run ruff check` step"
    scope = match.group(1).strip()

    assert f"uv run ruff check {scope}" in ci, (
        f"verify-local.sh lints {scope!r}, which is not what ci.yml's Ruff step lints. The "
        "local script must mirror CI exactly or it reports a green that CI will not honour."
    )
    # Not satisfiable by an empty or one-path scope: the harnesses and scripts/ are the
    # paths that were missing, so name them.
    for required in ("packages", "benchmark/sota-campaign", "scripts"):
        assert required in scope, f"verify-local.sh's ruff scope no longer covers {required}"


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
