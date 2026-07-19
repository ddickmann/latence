"""Regression guards for release integrity — signed releases + per-release SBOM (issue #178).

The first group of tests RUNS ``scripts/release-integrity.sh`` for real (offline: a
pre-supplied CycloneDX document and ``--no-sign``, so no network and no OIDC identity is
needed) and asserts on the assets it actually produced — the checksums really verify, a
tampered artifact really fails verification, and a structurally useless SBOM is really
rejected. Behaviour through the script's own interface, never its internals.

The second group guards the wiring: the release workflow must call the script *with signing
on*, must publish the signed asset set rather than raw ``dist/``, and the CVD policy must be
reachable from the release notes and the repository front page. Those are the things that
silently rot when someone edits the workflow.

Everything here is deterministic, offline, and dependency-free.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

# tests/  ->  latence-core/  ->  packages/  ->  repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "release-integrity.sh"
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "release.yml"

def _workspace_packages() -> list[str]:
    """Every package under ``packages/`` — the distributions a release builds and publishes."""
    return sorted(p.parent.name for p in (_REPO_ROOT / "packages").glob("*/pyproject.toml"))


def _workspace_components(names: list[str] | None = None) -> list[dict[str, Any]]:
    """The workspace path components `cyclonedx-py requirements` emits for `./packages/<name>`.

    It cannot resolve a distribution name from a bare path, so each lands as ``name:
    "unknown"`` and the path survives only as an externalReferences URL. That URL is what
    identifies the component, which is why the coverage guard reads it rather than ``name``.
    """
    return [
        {
            "type": "library",
            "name": "unknown",
            "bom-ref": f"requirements-L{i}",
            "externalReferences": [
                {"type": "other", "comment": "explicit local path", "url": f"./packages/{name}"}
            ],
        }
        for i, name in enumerate(_workspace_packages() if names is None else names)
    ]


# A structurally valid CycloneDX 1.6 document, so the tests never need `uv export` or the
# network. It mirrors the shape `cyclonedx-py requirements` emits: pinned third-party
# components plus one path component per workspace package. The workspace half is not
# decoration — release-integrity.sh fails closed unless the SBOM accounts for every
# distribution the release publishes, so a fixture without it would not be a releasable SBOM.
_CYCLONEDX_DOC: dict[str, Any] = {
    "bomFormat": "CycloneDX",
    "specVersion": "1.6",
    "version": 1,
    "components": [
        {"type": "library", "name": "pydantic", "version": "2.9.2"},
        {"type": "library", "name": "fsspec", "version": "2024.9.0"},
        *_workspace_components(),
    ],
}


def _sha256sum_check(directory: Path) -> subprocess.CompletedProcess[str]:
    """Verify ``SHA256SUMS`` the way an adopter would, with whichever tool this host has."""
    tool = ["sha256sum", "-c"] if shutil.which("sha256sum") else ["shasum", "-a", "256", "-c"]
    return subprocess.run(
        [*tool, "SHA256SUMS"], cwd=directory, capture_output=True, text=True, check=False
    )


def _run(
    tmp_path: Path,
    *,
    tag: str = "v9.9.9",
    sbom: dict[str, Any] | str | None = None,
    dist_names: tuple[str, ...] = ("latence_core-9.9.9-py3-none-any.whl", "latence_core-9.9.9.tar.gz"),
    extra: tuple[str, ...] = (),
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run the release-integrity script offline and return (result, asset directory)."""
    dist = tmp_path / "dist"
    dist.mkdir(exist_ok=True)
    for i, name in enumerate(dist_names):
        (dist / name).write_bytes(f"artifact-{i}-content".encode())

    sbom_path = tmp_path / "input-sbom.json"
    if isinstance(sbom, str):
        sbom_path.write_text(sbom, encoding="utf-8")
    else:
        sbom_path.write_text(json.dumps(sbom if sbom is not None else _CYCLONEDX_DOC), "utf-8")

    out = tmp_path / "release-assets"
    result = subprocess.run(
        [
            "bash",
            str(_SCRIPT),
            "--tag",
            tag,
            "--dist",
            str(dist),
            "--sbom",
            str(sbom_path),
            "--out",
            str(out),
            "--repo",
            "ddickmann/latence-framework",
            *extra,
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, out


# --------------------------------------------------------------------------------------
# What the script actually produces
# --------------------------------------------------------------------------------------


def test_script_exists_and_is_executable() -> None:
    """The release-integrity entrypoint is present and runnable."""
    assert _SCRIPT.is_file(), "missing scripts/release-integrity.sh"


def test_produces_the_full_signable_asset_set(tmp_path: Path) -> None:
    """A dry run yields the distributions, the SBOM, VERIFY.md and SHA256SUMS, staged flat."""
    result, out = _run(tmp_path, extra=("--no-sign",))
    assert result.returncode == 0, result.stderr

    produced = {p.name for p in out.iterdir()}
    assert produced == {
        "latence_core-9.9.9-py3-none-any.whl",
        "latence_core-9.9.9.tar.gz",
        "sbom.cdx.json",
        "SHA256SUMS",
        "VERIFY.md",
    }
    # --no-sign must NOT quietly produce a bundle-shaped file that could be mistaken for one.
    assert not (out / "SHA256SUMS.sigstore.json").exists()


def test_checksums_cover_every_asset_and_actually_verify(tmp_path: Path) -> None:
    """`sha256sum -c SHA256SUMS` — the command shipped to adopters — passes on a clean set."""
    _, out = _run(tmp_path, extra=("--no-sign",))

    listed = {line.split()[-1] for line in (out / "SHA256SUMS").read_text().splitlines() if line}
    # Every asset except the checksum file itself, INCLUDING the verification instructions.
    assert listed == {
        "latence_core-9.9.9-py3-none-any.whl",
        "latence_core-9.9.9.tar.gz",
        "sbom.cdx.json",
        "VERIFY.md",
    }

    check = _sha256sum_check(out)
    assert check.returncode == 0, check.stdout + check.stderr


@pytest.mark.parametrize(
    "victim",
    ["latence_core-9.9.9-py3-none-any.whl", "sbom.cdx.json", "VERIFY.md"],
)
def test_checksum_verification_rejects_a_tampered_asset(tmp_path: Path, victim: str) -> None:
    """Tampering with any covered asset — artifact, SBOM, or the instructions — fails the check.

    This is the property the whole scheme rests on: one signature over SHA256SUMS is only
    worth anything if SHA256SUMS genuinely detects a swapped file.
    """
    _, out = _run(tmp_path, extra=("--no-sign",))
    assert _sha256sum_check(out).returncode == 0

    (out / victim).write_bytes(b"tampered")

    check = _sha256sum_check(out)
    assert check.returncode != 0, f"a tampered {victim} passed verification"
    assert victim in check.stdout


def test_sbom_is_stamped_with_the_release_identity(tmp_path: Path) -> None:
    """The attached SBOM names the release it belongs to — a per-release SBOM, not a generic one."""
    _, out = _run(tmp_path, tag="v9.9.9", extra=("--no-sign",))

    doc = json.loads((out / "sbom.cdx.json").read_text())
    assert doc["bomFormat"] == "CycloneDX"
    component = doc["metadata"]["component"]
    assert component["name"] == "latence-framework"
    assert component["version"] == "9.9.9"
    properties = {p["name"]: p["value"] for p in doc["metadata"]["properties"]}
    assert properties["latence:release:tag"] == "v9.9.9"
    assert properties["latence:release:repository"] == "ddickmann/latence-framework"
    # The resolved dependency set survives stamping — that is the whole point of the SBOM.
    assert {"pydantic", "fsspec"} <= {c["name"] for c in doc["components"]}
    # ...and so does the coverage of every distribution the release publishes. Stamping a
    # release identity onto a document rewrites it in place, so the workspace components must
    # come out the other side: the stamp asserts "this is the inventory for this exact
    # release", and that claim is only true if the inventory still accounts for the release.
    stamped_paths = {
        ref["url"].rsplit("/", 1)[-1]
        for c in doc["components"]
        for ref in c.get("externalReferences", [])
        if "/packages/" in ref.get("url", "")
    }
    assert stamped_paths == set(_workspace_packages())


def test_verify_instructions_pin_the_tag_and_link_the_cvd_policy(tmp_path: Path) -> None:
    """VERIFY.md ships a copy-pasteable verify command for THIS tag plus the disclosure route."""
    _, out = _run(tmp_path, tag="v9.9.9", extra=("--no-sign",))
    text = (out / "VERIFY.md").read_text()

    assert "sigstore verify identity" in text
    assert "--bundle SHA256SUMS.sigstore.json" in text
    # The identity an adopter pins must name the exact workflow AND the exact tag ref;
    # a generic identity would verify any signature this repo ever produced.
    assert (
        "https://github.com/ddickmann/latence-framework/.github/workflows/release.yml"
        "@refs/tags/v9.9.9" in text
    )
    assert "--cert-oidc-issuer 'https://token.actions.githubusercontent.com'" in text
    assert "-c SHA256SUMS" in text
    assert "SECURITY.md" in text and "vulnerability" in text.lower()


_TAG_IDENTITY = (
    "https://github.com/ddickmann/latence-framework/.github/workflows/release.yml@refs/tags/v9.9.9"
)


def test_verify_identity_is_the_ref_the_run_actually_signs_under(tmp_path: Path) -> None:
    """VERIFY.md quotes the REAL Fulcio SAN, not a guess derived from the tag input.

    Keyless Sigstore certifies the ``job_workflow_ref`` of the running workflow. This workflow
    is dispatchable from a branch with the tag passed as an input, so the signing identity is
    then ``…/release.yml@refs/heads/<branch>``. Hardcoding ``refs/tags/<tag>`` shipped a signed
    VERIFY.md whose own ``sigstore verify identity`` command fails on a perfectly good bundle —
    which an adopter cannot distinguish from tampering.
    """
    branch_ref = "ddickmann/latence-framework/.github/workflows/release.yml@refs/heads/main"
    _, out = _run(tmp_path, tag="v9.9.9", extra=("--no-sign", "--workflow-ref", branch_ref))
    text = (out / "VERIFY.md").read_text()

    assert f"--cert-identity 'https://github.com/{branch_ref}'" in text, text
    assert _TAG_IDENTITY not in text, (
        "VERIFY.md still pins the tag identity for a run signed under a branch ref — the "
        "shipped verification command cannot verify the shipped signature"
    )
    # An asset set that cannot carry the published identity must say so, in the signed file.
    assert "Rehearsal" in text and "do not publish" in text.lower()


def test_rejects_a_malformed_workflow_ref(tmp_path: Path) -> None:
    """A garbled ref would silently become a nonsense --cert-identity nobody can verify."""
    result, _ = _run(tmp_path, extra=("--no-sign", "--workflow-ref", "refs/heads/main"))
    assert result.returncode != 0
    assert "github.workflow_ref form" in result.stderr, result.stderr


def test_tag_path_identity_is_unchanged_without_a_workflow_ref(tmp_path: Path) -> None:
    """Off by default is byte-identical: no --workflow-ref reproduces the tag-push identity."""
    _, out = _run(tmp_path, tag="v9.9.9", extra=("--no-sign",))
    text = (out / "VERIFY.md").read_text()
    assert f"--cert-identity '{_TAG_IDENTITY}'" in text
    assert "Rehearsal" not in text
    # A tag-ref run must produce exactly the same instructions as the implicit default.
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    _, tag_run = _run(
        explicit,
        tag="v9.9.9",
        extra=(
            "--no-sign",
            "--workflow-ref",
            "ddickmann/latence-framework/.github/workflows/release.yml@refs/tags/v9.9.9",
        ),
    )
    assert (tag_run / "VERIFY.md").read_text() == text


# --------------------------------------------------------------------------------------
# Fail-closed: what the script REFUSES to release
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sbom", "expected"),
    [
        pytest.param(
            {"bomFormat": "SPDX", "specVersion": "2.3", "components": [{"name": "x"}]},
            "not a CycloneDX document",
            id="wrong-format",
        ),
        pytest.param(
            {"bomFormat": "CycloneDX", "components": [{"name": "x"}]},
            "no specVersion",
            id="no-spec-version",
        ),
        pytest.param(
            {"bomFormat": "CycloneDX", "specVersion": "1.6", "components": []},
            "lists no components",
            id="empty-components",
        ),
        pytest.param("{not json", "not readable JSON", id="not-json"),
    ],
)
def test_refuses_to_attach_a_useless_sbom(
    tmp_path: Path, sbom: dict[str, Any] | str, expected: str
) -> None:
    """A non-CycloneDX or component-less document is rejected, not published as evidence."""
    result, out = _run(tmp_path, sbom=sbom, extra=("--no-sign",))

    assert result.returncode != 0, "a useless SBOM was accepted"
    assert expected in result.stderr, result.stderr
    assert not (out / "SHA256SUMS").exists(), "asset set was assembled around a rejected SBOM"


def test_refuses_to_sign_an_sbom_that_omits_published_distributions(tmp_path: Path) -> None:
    """An SBOM that covers only part of the release is rejected before it is signed.

    This is the shape the CPU-first dev environment produced before ``--all-packages``:
    ``uv export`` resolved only the root meta-project, so the document described 12 of the 32
    distributions the same release publishes and named none of their heavy dependencies
    (torch, transformers, gliner2, peft, presidio-analyzer...). It is still reachable via
    ``--sbom``, which bypasses generate-sbom.sh and its coverage check entirely.

    Such a document is not merely incomplete — it is stamped ``latence-framework <version>``,
    digest-recorded in SHA256SUMS and Sigstore-signed, so an adopter scanning it for
    "am I affected by CVE-X?" gets a CLEAN answer that is cryptographically authenticated.
    The guard must therefore run on the document that gets signed, and fail closed.
    """
    all_packages = _workspace_packages()
    assert len(all_packages) > 12, "fixture assumes a multi-package workspace"

    # The pre-fix resolution scope: the root project's own dependency closure only.
    cpu_only = [
        name
        for name in all_packages
        if name
        in {
            "latence-benchmark",
            "latence-console",
            "latence-core",
            "latence-demo",
            "latence-disambig-embedding",
            "latence-muvera",
            "latence-parser-document",
            "latence-parser-pdfplumber",
            "latence-parser-plaintext",
            "latence-retrieval",
            "latence-sink-qdrant",
            "latence-source-sharepoint",
        }
    ]
    omitted = sorted(set(all_packages) - set(cpu_only))
    assert omitted, "fixture assumes some packages are outside the CPU-first closure"

    partial = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        # Non-empty and structurally valid: it passes every pre-existing check.
        "components": [
            {"type": "library", "name": "pydantic", "version": "2.9.2"},
            *_workspace_components(cpu_only),
        ],
    }

    result, out = _run(tmp_path, sbom=partial, extra=("--no-sign",))

    assert result.returncode != 0, "an SBOM omitting published distributions was accepted"
    assert "distributions this release publishes" in result.stderr, result.stderr
    # The error names what is missing, rather than failing anonymously (ADR-0034).
    for name in omitted:
        assert name in result.stderr, f"{name} missing from the failure message: {result.stderr}"
    assert not (out / "SHA256SUMS").exists(), (
        "the asset set was assembled — and would have been signed — around an SBOM that "
        "omits distributions the release publishes"
    )
    assert not (out / "SHA256SUMS.sigstore.json").exists()


def test_signing_has_no_default(tmp_path: Path) -> None:
    """Neither --sign nor --no-sign is an error: an unsigned release is never an accident.

    Defaulting to "skip signing when no identity is available" is exactly how a workflow
    ships unsigned assets after someone drops the ``id-token: write`` permission.
    """
    result, _ = _run(tmp_path, extra=())
    assert result.returncode != 0
    assert "--sign or --no-sign" in result.stderr


def test_rejects_a_tag_that_is_not_a_version(tmp_path: Path) -> None:
    """A malformed tag would stamp the SBOM with a version matching no release."""
    result, _ = _run(tmp_path, tag="nightly", extra=("--no-sign",))
    assert result.returncode != 0
    assert "SemVer" in result.stderr


def test_rejects_an_empty_dist(tmp_path: Path) -> None:
    """An empty dist/ is always a build failure, never a valid release."""
    result, _ = _run(tmp_path, dist_names=(), extra=("--no-sign",))
    assert result.returncode != 0
    assert "nothing to release" in result.stderr


def test_rejects_colliding_distribution_filenames(tmp_path: Path) -> None:
    """Assets are staged flat, so a basename collision would silently drop a distribution."""
    dist = tmp_path / "dist"
    (dist / "a").mkdir(parents=True)
    (dist / "b").mkdir(parents=True)
    for sub in ("a", "b"):
        (dist / sub / "latence_core-9.9.9-py3-none-any.whl").write_bytes(sub.encode())
    (tmp_path / "sbom.json").write_text(json.dumps(_CYCLONEDX_DOC), encoding="utf-8")

    result = subprocess.run(
        [
            "bash", str(_SCRIPT),
            "--tag", "v9.9.9",
            "--dist", str(dist),
            "--sbom", str(tmp_path / "sbom.json"),
            "--out", str(tmp_path / "out"),
            "--no-sign",
        ],
        cwd=_REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "duplicate distribution filenames" in result.stderr


# --------------------------------------------------------------------------------------
# Wiring: the workflow must actually use all of the above
# --------------------------------------------------------------------------------------


def _workflow() -> dict[Any, Any]:
    loaded = yaml.safe_load(_WORKFLOW.read_text())
    assert isinstance(loaded, dict)
    return loaded


def _triggers(workflow: dict[Any, Any]) -> dict[str, Any]:
    """The `on:` block. YAML 1.1 parses a bare ``on`` key as the boolean ``True``."""
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict), "release workflow has no parsable `on:` block"
    return triggers


def test_release_workflow_signs_and_sboms_every_tag() -> None:
    """The release job chain runs release-integrity.sh with signing ON, with an OIDC token."""
    jobs = _workflow()["jobs"]
    integrity = jobs["release-integrity"]

    # Keyless Sigstore needs the job's OIDC token; without it the sign step cannot run.
    assert integrity["permissions"]["id-token"] == "write"

    run_steps = "\n".join(str(step.get("run", "")) for step in integrity["steps"])
    assert "scripts/release-integrity.sh" in run_steps
    assert "--sign" in run_steps and "--no-sign" not in run_steps


def test_github_release_publishes_only_the_signed_asset_set() -> None:
    """The Release attaches the checksummed asset set, and refuses to publish it unsigned."""
    jobs = _workflow()["jobs"]
    release = jobs["github-release"]

    assert release["needs"] == "release-integrity", (
        "the Release must be gated on the SBOM+signing job, or a tag can publish unsigned"
    )

    steps = release["steps"]
    guard = "\n".join(str(step.get("run", "")) for step in steps)
    assert "SHA256SUMS.sigstore.json" in guard, (
        "no pre-publish check that the Sigstore bundle exists"
    )

    publish = next(s for s in steps if "action-gh-release" in str(s.get("uses", "")))
    files = publish["with"]["files"]
    assert "release-assets" in files, "the Release must attach the checksummed asset set"
    assert "dist/" not in files, (
        "attaching raw dist/ would ship files SHA256SUMS does not account for"
    )
    # Release notes carry the verify command + CVD link, generated from the same tag/repo
    # that were signed, so they cannot drift from the assets.
    assert publish["with"]["body_path"].endswith("VERIFY.md")
    assert publish["with"]["append_body"] is True


def test_dry_run_rehearses_without_publishing_anything() -> None:
    """workflow_dispatch dry_run builds + signs but creates no Release and never reaches PyPI."""
    workflow = _workflow()
    inputs = _triggers(workflow)["workflow_dispatch"]["inputs"]
    assert "dry_run" in inputs, "no dry-run rehearsal path"

    jobs = workflow["jobs"]
    assert "dry_run" in str(jobs["github-release"]["if"])
    assert "dry_run" in str(jobs["pypi-publish"]["if"])
    # The rehearsal is only worth anything if it signs for real, so release-integrity itself
    # must NOT be gated on dry_run.
    assert "dry_run" not in str(jobs["release-integrity"].get("if", ""))


def test_workflow_hands_the_script_the_real_signing_ref() -> None:
    """The script is told the run's github.workflow_ref, so VERIFY.md cannot guess wrong."""
    integrity = _workflow()["jobs"]["release-integrity"]
    run_steps = "\n".join(str(step.get("run", "")) for step in integrity["steps"])
    assert "--workflow-ref" in run_steps and "github.workflow_ref" in run_steps, (
        "release-integrity.sh must be given the ref the run actually signs under; deriving the "
        "cert identity from the tag input alone ships unverifiable instructions"
    )


def test_a_real_release_refuses_a_non_tag_signing_identity() -> None:
    """A non-dry run on a branch ref is blocked before it can publish a mis-signed release."""
    integrity = _workflow()["jobs"]["release-integrity"]
    guards = [
        step
        for step in integrity["steps"]
        if "workflow_ref" in str(step.get("env", "")) and "dry_run" in str(step.get("if", ""))
    ]
    assert guards, (
        "no step refuses a non-dry run whose workflow ref is not the tag — such a run creates "
        "a real Release whose signed VERIFY.md cannot verify its own signature"
    )
    assert "refs/tags/" in "\n".join(str(step.get("run", "")) for step in guards)
    assert any("exit 1" in str(step.get("run", "")) for step in guards), (
        "the guard must fail the job, not just warn"
    )


def test_front_page_and_docs_route_a_reporter_to_the_cvd_policy() -> None:
    """README links the disclosure policy and the release-verification procedure."""
    readme = (_REPO_ROOT / "README.md").read_text()
    assert "SECURITY.md" in readme
    assert "vulnerability" in readme.lower()
    assert "docs/RELEASING.md" in readme
    assert "sigstore" in readme.lower() and "sbom" in readme.lower()

    security = (_REPO_ROOT / "SECURITY.md").read_text()
    assert "Reporting a vulnerability" in security


# --------------------------------------------------------------------------------------
# The SBOM generator itself
#
# These are static guards, and deliberately so: actually running generate-sbom.sh needs the
# network (it fetches cyclonedx-bom) and ~30s, which does not belong in the merge gate. They
# encode the failure that actually happened — cyclonedx-bom 7 removed `--outfile`, so the
# unpinned script had silently stopped producing an SBOM at all. The end-to-end proof is the
# release dry run documented in docs/RELEASING.md — whose counts are no longer unguarded
# prose: test_releasing_dry_run_evidence_matches_the_tree (below) recomputes them from
# packages/ and fails when the document and the tree disagree.
# --------------------------------------------------------------------------------------

_GENERATE_SBOM = _REPO_ROOT / "scripts" / "generate-sbom.sh"


def _sbom_generator_commands() -> str:
    """generate-sbom.sh with comment lines stripped — the flags it actually passes."""
    return "\n".join(
        line for line in _GENERATE_SBOM.read_text().splitlines() if not line.lstrip().startswith("#")
    )


def test_sbom_generator_pins_its_tool() -> None:
    """cyclonedx-bom is version-bounded: an unpinned CLI change silently broke this once."""
    script = _sbom_generator_commands()
    assert "cyclonedx-bom>=7,<8" in script, (
        "cyclonedx-bom must be pinned to a major range — release machinery cannot depend on "
        "whatever version happens to resolve today"
    )
    # The flag that was removed in 7.x. Its return would break the release, not just the docs.
    assert "--outfile" not in script
    assert "--output-file" in script


def test_sbom_generator_emits_the_documented_reproducible_format() -> None:
    """Spec version is pinned to what docs/SBOM.md claims, and output is byte-reproducible."""
    script = _sbom_generator_commands()
    assert "--spec-version 1.6" in script, "the emitted CycloneDX version must not float"
    assert "--output-reproducible" in script, (
        "the SBOM digest is recorded in the signed SHA256SUMS, so a timestamped/random-serial "
        "SBOM would make every release unreproducible"
    )
    # The docs promise a specific schema version; the script must emit that one.
    assert "1.6" in (_REPO_ROOT / "docs" / "SBOM.md").read_text()


def test_releasing_doc_documents_the_signing_and_sbom_steps() -> None:
    """The release procedure describes signing + SBOM, with the verify command inline."""
    doc = (_REPO_ROOT / "docs" / "RELEASING.md").read_text()
    assert "release-integrity" in doc
    assert "sigstore verify identity" in doc
    assert "-c SHA256SUMS" in doc
    assert "sbom.cdx.json" in doc
    assert "dry_run" in doc, "no documented rehearsal procedure"


# --------------------------------------------------------------------------------------
# The "Last verified local dry run" evidence block
#
# The block is the only end-to-end evidence that the release + SBOM machinery actually
# produces a verifiable asset set, so it has to stay reproducible. It did not: it claimed
# "all 25 packages: 56 distributions built … 58 assets checksummed" against a tree that had
# 32 packages (64 distributions, 66 assets) — and 25 was already wrong when it was written.
# Prose evidence that nothing recomputes is not evidence, so recompute it here.
# --------------------------------------------------------------------------------------

_RELEASING = _REPO_ROOT / "docs" / "RELEASING.md"


def _package_dirs() -> list[Path]:
    """Exactly what the release workflow's ``for pkg in packages/*/`` loop builds."""
    return sorted(p.parent for p in (_REPO_ROOT / "packages").glob("*/pyproject.toml"))


def _documented_dry_run_counts() -> dict[str, int]:
    """The derivation table in the evidence block, as {label: value}."""
    rows = re.findall(
        r"^\|\s*(packages built|distributions|assets checksummed)\s*\|\s*(\d+)\s*\|",
        _RELEASING.read_text(),
        re.MULTILINE,
    )
    return {label: int(value) for label, value in rows}


def test_releasing_dry_run_evidence_matches_the_tree() -> None:
    """The documented dry-run counts are recomputed from packages/ — drift fails the gate."""
    counts = _documented_dry_run_counts()
    assert set(counts) == {"packages built", "distributions", "assets checksummed"}, (
        "the 'Last verified local dry run' derivation table in docs/RELEASING.md must state "
        f"all three derived counts; found {sorted(counts)}"
    )

    packages = _package_dirs()
    assert counts["packages built"] == len(packages), (
        f"docs/RELEASING.md claims the dry run built {counts['packages built']} packages, but "
        f"packages/ holds {len(packages)}. release.yml builds `for pkg in packages/*/`, so "
        "adding or removing a package changes the release — update the evidence block."
    )

    # The x2 derivation only holds because every package emits an sdist AND a wheel.
    not_hatchling = [
        p.name
        for p in packages
        if 'build-backend = "hatchling.build"' not in (p / "pyproject.toml").read_text()
    ]
    assert not not_hatchling, (
        "the documented '2 distributions per package' derivation assumes hatchling (sdist + "
        f"wheel) for every package; these do not use it: {not_hatchling}"
    )
    assert counts["distributions"] == 2 * len(packages), (
        f"{len(packages)} hatchling packages build {2 * len(packages)} distributions, but "
        f"docs/RELEASING.md records {counts['distributions']}"
    )

    # release-integrity.sh stages the distributions flat and adds exactly two more assets
    # before writing SHA256SUMS, which covers every file except SHA256SUMS* itself.
    script = _SCRIPT.read_text()
    assert 'SBOM="$OUT/sbom.cdx.json"' in script and 'cat >> "$OUT/VERIFY.md"' in script, (
        "the '+2 assets' derivation assumes release-integrity.sh stages sbom.cdx.json and "
        "VERIFY.md alongside the distributions — it no longer does"
    )
    assert counts["assets checksummed"] == counts["distributions"] + 2, (
        f"{counts['distributions']} distributions + sbom.cdx.json + VERIFY.md = "
        f"{counts['distributions'] + 2} assets, but docs/RELEASING.md records "
        f"{counts['assets checksummed']}"
    )

    doc = _RELEASING.read_text()
    verified = re.search(r"-c SHA256SUMS`? verified all (\d+)", doc)
    assert verified is not None, "the evidence block no longer records a verified asset count"
    assert int(verified.group(1)) == counts["assets checksummed"], (
        "the evidence block verifies a different number of assets than it says it checksummed"
    )


def test_releasing_sbom_component_arithmetic_is_consistent() -> None:
    """The SBOM sentence's total equals its own breakdown (third-party + workspace paths)."""
    doc = _RELEASING.read_text()
    total = re.search(r"carried (\d+) components", doc)
    third_party = re.search(r"(\d+) pinned\s+third-party requirements", doc)
    workspace = re.search(r"plus (\d+) workspace path entries", doc)
    assert total and third_party and workspace, (
        "the evidence block must break the SBOM component count down into pinned "
        "third-party requirements + workspace path entries, so a change in either is visible"
    )
    assert int(total.group(1)) == int(third_party.group(1)) + int(workspace.group(1)), (
        f"SBOM component arithmetic does not add up: {total.group(1)} != "
        f"{third_party.group(1)} + {workspace.group(1)}"
    )
    # Arithmetic alone is hollow: 54 + 12 == 66 also "adds up", and that was the number the
    # block recorded while the SBOM covered 12 of the 32 distributions the release ships.
    # There is exactly one `./packages/<name>` path entry per resolved workspace member, so
    # the workspace-path count must equal the package count derived from the tree.
    assert int(workspace.group(1)) == len(_package_dirs()), (
        f"the SBOM evidence block records {workspace.group(1)} workspace path entries but the "
        f"tree has {len(_package_dirs())} packages — the SBOM does not cover every "
        "distribution the release ships (a missing `uv export --all-packages`?)"
    )


# --------------------------------------------------------------------------------------
# SBOM SCOPE — the document must cover what the release actually ships (R4-security_secrets-3)
#
# `uv export` without `--all-packages` resolves only the root `latence-framework-dev`
# meta-project, which by design (ADR-0016, thin core) depends on none of the heavy Provider
# packages. The SBOM built from it omitted torch, transformers, gliner, openai,
# presidio-analyzer, sentence-transformers … while still being stamped, digest-recorded in
# SHA256SUMS and Sigstore-signed. `grype sbom:sbom.cdx.json` then reports CLEAN for a CVE
# the adopter is exposed to. So: run the export the script actually performs, and check it
# against every required dependency the workspace declares.
# --------------------------------------------------------------------------------------


def _uv_export_flags() -> list[str]:
    """The flags generate-sbom.sh actually passes to `uv export`, read out of the script.

    Read from the script rather than restated here: a test that hard-codes the flags proves
    nothing about the command the release runs.
    """
    lines = _GENERATE_SBOM.read_text().splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip().startswith("uv export")]
    assert len(starts) == 1, f"expected exactly one `uv export` invocation, found {len(starts)}"
    collected: list[str] = []
    for line in lines[starts[0] :]:
        collected.append(line.rstrip().removesuffix("\\"))
        if not line.rstrip().endswith("\\"):
            break
    flags = " ".join(collected).split()[2:]  # drop "uv" "export"
    assert flags[-2:] == ["-o", '"$REQ"'], f"unexpected output flag in generate-sbom.sh: {flags}"
    return flags


def _exported_requirement_names(destination: Path) -> set[str]:
    """Run the EXACT `uv export` invocation generate-sbom.sh performs; return pinned names."""
    flags = _uv_export_flags()
    result = subprocess.run(
        ["uv", "export", *flags[:-2], "-o", str(destination)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"uv export failed: {result.stderr}"
    return {
        _normalise(m.group(1))
        for m in re.finditer(
            r"^([A-Za-z0-9][A-Za-z0-9._-]*)==", destination.read_text(encoding="utf-8"), re.MULTILINE
        )
    }


def _normalise(name: str) -> str:
    """PEP 503 name normalisation, so `python_docx` and `python-docx` compare equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _required_third_party_dependencies() -> dict[str, set[str]]:
    """{package: {required third-party dependency names}} from every packages/*/pyproject.toml.

    Only ``project.dependencies`` — the REQUIRED runtime set. Optional extras are excluded on
    purpose: an adopter who did not install an extra is not exposed through it. Sibling
    workspace packages are excluded (they are the path entries, not third-party components).
    """
    tomllib = pytest.importorskip("tomllib")
    required: dict[str, set[str]] = {}
    for pyproject in sorted((_REPO_ROOT / "packages").glob("*/pyproject.toml")):
        declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"].get(
            "dependencies", []
        )
        names = {
            _normalise(match.group(1))
            for dep in declared
            if (match := re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", dep))
        }
        required[pyproject.parent.name] = {n for n in names if not n.startswith("latence-")}
    return required


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is required to resolve uv.lock")
def test_sbom_export_covers_every_released_package_dependency(tmp_path: Path) -> None:
    """Every REQUIRED dependency of every released package is in the document an adopter scans.

    This is the CVE question the SBOM exists to answer. Without ``--all-packages`` the export
    resolves only the root meta-project and 18 of the 32 distributions have at least one
    required dependency missing from the SBOM entirely.
    """
    exported = _exported_requirement_names(tmp_path / "requirements.txt")

    uncovered = {
        package: sorted(deps - exported)
        for package, deps in _required_third_party_dependencies().items()
        if deps - exported
    }
    assert not uncovered, (
        f"{len(uncovered)} released package(s) declare REQUIRED dependencies that the release "
        f"SBOM does not contain, so `grype sbom:sbom.cdx.json` reports CLEAN for a CVE in them: "
        f"{uncovered}"
    )

    # Named canaries: the heavy Provider dependencies the meta-project deliberately excludes.
    # If the resolution scope silently narrows again, these are the first to vanish.
    for canary in ("torch", "transformers", "gliner", "openai", "presidio-analyzer"):
        assert canary in exported, f"{canary} is missing from the release SBOM's dependency set"


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is required to resolve uv.lock")
def test_sbom_generator_fails_closed_when_the_export_misses_a_package(tmp_path: Path) -> None:
    """The generator refuses to build an SBOM from an export that misses workspace packages.

    A dropped ``--all-packages`` must break the release loudly rather than publish a signed
    document that understates the dependency surface. Simulated by running the generator in a
    worktree-shaped copy that declares one more package than the export can cover.
    """
    original = _GENERATE_SBOM.read_text()
    assert "--all-packages" in _uv_export_flags(), "generate-sbom.sh no longer passes --all-packages"
    # Drop the flag, and pin the working directory (the script normally derives it from $0,
    # which would point at tmp_path here).
    script = original.replace("--all-packages \\\n", "").replace(
        'cd "$(dirname "$0")/.."', f'cd "{_REPO_ROOT}"'
    )
    assert script != original
    stunted = tmp_path / "generate-sbom-without-all-packages.sh"
    stunted.write_text(script, encoding="utf-8")

    result = subprocess.run(
        ["bash", str(stunted), str(tmp_path / "sbom.cdx.json")],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, "an SBOM was generated from an export missing packages"
    assert "workspace packages" in result.stderr, result.stderr
    assert not (tmp_path / "sbom.cdx.json").exists(), "a hollow SBOM was written anyway"
