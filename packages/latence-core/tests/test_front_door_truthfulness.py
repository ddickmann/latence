"""The front door may not lie (issue #189 — flip public).

The moment the repository goes public the README, the glossary and the docs site stop
being internal notes and become the product's only sales pitch. This module is the
machine-checked half of "nothing published may be false on day one": every claim these
tests cover is one an outsider can falsify in under a minute, so it is checked here
rather than trusted to a reviewer's eye.

Four independent falsifiers, each of which failed on the pre-#189 tree:

* **Shipped Runner adapters.** The README promised "thin adapters run the same Stages on
  Databricks or Airflow" while only ``latence-runner-airflow`` exists. Any *substrate*
  named in the front-door prose as something the framework runs on must correspond to a
  real ``packages/latence-runner-*`` distribution, unless the sentence explicitly marks it
  as not-shipped.
* **The ADR count.** The README quoted "32 architecture decisions" against 52 files on
  disk. Any count the README states about ``docs/adr/`` must equal the tree.
* **Dangling cross-references.** ``docs/retrieval-tooling.md`` and ADR-0048 pointed at a
  ``docs/INTEROP.md`` that was never written. No markdown link and no backticked
  ``docs/...md`` reference in the front-door set may point at a missing file.
* **The install story.** The README advertised a meta-package that is not publishable, and
  every ``pip install`` line must name a distribution that actually exists under
  ``packages/``.

These assert observable repository facts, never a flag that records intent.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

# tests/  ->  latence-core/  ->  packages/  ->  repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PACKAGES = _REPO_ROOT / "packages"
_README = _REPO_ROOT / "README.md"
_ADR_DIR = _REPO_ROOT / "docs" / "adr"

# The pages an outsider reads before they read any code. Broken claims here are the
# expensive ones, so this is the set the tests police.
_FRONT_DOOR = ("README.md", "CONTEXT.md", "CONTRIBUTING.md", "SECURITY.md")

_MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_BACKTICKED_DOC = re.compile(r"`(docs/[A-Za-z0-9_\-/.]+\.md)`")


def _tracked_paths() -> set[Path] | None:
    """Paths git actually tracks, or ``None`` outside a git checkout.

    Publication follows the *index*, not the working tree: a maintainer-local note sitting
    untracked in ``docs/`` never reaches the public repo or the deployed site, so it is not
    a front-door document and its internal references are not this gate's business.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "ls-files", "-z"],
            capture_output=True,
            check=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return {_REPO_ROOT / p for p in out.decode().split("\0") if p}


def _front_door_docs() -> list[Path]:
    """README + the root governance pages + every tracked page the docs site publishes."""
    pages = [_REPO_ROOT / name for name in _FRONT_DOOR]
    pages += sorted((_REPO_ROOT / "docs").rglob("*.md"))
    tracked = _tracked_paths()
    out = [p for p in pages if p.is_file() and (tracked is None or p in tracked)]
    assert out, "no front-door documents discovered"
    return out


def _shipped_runner_substrates() -> set[str]:
    """Substrate names backed by a real ``packages/latence-runner-*`` distribution."""
    return {
        pkg.name.removeprefix("latence-runner-").lower()
        for pkg in _PACKAGES.iterdir()
        if pkg.name.startswith("latence-runner-") and (pkg / "pyproject.toml").is_file()
    }


def test_readme_claims_no_runner_adapter_that_does_not_exist() -> None:
    """Every substrate the README says the Stages *run on* must be a shipped package.

    A substrate may still be *mentioned* — the Runner seam's whole point is that a third
    one is writable — but the sentence must say so. The falsifier is a sentence that
    presents a substrate as available alongside one that genuinely is.
    """
    shipped = _shipped_runner_substrates()
    assert "airflow" in shipped, (
        "expected a shipped Airflow Runner adapter under packages/latence-runner-airflow"
    )

    # Substrates a reader could plausibly try to `pip install` a Runner for.
    candidates = ("databricks", "dagster", "prefect", "temporal", "ray", "kubeflow")
    # Wording that turns a mention into an honest not-yet.
    hedges = (
        "not shipped",
        "not** shipped",
        "seam-compatible",
        "follow-on",
        "follow-ons",
        "would be",
        "could be",
        "writable",
    )

    problems: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", _README.read_text()):
        low = sentence.lower()
        if "runner" not in low and "adapter" not in low:
            continue
        for name in candidates:
            if name in low and name not in shipped and not any(h in low for h in hedges):
                problems.append(f"{name!r} presented as a shipped Runner: {sentence.strip()!r}")

    assert not problems, (
        "README claims Runner adapters that do not exist under packages/:\n"
        + "\n".join(problems)
        + f"\n(shipped Runner substrates: {sorted(shipped) or 'none'})"
    )


def test_readme_adr_count_matches_the_tree() -> None:
    """Any "N architecture decisions" the README states must equal the files on disk."""
    actual = len([p for p in _ADR_DIR.glob("[0-9][0-9][0-9][0-9]-*.md")])
    assert actual > 0, "no ADR files discovered under docs/adr/"

    text = _README.read_text()
    claims = [int(m) for m in re.findall(r"\((\d+)\s+architecture decisions\)", text)]
    claims += [int(m) for m in re.findall(r"the\s+(\d+)\s+architecture decisions", text)]
    assert claims, "README no longer states an ADR count — keep the claim, keep it true"

    wrong = [c for c in claims if c != actual]
    assert not wrong, (
        f"README claims {wrong} architecture decisions; docs/adr/ holds {actual}. "
        "Update the README (or the ADRs) so the front door matches the tree."
    )


def test_no_dangling_cross_references_in_the_front_door() -> None:
    """No relative link or backticked ``docs/...md`` reference points at a missing file."""
    problems: list[str] = []
    for page in _front_door_docs():
        for lineno, line in enumerate(page.read_text().splitlines(), 1):
            targets = [
                t
                for t in _MD_LINK.findall(line)
                if not t.startswith(("http://", "https://", "mailto:", "#"))
            ]
            for target in targets:
                resolved = (page.parent / target.split("#", 1)[0]).resolve()
                if not resolved.exists():
                    problems.append(f"{page.relative_to(_REPO_ROOT)}:{lineno}: link -> {target}")
            for doc_ref in _BACKTICKED_DOC.findall(line):
                if not (_REPO_ROOT / doc_ref).exists():
                    problems.append(f"{page.relative_to(_REPO_ROOT)}:{lineno}: ref -> {doc_ref}")

    assert not problems, "dangling cross-references in published docs:\n" + "\n".join(problems)


def _readme_shell_blocks() -> str:
    """The README's fenced ``bash`` blocks — the commands a reader copy-pastes.

    Only fenced code counts as an instruction. Prose *about* installing (including the
    sentence that tells the reader a ``latence`` meta-package does not exist) is not a
    command and must not be read as one.
    """
    lines = _README.read_text().splitlines()
    out: list[str] = []
    inside = False
    for line in lines:
        if line.startswith("```"):
            inside = line.strip() != "```"
            continue
        if inside:
            out.append(line)
    return "\n".join(out)


def test_readme_install_lines_name_real_distributions() -> None:
    """Every ``pip install latence-*`` command in the README maps to a real package."""
    distributions = {
        pkg.name for pkg in _PACKAGES.iterdir() if (pkg / "pyproject.toml").is_file()
    }
    named = set(re.findall(r"pip install (?:-e )?'?(latence[a-z0-9-]*)", _readme_shell_blocks()))
    assert named, "README no longer documents an install — the front door needs one"

    missing = sorted(n for n in named if n not in distributions)
    assert not missing, (
        f"README tells the reader to install {missing}, which does not exist under packages/. "
        "There is no publishable `latence` meta-package; name the real distributions."
    )


def test_readme_does_not_promise_a_meta_package() -> None:
    """The pre-#189 README promised a meta-package that pins the demo set; none exists.

    Guarded as a distinct fact because the sentence did not contain a ``pip install`` line,
    so the install-line test above cannot see it.
    """
    text = _README.read_text()
    assert not re.search(r"[Tt]he meta-package pins", text), (
        "README promises a meta-package; only per-Provider distributions are publishable "
        "(ADR-0016). Point the reader at latence-demo instead."
    )
