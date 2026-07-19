"""The front door may not lie (issue #189 — flip public).

The moment the repository goes public the README, the glossary and the docs site stop
being internal notes and become the product's only sales pitch. This module is the
machine-checked half of "nothing published may be false on day one": every claim these
tests cover is one an outsider can falsify in under a minute, so it is checked here
rather than trusted to a reviewer's eye.

Five independent falsifiers, each of which failed on the tree that introduced it:

* **Shipped Runner adapters.** The README promised "thin adapters run the same Stages on
  Databricks or Airflow" while only ``latence-runner-airflow`` exists. Any *substrate*
  named in the front-door prose as something the framework runs on must correspond to a
  real ``packages/latence-runner-*`` distribution, unless the sentence explicitly marks it
  as not-shipped.
* **The ADR count.** The README quoted "32 architecture decisions" against 52 files on
  disk. Any count *any* front-door page states about ``docs/adr/`` must equal the tree —
  the first version of this guard read ``README.md`` alone, which let the docs-site home
  page and the Concepts index sit at "44 architecture decisions" while the README, the
  Decision Log and the go-live runbook all said 52.
* **Dangling cross-references.** ``docs/retrieval-tooling.md`` and ADR-0048 pointed at a
  ``docs/INTEROP.md`` that was never written. No markdown link and no backticked
  ``docs/...md`` reference in the front-door set may point at a missing file.
* **The install story.** The README advertised a meta-package that is not publishable, and
  every ``pip install`` line must name a distribution that actually exists under
  ``packages/``.
* **The licence summary.** The README's index called ``THIRD-PARTY-LICENSES.md`` "(all
  permissive)" *after* the release-fleet stream added ~40 proprietary NVIDIA CUDA wheels and
  the Gemma opt-in to that very file. No published page may summarise the dependency tree as
  uniformly permissive while the disclosure file says otherwise.

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


# How the published pages actually phrase a size claim about the Decision Log. A count is a
# count wherever it appears, so these run over every front-door page rather than the README.
_ADR_COUNT_CLAIM = re.compile(
    r"(?<![\d.])(\d{1,3})\s+(?:architecture decisions|architecture decision records|ADRs\b)"
    r"|holds\s+(\d{1,3})\s+records"
)


def _adr_count_claims(text: str) -> list[tuple[int, int]]:
    """``(lineno, claimed_count)`` for every ADR-count claim in ``text``."""
    out: list[tuple[int, int]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for match in _ADR_COUNT_CLAIM.finditer(line):
            out.append((lineno, int(match.group(1) or match.group(2))))
    return out


def test_adr_count_matches_the_tree_on_every_front_door_page() -> None:
    """Any "N architecture decisions" a *published* page states must equal the files on disk.

    Not just the README: the docs-site ``Home`` page is the first thing an evaluator sees
    after the go-live runbook's deploy step, and it drifted 8 records behind the log it links
    to while this test was green.
    """
    actual = len([p for p in _ADR_DIR.glob("[0-9][0-9][0-9][0-9]-*.md")])
    assert actual > 0, "no ADR files discovered under docs/adr/"

    problems: list[str] = []
    total_claims = 0
    for page in _front_door_docs():
        for lineno, claimed in _adr_count_claims(page.read_text()):
            total_claims += 1
            if claimed != actual:
                problems.append(f"{page.relative_to(_REPO_ROOT)}:{lineno}: claims {claimed}")

    assert not problems, (
        f"published pages state an ADR count that docs/adr/ contradicts (it holds {actual}):\n"
        + "\n".join(problems)
        + "\nUpdate the pages (or the ADRs) so every front door matches the tree."
    )
    assert total_claims, "the docs no longer state an ADR count — keep the claim, keep it true"

    # The README specifically must keep stating it: it is the page most readers never leave.
    assert _adr_count_claims(_README.read_text()), (
        "README no longer states an ADR count — keep the claim, keep it true"
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


# --------------------------------------------------------------------------------------
# Section anchors (R4-docs_truth-5)
# --------------------------------------------------------------------------------------
#
# The guard above resolves `target.split("#", 1)[0]` — it throws the fragment away, so it
# passes on *any* wrong anchor, and it skips same-page `#...` links entirely. That blind
# spot let the README's only pointer to incremental corpus updates advertise
# `...walkthrough.md#7-incremental-updates-...` while the material lives under `## 8.`;
# `mkdocs build --strict` never sees README.md, so nothing else could catch it either.

# Anchors declared by hand rather than derived from a heading: `attr_list`'s `{#id}` and
# raw HTML. Both are honoured by GitHub and by the docs site.
_ATTR_LIST_ID = re.compile(r"\{[^}]*#([A-Za-z0-9_\-:.]+)[^}]*\}")
_HTML_ID = re.compile(r"<[^>]*\b(?:id|name)\s*=\s*[\"']([^\"']+)[\"']")
# A mkdocstrings injection point. The anchors under it are generated at build time from
# Python objects, so no static reader can enumerate them.
_MKDOCSTRINGS = re.compile(r"^:::\s+\S", re.MULTILINE)


def _github_slug(heading: str) -> str:
    """Slug GitHub gives a heading — the renderer these pages are actually read in.

    ``mkdocs.yml`` pins ``pymdownx.slugs.slugify`` for exactly this reason (a hand-written
    ``#anchor`` must resolve on github.com *and* on the docs site), so reproducing GitHub's
    rule here keeps the guard aligned with both. The rule: strip inline markup, lowercase,
    drop every character that is not a word character, hyphen or space, then turn spaces
    into hyphens *without collapsing runs* — which is why "GPU requirements + the Blackwell
    note" becomes ``gpu-requirements--the-blackwell-note`` and "5.4 Release-signing" becomes
    ``54-release-signing``.
    """
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading.strip())  # [text](url) -> text
    text = text.replace("`", "")
    text = re.sub(r"[*~]", "", text)  # emphasis / strikethrough markers
    text = re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE)  # `\w` keeps `_`, as GitHub does
    return text.lower().replace(" ", "-")


def _anchors_of(path: Path) -> set[str] | None:
    """Every anchor ``path`` defines, or ``None`` when they cannot be known statically.

    ``None`` means "generated": a page carrying a mkdocstrings ``:::`` block grows anchors
    from Python objects at build time (``docs/reference/*.md``), and guessing at them would
    trade a real check for a flaky one. Those targets are left to ``mkdocs --strict``.
    """
    text = path.read_text()
    if _MKDOCSTRINGS.search(text):
        return None

    anchors: set[str] = set()
    seen: dict[str, int] = {}
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        anchors.update(_ATTR_LIST_ID.findall(line))
        anchors.update(_HTML_ID.findall(line))
        match = re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
        if not match:
            continue
        slug = _github_slug(re.sub(_ATTR_LIST_ID, "", match.group(1)))
        # GitHub disambiguates repeats with -1, -2, … in document order.
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchors.add(slug if count == 0 else f"{slug}-{count}")
    return anchors


def _dangling_anchors(pages: list[Path], root: Path) -> tuple[list[str], int]:
    """``(problems, links_checked)`` for the fragment half of the cross-reference check."""
    problems: list[str] = []
    checked = 0
    for page in pages:
        for lineno, line in enumerate(page.read_text().splitlines(), 1):
            for target in _MD_LINK.findall(line):
                if target.startswith(("http://", "https://", "mailto:")) or "#" not in target:
                    continue
                path_part, fragment = target.split("#", 1)
                if not fragment:
                    continue
                resolved = page if not path_part else (page.parent / path_part).resolve()
                if not resolved.is_file() or resolved.suffix != ".md":
                    continue  # missing files are the neighbouring test's finding
                anchors = _anchors_of(resolved)
                if anchors is None:
                    continue  # build-time generated anchors — see _anchors_of
                checked += 1
                if fragment.lower() not in anchors:
                    where = resolved.relative_to(root)
                    problems.append(
                        f"{page.relative_to(root)}:{lineno}: #{fragment} "
                        f"is not a heading in {where}"
                    )
    return problems, checked


def test_no_dangling_section_anchors_in_the_front_door() -> None:
    """Every ``#fragment`` a published page links to must exist on the page it points at.

    A wrong anchor is silent in every gate the repository had: it renders fine, it links to
    a real file, and the reader simply lands at the top of a long document with no idea
    where the material they were promised is. The falsifier this was written against is
    ``README.md``'s pointer at the pod tutorial's *incremental updates* section — the only
    route from the front door to ``latence delta`` / ``retract`` / ``purge``.
    """
    problems, checked = _dangling_anchors(_front_door_docs(), _REPO_ROOT)

    # Non-vacuity: if the front door stops linking to sections at all, this guard is
    # checking nothing and must say so rather than pass.
    assert checked, (
        "no front-door page links to a section anchor any more — this guard is inspecting "
        "nothing. Restore the cross-references or retire the guard deliberately."
    )
    assert not problems, (
        "published pages link to section anchors that do not exist "
        f"(of {checked} fragment links checked):\n"
        + "\n".join(problems)
        + "\nGitHub renders these as a jump to the top of the page."
    )


def test_anchor_resolver_rejects_a_wrong_section_number(tmp_path: Path) -> None:
    """The resolver itself catches the exact shape of the defect, on a synthetic page.

    Keeps the guard above honest: were the README's link ever deleted rather than fixed,
    the tree-wide test would have nothing to find, but this one still proves the resolver
    distinguishes a right anchor from a wrong one.
    """
    target = tmp_path / "tutorial.md"
    target.write_text(
        "## 7. Discover the final output\n\ntext\n\n"
        "## 8. Incremental updates: add / update / retract / purge\n\ntext\n"
        "## GPU requirements + the Blackwell note\n\ntext\n"
        "## Custom {#pinned-id}\n\ntext\n"
    )
    anchors = _anchors_of(target)
    assert anchors is not None
    assert "8-incremental-updates-add--update--retract--purge" in anchors
    assert "7-incremental-updates-add--update--retract--purge" not in anchors
    assert "gpu-requirements--the-blackwell-note" in anchors  # `+` removed, both spaces kept
    assert "pinned-id" in anchors

    page = tmp_path / "page.md"
    page.write_text("[bad](./tutorial.md#7-incremental-updates-add--update--retract--purge)\n")
    problems, checked = _dangling_anchors([page], tmp_path)
    assert checked == 1 and len(problems) == 1, (problems, checked)

    page.write_text("[good](./tutorial.md#8-incremental-updates-add--update--retract--purge)\n")
    assert _dangling_anchors([page], tmp_path) == ([], 1)


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
    distributions = {pkg.name for pkg in _PACKAGES.iterdir() if (pkg / "pyproject.toml").is_file()}
    named = set(re.findall(r"pip install (?:-e )?'?(latence[a-z0-9-]*)", _readme_shell_blocks()))
    assert named, "README no longer documents an install — the front door needs one"

    missing = sorted(n for n in named if n not in distributions)
    assert not missing, (
        f"README tells the reader to install {missing}, which does not exist under packages/. "
        "There is no publishable `latence` meta-package; name the real distributions."
    )


# --------------------------------------------------------------------------------------
# The licence summary (R3-docs_truth-5)
# --------------------------------------------------------------------------------------

_LICENCES = _REPO_ROOT / "THIRD-PARTY-LICENSES.md"

# How a page phrases a blanket "nothing here is restricted" claim. Kept deliberately narrow:
# only sweeping statements, never a scoped one ("the default Embedder is permissive").
_BLANKET_PERMISSIVE = re.compile(
    r"\ball\b[^.\n]{0,60}\bpermissive\b|\bpermissive\b[^.\n]{0,30}\bonly\b|\bpermissive-only\b",
    re.IGNORECASE,
)

# Wording that turns a blanket claim into an honest one by naming the carve-out in the same
# breath. `docs/CONTINUITY.md` §2 does exactly this and must keep passing.
_CARVE_OUT = (
    "proprietary",
    "eula",
    "nvidia",
    "cuda",
    "exception",
    "carve-out",
    "carve out",
    "opt-in",
    "restricted",
    "not open-source",
    "non-permissive",
)

# Evidence, inside THIRD-PARTY-LICENSES.md itself, that something in it is NOT permissive.
_NON_PERMISSIVE_EVIDENCE = re.compile(
    r"LicenseRef-NVIDIA-Proprietary|proprietary \((?:CUDA|cuDNN|NCCL|NVSHMEM) EULA\)"
    r"|not\*{0,2} open-source|Not OSI open-source",
    re.IGNORECASE,
)


def _blocks(text: str) -> list[tuple[int, str]]:
    """``(start_lineno, block_text)`` for each paragraph / top-level bullet.

    A claim and its qualification live in the same bullet, not necessarily on the same
    physical line, so the unit of judgement is the block — a top-level ``- `` item together
    with its indented continuation lines, or a blank-line-delimited paragraph.
    """
    out: list[tuple[int, str]] = []
    start = 1
    buf: list[str] = []

    def flush() -> None:
        if any(line.strip() for line in buf):
            out.append((start, "\n".join(buf)))

    for lineno, line in enumerate(text.splitlines(), 1):
        starts_block = not line.strip() or bool(re.match(r"^(?:[-*+]\s|#{1,6}\s|\|)", line))
        if starts_block and buf:
            flush()
            buf, start = [], lineno
        if not buf:
            start = lineno
        buf.append(line)
    flush()
    return out


def test_no_front_door_page_calls_the_dependency_tree_wholly_permissive() -> None:
    """A page that summarises ``THIRD-PARTY-LICENSES.md`` may not call it uniformly permissive.

    The pre-fix README's front-door index read "runtime dep + model licenses (all permissive)"
    while the very file it links to opens with "NVIDIA CUDA wheels that are **not**
    open-source", tabulates ~40 wheels as ``LicenseRef-NVIDIA-Proprietary``, and records
    EmbeddingGemma as "Not OSI open-source". A procurement reviewer who reads only the README
    concludes the tree is licence-clean and skips the EULA review the repo's own disclosure
    file says is required.

    Ground truth is read out of ``THIRD-PARTY-LICENSES.md`` rather than hard-coded, so the day
    a *new* proprietary wheel lands the guard re-arms itself. A blanket claim is still allowed
    when the same block names the carve-out — which is how ``docs/CONTINUITY.md`` §2 states it.
    """
    assert _LICENCES.is_file(), "THIRD-PARTY-LICENSES.md is missing"
    licence_text = _LICENCES.read_text()

    # Non-vacuity: the guard only means something while the file really does disclose
    # non-permissive entries. If that ever stops being true, this assertion fails loudly and a
    # human decides whether the blanket claim has become honest again.
    evidence = _NON_PERMISSIVE_EVIDENCE.findall(licence_text)
    assert evidence, (
        "THIRD-PARTY-LICENSES.md no longer discloses any non-permissive entry. If the "
        "proprietary CUDA wheels and the Gemma opt-in are genuinely gone, retire this guard "
        "deliberately; do not let it pass by finding nothing."
    )

    problems: list[str] = []
    citing = 0
    for page in _front_door_docs():
        for lineno, block in _blocks(page.read_text()):
            if "THIRD-PARTY-LICENSES.md" not in block:
                continue
            citing += 1
            low = block.lower()
            if _BLANKET_PERMISSIVE.search(block) and not any(h in low for h in _CARVE_OUT):
                problems.append(
                    f"{page.relative_to(_REPO_ROOT)}:{lineno}: " + " ".join(block.split())[:200]
                )

    assert citing, "no published page cites THIRD-PARTY-LICENSES.md — keep the pointer"
    assert not problems, (
        "published pages summarise THIRD-PARTY-LICENSES.md as uniformly permissive, but that "
        f"file discloses {len(evidence)} non-permissive entries (NVIDIA CUDA EULAs, Gemma):\n"
        + "\n".join(problems)
        + "\nSay 'permissive by default' and name the disclosed exceptions."
    )


def test_readme_still_points_at_the_licence_disclosure() -> None:
    """Correcting the claim must not delete the pointer to the disclosure file."""
    assert "THIRD-PARTY-LICENSES.md" in _README.read_text(), (
        "README no longer links THIRD-PARTY-LICENSES.md — the front door must reach the "
        "licence disclosure in one hop."
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
