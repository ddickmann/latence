# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""``latence-campaign`` — the e2e-sota-run campaign CLI.

    latence-campaign init    [--campaign-root DIR] [--datasets a,b,c] [--pod-ssh TARGET]
                             [-o campaign.yaml] [--non-interactive]
    latence-campaign status  [-c campaign.yaml] [--repo-root DIR]
    latence-campaign run     <stage> [-c campaign.yaml] [--repo-root DIR] [--fresh]
                             [--accept-non-commercial]
    latence-campaign approve <stage> [-c campaign.yaml]
    latence-campaign resume  [-c campaign.yaml] [--repo-root DIR] [--accept-non-commercial]

Stdlib ``argparse`` deliberately, matching ``latence-label``: the campaign must be drivable
straight from a clone with the framework installed and nothing else.

``init`` follows the ``latence setup`` wizard contract: flag > prompt > default, "why it
matters" lines to stderr, ONE frozen answers struct from either path, generate →
dry-validate → write, and exactly one next command. Exit codes, also per that contract:
**2** usage / contradiction, **1** invalid artifact or blocked/failed stage, **0** success.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from latence_benchmark.campaign.stages import (
    CampaignContext,
    GateRefusedError,
    StageBlockedError,
    registry,
    run_stage,
    stage_dir,
)
from latence_benchmark.campaign.state import (
    APPROVED,
    BLOCKED,
    DONE,
    STATE_FILENAME,
    CampaignState,
    CampaignStateError,
)
from latence_benchmark.campaign.wizard import (
    KNOWN_DATASETS,
    CampaignAnswers,
    CampaignConfigError,
    build_campaign_config,
    parse_campaign_config,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from latence_benchmark.campaign.stages import StageSpec

DEFAULT_CONFIG = Path("campaign.yaml")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="latence-campaign",
        description="Run the staged e2e-sota-run campaign: gated stages, filesystem "
        "done-probes, atomic STATE.json.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="The wizard: answers -> a validated campaign.yaml.")
    init.add_argument(
        "--campaign-root",
        default=None,
        help="Directory holding STATE.json + stages/ artifacts. Default: the --output "
        "file's directory.",
    )
    init.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated corpus selection. Default: all of "
        f"{','.join(KNOWN_DATASETS)}.",
    )
    init.add_argument(
        "--pod-ssh",
        default=None,
        help="SSH target of the GPU pod for later stages (user@host[:port] or an alias); "
        "empty for the local-only stages.",
    )
    init.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Where to save the generated campaign.yaml.",
    )
    init.add_argument(
        "--non-interactive",
        action="store_true",
        help="Take every answer from flags (no prompts). Same inputs -> byte-identical "
        "config.",
    )

    def _common(p: argparse.ArgumentParser, *, repo_root: bool) -> None:
        p.add_argument(
            "--config",
            "-c",
            type=Path,
            default=DEFAULT_CONFIG,
            help="The campaign.yaml to operate on.",
        )
        if repo_root:
            p.add_argument(
                "--repo-root",
                type=Path,
                default=Path.cwd(),
                help="The latence-framework checkout the stages operate on (default: cwd).",
            )

    status = sub.add_parser("status", help="Render the stage table + probe results.")
    _common(status, repo_root=True)

    run = sub.add_parser("run", help="Run ONE stage (refused unless the previous one is approved).")
    run.add_argument("stage", help="Stage name, e.g. s1-corpus.")
    _common(run, repo_root=True)
    run.add_argument(
        "--fresh",
        action="store_true",
        help="Reset a done/blocked stage and redo it (re-runs are otherwise no-ops).",
    )
    run.add_argument(
        "--accept-non-commercial",
        action="store_true",
        help="Explicitly acknowledge research-only use for non-commercial corpora "
        "(OHR-Bench). Recorded in the stage metrics; never defaulted on.",
    )

    approve = sub.add_parser(
        "approve", help="Record the operator sign-off on a done stage (opens the next gate)."
    )
    approve.add_argument("stage", help="Stage name to approve.")
    _common(approve, repo_root=False)

    resume = sub.add_parser(
        "resume", help="Continue an interrupted campaign (filesystem probes, not exit codes)."
    )
    _common(resume, repo_root=True)
    resume.add_argument("--accept-non-commercial", action="store_true", help=argparse.SUPPRESS)

    return parser


# -- init -------------------------------------------------------------------------------------


def _ask(flag: str | None, prompt: str, why: str, default: str, *, interactive: bool) -> str:
    """flag > prompt > default — the `latence setup` resolution order, verbatim."""
    if flag is not None:
        return flag
    if not interactive:
        return default
    print(f"  ({why})", file=sys.stderr)
    print(f"{prompt} [{default}]: ", end="", file=sys.stderr, flush=True)
    raw = input().strip()
    return raw or default


def _cmd_init(args: argparse.Namespace) -> int:
    interactive = not args.non_interactive
    output: Path = args.output

    root = _ask(
        args.campaign_root,
        "Campaign root (STATE.json + per-stage artifacts live here)",
        "the campaign's working directory; safe to keep next to campaign.yaml",
        str(output.parent.resolve()),
        interactive=interactive,
    )
    datasets_raw = _ask(
        args.datasets,
        "Corpus datasets (comma-separated)",
        "the 6+2 S0 selection; trim it to iterate faster, the license gate runs either way",
        ",".join(KNOWN_DATASETS),
        interactive=interactive,
    )
    pod_ssh = _ask(
        args.pod_ssh,
        "Pod SSH target (empty for local-only stages)",
        "later stages (OCR serving, GPU extraction) run on this host; S1 only records it",
        "",
        interactive=interactive,
    )

    datasets = tuple(d.strip() for d in datasets_raw.split(",") if d.strip())
    try:
        answers = CampaignAnswers(
            campaign_root=root, datasets=datasets, pod_ssh=pod_ssh.strip()
        )
    except CampaignConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Generate -> DRY-VALIDATE -> write: a config the CLI itself cannot re-load is never
    # saved (exit 1 = invalid artifact, distinct from exit 2 = contradictory answers).
    text = build_campaign_config(answers)
    try:
        parse_campaign_config(text)
    except CampaignConfigError as exc:
        print(f"error: generated campaign.yaml failed validation: {exc}", file=sys.stderr)
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")

    print(f"Saved campaign config to: {output}")
    print(f"  campaign root: {answers.campaign_root}")
    print(f"  corpus ({len(answers.datasets)}): {', '.join(answers.datasets)}")
    print(f"  pod ssh: {answers.pod_ssh or '(none — local-only stages)'}")
    print("")
    print(f"Run `latence-campaign status -c {output}` to see the stage table.")
    return 0


# -- shared loading ---------------------------------------------------------------------------


def _load(
    config: Path,
) -> tuple[CampaignAnswers, Path, Sequence[StageSpec], CampaignState]:
    if not config.is_file():
        msg = f"Campaign config not found: {config} (run `latence-campaign init` first)."
        raise CampaignConfigError(msg)
    answers = parse_campaign_config(config.read_text(encoding="utf-8"))
    campaign_root = Path(answers.campaign_root)
    specs = registry()
    state = CampaignState(campaign_root / STATE_FILENAME, [s.name for s in specs])
    return answers, campaign_root, specs, state


def _context(
    args: argparse.Namespace, answers: CampaignAnswers, campaign_root: Path
) -> CampaignContext:
    return CampaignContext(
        campaign_root=campaign_root,
        repo_root=Path(args.repo_root).resolve(),
        answers=answers,
        accept_non_commercial=bool(getattr(args, "accept_non_commercial", False)),
    )


# -- status -----------------------------------------------------------------------------------


def _cmd_status(args: argparse.Namespace) -> int:
    answers, campaign_root, specs, state = _load(args.config)
    ctx = _context(args, answers, campaign_root)

    print(f"campaign: {campaign_root}  ({len(answers.datasets)} dataset(s): "
          f"{', '.join(answers.datasets)})")
    print("")
    header = f"{'stage':<14} {'status':<9} {'done-probe':<12} detail"
    print(header)
    print("-" * len(header))
    for spec in specs:
        st = state.stage(spec.name)
        probe = "satisfied" if spec.done_probe(ctx) else "unsatisfied"
        if st.status == BLOCKED:
            detail = st.reason
        elif st.status == APPROVED:
            detail = f"approved {st.approved_at}"
        else:
            detail = spec.progress_probe(ctx)
        print(f"{spec.name:<14} {st.status:<9} {probe:<12} {detail}")
        metrics = stage_dir(campaign_root, spec.name) / "metrics.json"
        marker = "present" if metrics.is_file() else "absent"
        print(f"{'':<14} {'':<9} {'':<12} metrics: {metrics} ({marker})")
    return 0


# -- run / approve / resume -------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    answers, campaign_root, specs, state = _load(args.config)
    ctx = _context(args, answers, campaign_root)
    note = run_stage(specs, args.stage, ctx, state, fresh=args.fresh)
    print(note)
    if state.status_of(args.stage) == DONE:
        print(
            f"Inspect the gate artifact, then `latence-campaign approve {args.stage} "
            f"-c {args.config}` to open the next stage."
        )
    return 0


def _cmd_approve(args: argparse.Namespace) -> int:
    _answers, _campaign_root, _specs, state = _load(args.config)
    state.approve(args.stage)
    state.save()
    print(f"{args.stage}: approved — the next stage's gate is open.")
    return 0


def _cmd_resume(args: argparse.Namespace) -> int:
    answers, campaign_root, specs, state = _load(args.config)
    ctx = _context(args, answers, campaign_root)

    for spec in specs:
        status = state.status_of(spec.name)
        if status == APPROVED:
            continue
        if status == BLOCKED:
            print(
                f"{spec.name}: BLOCKED — {state.stage(spec.name).reason}\n"
                f"Fix the cause, then `latence-campaign run {spec.name} --fresh`.",
                file=sys.stderr,
            )
            return 1
        if status == DONE:
            print(
                f"{spec.name}: done, awaiting sign-off — "
                f"`latence-campaign approve {spec.name}` opens the next stage."
            )
            return 0
        # pending or interrupted-while-running: the probe (inside run_stage) decides
        # whether work remains; a satisfied probe completes it without re-running.
        note = run_stage(specs, spec.name, ctx, state)
        print(note)
        return 0

    print("All stages are approved — the campaign is complete.")
    return 0


# -- entry ------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    commands = {
        "init": _cmd_init,
        "status": _cmd_status,
        "run": _cmd_run,
        "approve": _cmd_approve,
        "resume": _cmd_resume,
    }
    handler = commands[args.command]
    try:
        return handler(args)
    except GateRefusedError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except StageBlockedError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 1
    except (CampaignConfigError, CampaignStateError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
