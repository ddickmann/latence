"""T-OCR-1 — ``serve.sh`` must not pay the Blackwell tax on an Ampere host.

``deploy/env/served-ocr-server/serve.sh`` exported ``VLLM_USE_FLASHINFER_SAMPLER=0`` for **every**
host (serve.sh:27). That setting is FM-FLASHINFER-SM120's recorded fix: FlashInfer's capability
check misparses compute capability ``12.0`` as the integer ``12``, which compares below 75, so a
genuinely capable ``sm_120`` device is refused. ``86`` parses fine — the failure mode **does not
exist** on Ampere, so forcing the workaround there disables a working fast path for nothing.

These tests run the REAL script. The only things faked are the two external binaries the script
shells out to — ``nvidia-smi`` (the capability probe) and ``vllm`` itself, which records the
environment and argv it was exec'd with instead of loading a model. No function of ours is
monkeypatched: the capability parse, the fail-safe branch and the flag list are all executed as an
operator would execute them. That the resulting server actually starts, captures CUDA graphs and
serves pages is **GPU-ONLY** and is proven on the pod, not here.

The asymmetry that decides the unknown-capability branch: a wrong "leave FlashInfer on" is a
**startup abort** with a documented one-line fix; a wrong "disable" is merely slower. Same shape as
D2's JIT argument — so an unreadable/absent probe **fails safe** and keeps the workaround, and only
a positively-identified pre-12.0 device opts out of it.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml

# packages/latence-core/tests/ -> repo root is three parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SERVE = _REPO_ROOT / "deploy" / "env" / "served-ocr-server" / "serve.sh"

#: What the fake ``vllm`` reports when an environment variable was never exported at all.
_UNSET = "<unset>"

_BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(_BASH is None, reason="bash is required to execute serve.sh")


def _bash() -> str:
    assert _BASH is not None  # narrowed by the module-level skipif
    return _BASH


def _write_exec(path: Path, body: str) -> None:
    """Write an executable stub interpreted by the same bash the test runs the script with."""
    path.write_text(f"#!{_bash()}\n{body}", encoding="utf-8")
    path.chmod(0o755)


class ServeInvocation:
    """What ``serve.sh`` handed to ``vllm``: the argv, plus the env vars that decide the fast path."""

    def __init__(self, env: dict[str, str], args: list[str], stderr: str) -> None:
        self.env = env
        self.args = args
        self.stderr = stderr

    @property
    def flashinfer_sampler(self) -> str:
        return self.env.get("VLLM_USE_FLASHINFER_SAMPLER", _UNSET)

    @property
    def attention_backend(self) -> str:
        return self.env.get("VLLM_ATTENTION_BACKEND", _UNSET)

    def value_of(self, flag: str) -> str:
        """The value that follows ``flag`` in the argv (the script uses ``--flag value`` form)."""
        assert flag in self.args, f"{flag} is absent from {self.args!r}"
        return self.args[self.args.index(flag) + 1]


def _run_serve(
    tmp_path: Path,
    *,
    compute_cap: str | None,
    env_overrides: dict[str, str] | None = None,
    extra_args: Sequence[str] = (),
) -> ServeInvocation:
    """Execute the real ``serve.sh`` against a faked ``nvidia-smi`` and a recording ``vllm``.

    ``compute_cap`` is what the faked ``nvidia-smi --query-gpu=compute_cap`` prints (one line per
    GPU); ``None`` means the binary is **absent** — the probe cannot answer at all. ``PATH`` holds
    nothing but the stub directory, so a real ``nvidia-smi`` on the host machine can never leak in
    and make this test's verdict depend on where it runs.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True)
    capture = tmp_path / "vllm-invocation.txt"

    _write_exec(
        bindir / "vllm",
        "{\n"
        '  printf "ENV=VLLM_USE_FLASHINFER_SAMPLER=%s\\n" '
        '"${VLLM_USE_FLASHINFER_SAMPLER-' + _UNSET + '}"\n'
        '  printf "ENV=VLLM_ATTENTION_BACKEND=%s\\n" "${VLLM_ATTENTION_BACKEND-' + _UNSET + '}"\n'
        '  printf "ARG=%s\\n" "$@"\n'
        '} > "$LATENCE_OCR_TEST_CAPTURE"\n',
    )
    if compute_cap is not None:
        _write_exec(bindir / "nvidia-smi", f"printf '%s\\n' {shlex.quote(compute_cap)}\n")

    env = {
        "PATH": str(bindir),
        "HOME": str(tmp_path),
        "LATENCE_OCR_TEST_CAPTURE": str(capture),
    }
    env.update(env_overrides or {})
    proc = subprocess.run(  # noqa: S603 - fixed argv, stub PATH, no shell
        [_bash(), str(_SERVE), *extra_args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"serve.sh exited {proc.returncode}\nstderr:\n{proc.stderr}"
    assert capture.is_file(), f"serve.sh never reached `vllm`\nstderr:\n{proc.stderr}"

    captured_env: dict[str, str] = {}
    args: list[str] = []
    for line in capture.read_text(encoding="utf-8").splitlines():
        if line.startswith("ENV="):
            name, _, value = line[len("ENV=") :].partition("=")
            if value != _UNSET:
                captured_env[name] = value
        elif line.startswith("ARG="):
            args.append(line[len("ARG=") :])
    return ServeInvocation(captured_env, args, proc.stderr)


# --- the capability gate ----------------------------------------------------------------------


def test_ampere_does_not_get_the_flashinfer_sampler_disabled(tmp_path: Path) -> None:
    """sm_86 parses to 86, which is >= 75 — the workaround must not fire. RED before T-OCR-1."""
    run = _run_serve(tmp_path, compute_cap="8.6")
    assert run.flashinfer_sampler == _UNSET, (
        "serve.sh disabled the FlashInfer sampler on an sm_86 host. FM-FLASHINFER-SM120 is a 12.0 "
        "parse bug; 86 parses fine, so this is a pure throughput tax on Ampere."
    )


def test_blackwell_still_gets_the_recorded_fix(tmp_path: Path) -> None:
    """sm_120 is the device the workaround exists for — it must still be applied automatically."""
    run = _run_serve(tmp_path, compute_cap="12.0")
    assert run.flashinfer_sampler == "0", (
        "serve.sh left the FlashInfer sampler enabled on sm_120 — that is the exact configuration "
        "that aborts with 'FlashInfer requires sm75+' (FM-FLASHINFER-SM120)."
    )


def test_absent_probe_fails_safe(tmp_path: Path) -> None:
    """No ``nvidia-smi`` => unknown capability => keep the workaround (slow beats a startup abort)."""
    run = _run_serve(tmp_path, compute_cap=None)
    assert run.flashinfer_sampler == "0"


def test_unparseable_capability_fails_safe(tmp_path: Path) -> None:
    """A driver that answers ``N/A`` is not an Ampere identification — keep the workaround."""
    run = _run_serve(tmp_path, compute_cap="N/A")
    assert run.flashinfer_sampler == "0"


def test_any_blackwell_in_a_mixed_host_disables_it(tmp_path: Path) -> None:
    """The probe reports one line per GPU; a single >= 12.0 device decides for the process."""
    run = _run_serve(tmp_path, compute_cap="8.6\n12.0")
    assert run.flashinfer_sampler == "0"


def test_pre_blackwell_generations_are_all_left_alone(tmp_path: Path) -> None:
    """Turing/Ampere/Ada/Hopper all parse above 75 — none of them needs the workaround."""
    for cap in ("7.5", "8.0", "8.6", "8.9", "9.0"):
        run = _run_serve(tmp_path / f"cap-{cap}", compute_cap=cap)
        assert run.flashinfer_sampler == _UNSET, f"sm_{cap} was needlessly taxed"


def test_operator_override_wins_in_both_directions(tmp_path: Path) -> None:
    """The escape hatch survives the gate: an explicit value is never second-guessed."""
    forced_on = _run_serve(
        tmp_path / "forced-on",
        compute_cap="12.0",
        env_overrides={"VLLM_USE_FLASHINFER_SAMPLER": "1"},
    )
    assert forced_on.flashinfer_sampler == "1", "an operator opting IN on sm_120 was overruled"

    forced_off = _run_serve(
        tmp_path / "forced-off",
        compute_cap="8.6",
        env_overrides={"VLLM_USE_FLASHINFER_SAMPLER": "0"},
    )
    assert forced_off.flashinfer_sampler == "0", "an operator opting OUT on sm_86 was overruled"


def test_capability_can_be_declared_when_the_probe_is_unavailable(tmp_path: Path) -> None:
    """A container without ``nvidia-smi`` can still identify itself instead of paying the tax."""
    run = _run_serve(
        tmp_path,
        compute_cap=None,
        env_overrides={"LATENCE_OCR_COMPUTE_CAP": "8.6"},
    )
    assert run.flashinfer_sampler == _UNSET


def test_the_decision_is_announced_on_stderr(tmp_path: Path) -> None:
    """An operator reading the startup log must be able to see which branch was taken."""
    ampere = _run_serve(tmp_path / "ampere", compute_cap="8.6")
    blackwell = _run_serve(tmp_path / "blackwell", compute_cap="12.0")
    assert "VLLM_USE_FLASHINFER_SAMPLER" in ampere.stderr
    assert "VLLM_USE_FLASHINFER_SAMPLER" in blackwell.stderr
    assert ampere.stderr != blackwell.stderr


# --- the serving surface ----------------------------------------------------------------------


def test_async_scheduling_is_requested(tmp_path: Path) -> None:
    """LightOn's own published serving recipe. RED before T-OCR-1."""
    run = _run_serve(tmp_path, compute_cap="8.6")
    assert "--async-scheduling" in run.args


def test_cudagraph_mode_is_stated_explicitly_and_is_the_attainable_one(tmp_path: Path) -> None:
    """``FULL_AND_PIECEWISE``, stated rather than inherited. RED before T-OCR-1.

    Not ``FULL``: FlashAttention-2 declares ``UNIFORM_BATCH`` support, so vLLM auto-downgrades a
    ``FULL`` request to exactly this mode anyway — asking for it would only make the log lie about
    what we intended. Not ``cudagraph_mm_encoder``: ``LightOnOCRForConditionalGeneration`` is absent
    from vLLM's supported-models table for encoder CUDA graphs, so it is a no-op at best.
    """
    run = _run_serve(tmp_path, compute_cap="8.6")
    config = json.loads(run.value_of("--compilation-config"))
    assert config["cudagraph_mode"] == "FULL_AND_PIECEWISE"
    assert "cudagraph_mm_encoder" not in config


def test_the_attention_backend_is_left_to_vllm(tmp_path: Path) -> None:
    """FA2 is already the sm_86 default and ``VLLM_ATTENTION_BACKEND`` is deprecated for removal."""
    run = _run_serve(tmp_path, compute_cap="8.6")
    assert run.attention_backend == _UNSET
    assert not [a for a in run.args if a.startswith("--attention-backend")]


def test_enforce_eager_is_never_passed(tmp_path: Path) -> None:
    """The refuted remedy: it clears the abort by turning cudagraphs OFF, flooring throughput."""
    for cap in ("8.6", "12.0", None):
        run = _run_serve(tmp_path / f"cap-{cap}", compute_cap=cap)
        assert "--enforce-eager" not in run.args


def test_the_measured_settings_survive(tmp_path: Path) -> None:
    """The live-run values this ticket does NOT change stay exactly where they were measured."""
    run = _run_serve(tmp_path, compute_cap="8.6")
    assert run.value_of("--dtype") == "bfloat16"
    assert run.value_of("--max-model-len") == "10000"
    assert run.value_of("--limit-mm-per-prompt") == '{"image": 1}'
    assert run.value_of("--mm-processor-cache-gb") == "0"
    assert "--no-enable-prefix-caching" in run.args


def test_operator_arguments_are_still_appended_verbatim(tmp_path: Path) -> None:
    """``"$@"`` must stay last so any flag above can be overridden without editing the file."""
    run = _run_serve(
        tmp_path,
        compute_cap="8.6",
        extra_args=("--gpu-memory-utilization", "0.80"),
    )
    assert run.args[-2:] == ["--gpu-memory-utilization", "0.80"]


def test_gpu_memory_utilization_is_operator_settable(tmp_path: Path) -> None:
    """T-OCR-1's 24 GB re-derivation (start 0.80, fall back to 0.72) is a GPU-ONLY measurement.

    The value that ships is the one that was actually measured — 0.72 on the live 519-page run. The
    24 GB sweep happens on the pod through this env knob and the number it produces is written back
    then; hard-coding an unmeasured 0.80 here would be exactly the "constant to copy" the ticket
    forbids.
    """
    default = _run_serve(tmp_path / "default", compute_cap="8.6")
    assert default.value_of("--gpu-memory-utilization") == "0.72"

    swept = _run_serve(
        tmp_path / "swept",
        compute_cap="8.6",
        env_overrides={"LATENCE_OCR_GPU_MEM_UTIL": "0.80"},
    )
    assert swept.value_of("--gpu-memory-utilization") == "0.80"


# --- static: the refuted remedy appears in no executable surface -------------------------------

_COMMENT = re.compile(r"^\s*#")


def test_serve_sh_names_enforce_eager_only_in_a_comment() -> None:
    """Every mention must be a warning, never a line bash would execute."""
    offenders = [
        (number, line)
        for number, line in enumerate(_SERVE.read_text(encoding="utf-8").splitlines(), start=1)
        if "--enforce-eager" in line and not _COMMENT.match(line)
    ]
    assert not offenders, f"serve.sh would pass --enforce-eager: {offenders!r}"


def test_no_deployment_surface_ships_enforce_eager_as_an_argument() -> None:
    """The compose command list and the chart's extraArgs default are argv, not prose."""
    compose = yaml.safe_load(
        (_REPO_ROOT / "deploy" / "docker-compose.yml").read_text(encoding="utf-8")
    )
    for name, service in compose["services"].items():
        command = service.get("command") or []
        assert not [a for a in command if "--enforce-eager" in str(a)], (
            f"docker-compose service {name} passes --enforce-eager"
        )

    values = yaml.safe_load(
        (_REPO_ROOT / "deploy" / "helm" / "latence" / "values.yaml").read_text(encoding="utf-8")
    )
    extra_args = values["vllmOcr"].get("extraArgs") or []
    assert not [a for a in extra_args if "--enforce-eager" in str(a)], (
        "the chart's vllmOcr.extraArgs default passes --enforce-eager"
    )
