#!/usr/bin/env bash
# Drive the extended end-to-end run to completion without a human in the polling loop.
#
# The defect this exists to fix, stated plainly: RUNBOOK.md is a procedure for a *person*, and a
# person supervising a 10-hour GPU run is the least reliable component in it. The first attempt at
# this run died when the SSH session that launched the corpus fetch went away. Nothing noticed for
# hours, because "noticing" was a human remembering to look.
#
# So this script owns the loop, built around three properties the manual procedure did not have:
#
#   1. RESUMABLE BY PROBE, NOT BY EXIT CODE. Each stage declares a `done` probe that asks the
#      filesystem whether the work exists. Re-running is therefore always safe, and a stage that
#      exited 0 without producing its artifact is NOT treated as complete — the exact failure mode
#      that let a "finished" corpus fetch leave two of three sources empty.
#
#   2. STALL DETECTION, NOT JUST FAILURE DETECTION. A hung download and a crashed one look identical
#      to `wait`. Each stage declares a `progress` probe returning a monotonic number. If it does not
#      advance for $STALL_SECS the stage is killed and retried.
#
#   3. ESCALATION IS A STATE, NOT A CRASH. When retries are exhausted the run stops at that stage and
#      records what was tried. A supervisor that silently skips a broken stage and reports success at
#      the end is worse than one that stops.
#
# Every command below was verified against the code before being written here — an earlier draft of
# this file guessed `--out`, `--stage` and a `latence labels derive` subcommand, none of which exist.
# A supervisor full of guesses fails more autonomously, not less.
#
# Usage:
#   setsid nohup bash benchmark/e2e/supervise.sh >> /workspace/run/supervisor.log 2>&1 < /dev/null &

set -u -o pipefail

# A detached, non-interactive shell does not read the rc files that put `uv` on PATH, so the
# supervisor saw `uv: command not found` while an interactive SSH session ran it fine. Anything
# installed to a user-local bin has to be put back on PATH explicitly here — the run must not depend
# on how the process happened to be started.
export PATH="${HOME:-/root}/.local/bin:${HOME:-/root}/.cargo/bin:/usr/local/bin:$PATH"

REPO="${REPO:-/workspace/latence-framework}"
CORPUS_ROOT="${CORPUS_ROOT:-/workspace/corpus}"
RUN_DIR="${RUN_DIR:-/workspace/run}"
OUT_DIR="${OUT_DIR:-/workspace/latence-out}"
RUN_ID="${RUN_ID:-govdocs1-v1}"

# The two-pass split is forced by VRAM: GLiNER2 at micro-batch 4 peaked at 21.6 GiB on a 32 GB card,
# so on 24 GB the vLLM OCR server and the learned stages can never co-exist. Pass 1 parses with the
# server up; pass 2 runs everything with it down, resuming parse from checkpoint.
#
# Stage-subsetting does not exist in the CLI (no --only/--until/--stage anywhere; `Pipeline` forbids
# extra keys). The supported mechanism is two stack YAMLs sharing one run_id + storage_uri, where
# pass 2 replays pass 1's `parse` checkpoint. Verified empirically end-to-end.
STACK_FULL="$RUN_DIR/latence.stack.yaml"
STACK_PARSE="$RUN_DIR/latence.parse-only.yaml"
LABEL_SCHEMA="${LABEL_SCHEMA:-$REPO/benchmark/e2e/label-schema-predefined.json}"
# Checkpoints are keyed on STAGE NAME ONLY — no config hash, no provider identity. Parse output is a
# single JSONL, not a directory (RUNBOOK §6 is wrong about this path).
CKPT_DIR="$OUT_DIR/_latence/runs/$RUN_ID/checkpoints"

# The SERVICE environment (ADR-0030): vLLM lives in its own venv so its torch cannot fight the
# pipeline environment's. serve.sh does `exec vllm serve`, so that venv's bin must be on PATH.
VLLM_BIN="${VLLM_BIN:-/workspace/vllm-env/bin/vllm}"
[ -x "$VLLM_BIN" ] && export PATH="$(dirname "$VLLM_BIN"):$PATH"

MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
POLL_SECS="${POLL_SECS:-30}"
# 20 minutes. Deliberately generous: a torch.compile warmup or a 90-zip unzip legitimately produces
# no countable progress for a long time, and a watchdog that fires on healthy work trains its
# operator to disable it.
STALL_SECS="${STALL_SECS:-1200}"
DAEMON_READY_SECS="${DAEMON_READY_SECS:-1800}"

STAGE_DIR="$RUN_DIR/stages"
STATE="$RUN_DIR/STATE.json"
mkdir -p "$STAGE_DIR"

# One supervisor per run directory. Two would race on STATE.json and start duplicate stage children,
# which is how two concurrent downloaders came to corrupt a shared archive. The keeper relaunches on
# death, so without this a slow-dying supervisor and its replacement could briefly coexist.
exec 8>"$RUN_DIR/.supervisor.lock"
if command -v flock >/dev/null 2>&1; then
  flock -n 8 || { echo "another supervisor holds $RUN_DIR/.supervisor.lock — exiting"; exit 0; }
fi

# Liveness is a PID FILE, never a `pgrep -f` pattern.
#
# `pgrep -f supervise.sh` matches ANY command line that merely mentions the string — including the
# shell that launched us, if its argv contained the path (an `scp`+`mv` one-liner does). That
# phantom cost SEVEN HOURS: the keeper saw a match, concluded the supervisor was healthy, and never
# relaunched the one that had actually exited. The same unsoundness made `pkill -f` repeatedly kill
# the SSH session issuing it.
#
# A pid file plus a liveness check on that exact pid cannot be fooled by a string in someone's argv.
echo $$ > "$RUN_DIR/supervisor.pid"
trap 'rm -f "$RUN_DIR/supervisor.pid"' EXIT

LATENCE="cd '$REPO' && uv run latence"

# ------------------------------------------------------------------ stage table
#
# kind: `task`   — runs to completion under the stall watchdog
#       `daemon` — launched detached, then polled until its done-probe answers. A server must NOT run
#                  under the watchdog: the watchdog waits for exit, and a healthy server never exits,
#                  so the stage would hang until the stall timer killed the thing it just started.

stage_names=()
declare -A S_DESC S_DONE S_PROG S_CMD S_KIND S_STALL

# Per-stage stall budget. One global timeout cannot serve both a 30-second copy and a multi-hour
# GPU pass: 20 minutes killed `parse` three times while it was legitimately working, and each kill
# abandoned a 57 GB checkpoint temp file — 171 GB of orphans before anyone noticed.
set_stall() { S_STALL["$1"]="$2"; }

add_stage() {
  stage_names+=("$1"); S_KIND["$1"]="$2"; S_DESC["$1"]="$3"
  S_DONE["$1"]="$4"; S_PROG["$1"]="$5"; S_CMD["$1"]="$6"
  # `[ -s file ]` means NON-EMPTY, and `touch` produces an empty file — so a stage can do all of its
  # real work, succeed, and still be judged incomplete. That cost three attempts and a BLOCKED run
  # on a preflight whose every actual check had passed. Markers must therefore carry content, and
  # this catches the mismatch at load time rather than after the work is done.
  case "$6" in
    *"touch '$RUN_DIR/"*)
      case "$4" in
        *"-s '$RUN_DIR/"*)
          echo "FATAL: stage '$1' writes its marker with touch (empty) but probes it with -s" >&2
          echo "       (non-empty). The stage would run, succeed, and never be marked complete." >&2
          exit 2
          ;;
      esac
      ;;
  esac
}

# Fail at second zero, not forty minutes in.
#
# `envspec.py` models TWO environments: the pipeline environment (the workspace packages) and the
# SERVICE environment — the vLLM OCR server behind the ADR-0030 endpoint seam. Installing the first
# and not the second is invisible until a stage actually reaches for the server, which here meant
# the corpus finished, the wizard ran, and only then did `vllm: not found` surface at stage 3 of 9.
#
# Every external binary the generated stack depends on is checked here, before any work happens. A
# preflight that runs after the expensive part is not a preflight.
add_stage preflight task \
  "Verify the tools every later stage needs are actually present" \
  "[ -s '$RUN_DIR/.preflight.ok' ]" \
  "echo 0" \
  "missing=''; \
   for tool in uv python3 unzip curl nvidia-smi; do \
     command -v \$tool >/dev/null 2>&1 || missing=\"\$missing \$tool\"; \
   done; \
   command -v vllm >/dev/null 2>&1 || [ -x '$VLLM_BIN' ] || missing=\"\$missing vllm(service-env)\"; \
   python3 -c 'import yaml' 2>/dev/null || missing=\"\$missing python3-yaml\"; \
   if [ -n \"\$missing\" ]; then \
     echo \"PREFLIGHT FAILED — missing:\$missing\"; \
     echo 'These are needed by LATER stages. Install them before the run rather than discovering'; \
     echo 'the gap after the corpus and wizard stages have already spent their time.'; \
     exit 1; \
   fi; \
   nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader \
     | tee '$RUN_DIR/.preflight.ok'"

add_stage corpus task \
  "Acquire GovDocs1 + SEC VPRR + OHR-Bench and write the measured manifest" \
  "[ -s '$CORPUS_ROOT/manifest.json' ]" \
  "echo \$(find '$CORPUS_ROOT' -type f ! -path '*_work*' 2>/dev/null | wc -l)\$(du -sk '$CORPUS_ROOT/_work' 2>/dev/null | cut -f1)" \
  "cd '$REPO' && CORPUS_ROOT='$CORPUS_ROOT' GOVDOCS_ZIPS=90 VPRR_PDFS=${VPRR_PDFS:-0} \
     EDGAR_XLSX=${EDGAR_XLSX:-0} ZENODO_OOXML=${ZENODO_OOXML:-0} FETCH_OHR=${FETCH_OHR:-0} \
     bash benchmark/corpus/fetch_corpus.sh"

# The wizard is the entry gate. The parse-only YAML is DERIVED from the wizard's output by dropping
# stages, never by authoring config — the emitted header explicitly invites diffable edits, and
# `latence stack check` exists to lint them, so the entry-gate discipline survives the trim.
add_stage wizard_parse task \
  "Generate the stack from the wizard, derive the parse-only pass, and lint both" \
  "[ -s '$STACK_PARSE' ]" \
  "ls '$RUN_DIR'/*.yaml 2>/dev/null | wc -l" \
  "$LATENCE setup --profile enterprise-sota \
     --source '$CORPUS_ROOT' --storage 'file://$OUT_DIR' \
     --chunk-max-tokens 768 --chunk-overlap-tokens 96 \
     --ocr-dpi 200 --ocr-max-longest-dim 1540 \
     --non-interactive -o '$STACK_FULL' \
   && python3 -c \"
import yaml, sys
d = yaml.safe_load(open('$STACK_FULL'))
keep = {'source', 'intake_screen', 'parse'}
kept = [s for s in d['stages'] if s['name'] in keep]
missing = keep - {s['name'] for s in kept}
if missing:
    sys.exit('wizard emitted no ' + ', '.join(sorted(missing)) + ' stage - cannot derive parse-only pass')
d['stages'] = kept
yaml.safe_dump(d, open('$STACK_PARSE','w'), sort_keys=False)
\" \
   && $LATENCE stack check '$STACK_PARSE'"

add_stage ocr_server daemon \
  "Bring up vLLM/LightOnOCR on sm_86 (compiled, CUDA graphs, no enforce-eager)" \
  "curl -sf --max-time 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1" \
  "wc -c < '$STAGE_DIR/ocr_server.log' 2>/dev/null || echo 0" \
  "cd '$REPO' && bash deploy/env/served-ocr-server/serve.sh"

# The progress probe must watch what grows DURING the stage, not what appears when it ends.
# `parse.jsonl` only exists on completion, so probing it read 0 for twenty minutes and the watchdog
# concluded a healthy stage was hung. The checkpoint DIRECTORY grows continuously — `source.jsonl`
# alone reaches ~57 GB as file bytes are materialised — so its size is the honest signal.
#
# A stale `.tmp` from a previous attempt is removed first: the runner writes checkpoints to a
# uuid-named temp and renames on success, so a killed attempt leaves tens of GB that nothing will
# ever claim, and three of them filled 171 GB here.
add_stage parse task \
  "Pass 1 — parse the corpus with the OCR server up" \
  "[ -s '$CKPT_DIR/parse.jsonl' ]" \
  "du -sk '$CKPT_DIR' 2>/dev/null | cut -f1" \
  "rm -f '$CKPT_DIR'/*.tmp; $LATENCE process --config '$STACK_PARSE' --run-id '$RUN_ID'"
# 96k files, ~48 GB, materialised through a network filesystem and then OCR'd on one 3090. Hours,
# legitimately. The probe above is what detects a REAL hang; this budget only needs to exceed the
# longest honest gap between two writes.
set_stall parse 5400

# Pass 2 overwrites quality-report.json, and replayed rows report duration 0.0 — so the parse timing,
# VRAM and drift evidence exists ONLY in the pass-1 report. Copy it out before pass 2 can destroy it.
add_stage parse_evidence task \
  "Preserve the pass-1 quality report before pass 2 overwrites it" \
  "[ -s '$RUN_DIR/parse-quality-report.json' ]" \
  "echo 0" \
  "cp '$OUT_DIR/_latence/runs/$RUN_ID/quality-report.json' '$RUN_DIR/parse-quality-report.json'"

add_stage ocr_down task \
  "Tear the OCR server down and confirm VRAM is actually released" \
  "! pgrep -f 'vllm serve' >/dev/null 2>&1" \
  "echo 0" \
  "pkill -f 'vllm serve'; sleep 20; nvidia-smi"

# The label pool is PREDEFINED — 37 authored types, no LLM, no profiling pass.
#
# This reverses an earlier choice, on measurement. Per-chunk narrowing via label_inducer.lexicon
# gives a 3.3x reduction in label-slots scored, and it was chosen to respect GLiNER2's uni-encoder
# cost. Measured against real GovDocs1 text, it also costs recall on exactly the types that matter
# most, because a lexicon cue only fires when the cue word is part of the entity's own surface form:
#
#   monetary amount   36.4% of chunks containing one are ever scored for it
#   person            64.7%
#   date              68.8%
#   organization       100%   (its cues ARE the surface form: Inc, Corp, University)
#
# A chunk holding "$3,200,000" that is never asked about money produces no error and no missing-data
# signal — the question simply was not posed. Worse, `_compile_cue` wraps every cue in \b…\b, so a
# cue like `$` or `%` can never match, is accepted silently, and is reported nowhere.
#
# The compute argument for narrowing was also weaker than it looked: entity, relation and PII slots
# are bucketed and scored SEPARATELY, so a forward pass sees 21 entity labels, not 37 — already
# inside the range where GLiNER2 degradation is documented. Full recall is worth 3.3x here.
add_stage labels task \
  "Install the predefined 37-type label pool (no LLM, no profiling, full recall)" \
  "[ -s '$LABEL_SCHEMA' ]" \
  "echo 0" \
  "test -s '$LABEL_SCHEMA' || { echo \"label schema missing at $LABEL_SCHEMA\"; exit 1; }; \
   python3 -c \"
import json, sys
d = json.load(open('$LABEL_SCHEMA'))
slots = {k: len(v) for k, v in d.items()}
total = sum(slots.values())
if not 30 <= total <= 40:
    sys.exit('label pool is ' + str(total) + ' types; the operator specified 30-40')
print('label pool OK:', slots, 'TOTAL', total)
\""

# Regenerating the stack with the derived pool requires T-WIZ-2 (`--label-candidates`). Riding the
# pool on `--predefined-labels` instead LOOKS correct and is not: with no induce stage no chunk
# carries induced_labels, `effective_labels` returns the config labels unchanged, and every chunk
# gets all 40 — reintroducing the defect the per-chunk redesign removed.
add_stage wizard_full task \
  "Regenerate the full stack wired to label_inducer.lexicon with the derived pool" \
  "[ -s '$RUN_DIR/.wizard_full.done' ]" \
  "echo 0" \
  "$LATENCE setup --profile enterprise-sota \
     --source '$CORPUS_ROOT' --storage 'file://$OUT_DIR' \
     --chunk-max-tokens 768 --chunk-overlap-tokens 96 \
     --ocr-dpi 200 --ocr-max-longest-dim 1540 \
     --predefined-labels '$LABEL_SCHEMA' \
     --non-interactive -o '$STACK_FULL' \
   && python3 -c \"
import json, yaml, sys
full = {s['name']: s for s in yaml.safe_load(open('$STACK_FULL'))['stages']}
par  = {s['name']: s for s in yaml.safe_load(open('$STACK_PARSE'))['stages']}
# The checkpoint skip is keyed on stage NAME alone — no config hash. If pass 2 changed anything
# about these three, pass 1's output would be replayed silently under the new config and we would
# measure the old run while believing it was the new one.
for name in ('source', 'intake_screen', 'parse'):
    if full.get(name) != par.get(name):
        sys.exit('FATAL: ' + name + ' differs between passes; the name-keyed checkpoint would replay pass 1 under pass 2 config')
# Zero LLM induction, asserted structurally: predefined-only is an OMISSION, so the proof is that
# no induce Stage exists at all — not that some flag was set.
if any('induce' in n for n in full):
    sys.exit('FATAL: an induce Stage was emitted; predefined-only must emit none')
if any(s.get('provider', '').startswith('label_inducer') for s in full.values()):
    sys.exit('FATAL: a label_inducer Provider is wired; the run must make zero induction calls')
# The pool must actually reach the extractor. A stack that carries the labels nowhere but the
# config header would extract against demo defaults and look fine.
pool = json.load(open('$LABEL_SCHEMA'))
got = len(full['extract']['config'].get('labels', []))
want = len(pool['entity_types'])
if got != want:
    sys.exit('FATAL: extract.config.labels has ' + str(got) + ' types, pool has ' + str(want))
print('verified: no induce Stage, no inducer Provider, extract carries all ' + str(want) + ' entity types')
\" \
   && $LATENCE stack check '$STACK_FULL' \
   && date -u +%Y-%m-%dT%H:%M:%SZ > '$RUN_DIR/.wizard_full.done'"

add_stage full_run task \
  "Pass 2 — chunk, PII screen, extract, KG, index (parse replays from checkpoint)" \
  "[ -s '$OUT_DIR/_latence/runs/$RUN_ID/export/corpus.jsonl' ]" \
  "find '$OUT_DIR/_latence/runs/$RUN_ID' -type f 2>/dev/null | wc -l" \
  "$LATENCE process --config '$STACK_FULL' --run-id '$RUN_ID'"

# ------------------------------------------------------------------ state

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

declare -A ST_STATUS ST_ATTEMPTS ST_STARTED ST_ENDED ST_NOTE
# Rewritten in full after every transition via temp file + rename, so a reader never observes a
# half-written state. STATE.json is never the thing that lies.
write_state() {
  local overall="$1" current="$2" tmp="$STATE.tmp"
  {
    printf '{\n  "run": "e2e-extended",\n  "updated": "%s",\n' "$(now)"
    printf '  "overall": "%s",\n  "current_stage": "%s",\n' "$overall" "$current"
    printf '  "run_id": "%s",\n  "host": "%s",\n  "pid": %s,\n' "$RUN_ID" "$(hostname)" "$$"
    printf '  "stages": [\n'
    local first=1 n
    for n in "${stage_names[@]}"; do
      [ $first -eq 1 ] || printf ',\n'; first=0
      printf '    {"name": "%s", "status": "%s", "attempts": %s, "started": "%s", "ended": "%s", "log": "%s", "note": "%s", "description": "%s"}' \
        "$n" "${ST_STATUS[$n]:-pending}" "${ST_ATTEMPTS[$n]:-0}" \
        "${ST_STARTED[$n]:-}" "${ST_ENDED[$n]:-}" "$STAGE_DIR/$n.log" \
        "$(printf '%s' "${ST_NOTE[$n]:-}" | tr '\n' ' ' | sed 's/"/\\"/g')" \
        "${S_DESC[$n]}"
    done
    printf '\n  ]\n}\n'
  } > "$tmp" && mv "$tmp" "$STATE"
}

say() { printf '[%s] %s\n' "$(now)" "$*"; }

# ------------------------------------------------------------------ execution

# Run a task stage under stall detection. Returns 0 on clean exit, 124 on stall, else the exit code.
# The command runs in its own process group so a kill takes its children (curl, xargs, python, vllm)
# with it — orphans would leave the retry contending with the process it replaced.
run_with_watchdog() {
  local name="$1" cmd="$2" log="$STAGE_DIR/$1.log"
  local stall="${S_STALL[$name]:-$STALL_SECS}"
  setsid bash -c "$cmd" >> "$log" 2>&1 < /dev/null &
  local pid=$! pgid
  pgid=$(ps -o pgid= "$pid" 2>/dev/null | tr -d ' ') || pgid=""

  local last_metric="" last_change=$SECONDS metric
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL_SECS"
    metric=$(eval "${S_PROG[$name]}" 2>/dev/null | tr -d ' \n') || metric=""
    if [ "$metric" != "$last_metric" ]; then
      last_metric="$metric"; last_change=$SECONDS
      say "  $name: progress=$metric"
    elif [ $(( SECONDS - last_change )) -ge "$stall" ]; then
      say "  $name: STALLED at progress=$metric for ${stall}s — killing"
      [ -n "$pgid" ] && kill -TERM -"$pgid" 2>/dev/null
      sleep 10
      [ -n "$pgid" ] && kill -KILL -"$pgid" 2>/dev/null
      wait "$pid" 2>/dev/null
      return 124
    fi
  done
  wait "$pid"; return $?
}

# Launch a server detached and wait for it to answer its own readiness probe.
run_daemon() {
  local name="$1" cmd="$2" log="$STAGE_DIR/$1.log"
  setsid bash -c "$cmd" >> "$log" 2>&1 < /dev/null &
  local waited=0
  while [ "$waited" -lt "$DAEMON_READY_SECS" ]; do
    sleep "$POLL_SECS"; waited=$(( waited + POLL_SECS ))
    if eval "${S_DONE[$name]}" 2>/dev/null; then
      say "  $name: ready after ${waited}s"
      return 0
    fi
    say "  $name: waiting for readiness (${waited}s / ${DAEMON_READY_SECS}s)"
  done
  return 124
}

# ------------------------------------------------------------------ main loop

selected=""
[ -n "${ONLY:-}" ] && selected=",${ONLY},"

say "supervisor up — pid $$, ${#stage_names[@]} stages, run_id=$RUN_ID, stall=${STALL_SECS}s"
write_state running ""

for name in "${stage_names[@]}"; do
  if [ -n "$selected" ] && [ "${selected#*,$name,}" = "$selected" ]; then
    ST_STATUS[$name]=skipped; continue
  fi

  # The done-probe runs BEFORE any work. This is what makes the supervisor safe to restart at any
  # moment, including after a pod reboot.
  if eval "${S_DONE[$name]}" 2>/dev/null; then
    say "$name: already complete (done-probe satisfied) — skipping"
    ST_STATUS[$name]=complete; ST_NOTE[$name]="satisfied on entry"
    write_state running "$name"; continue
  fi

  ST_STARTED[$name]="$(now)"
  attempt=1
  while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    ST_STATUS[$name]=running; ST_ATTEMPTS[$name]=$attempt
    write_state running "$name"
    say "$name: attempt $attempt/$MAX_ATTEMPTS — ${S_DESC[$name]}"

    if [ "${S_KIND[$name]}" = daemon ]; then
      run_daemon "$name" "${S_CMD[$name]}"; rc=$?
    else
      run_with_watchdog "$name" "${S_CMD[$name]}"; rc=$?
    fi

    # Exit code and completion are independent questions. A stage may exit 0 having produced nothing
    # (the corpus fetch did exactly this), and may exit non-zero having finished the part we need.
    # The probe is the authority; the exit code only colours the note.
    if eval "${S_DONE[$name]}" 2>/dev/null; then
      ST_STATUS[$name]=complete; ST_ENDED[$name]="$(now)"
      ST_NOTE[$name]="done-probe satisfied (exit $rc)"
      say "$name: COMPLETE"
      write_state running "$name"; break
    fi

    if [ "$rc" -eq 124 ]; then
      ST_NOTE[$name]="attempt $attempt stalled/timed out with no progress"
    else
      ST_NOTE[$name]="attempt $attempt exited $rc without satisfying the done-probe; see $STAGE_DIR/$name.log"
    fi
    say "$name: attempt $attempt failed — ${ST_NOTE[$name]}"
    attempt=$(( attempt + 1 ))
    [ "$attempt" -le "$MAX_ATTEMPTS" ] && sleep $(( 30 * attempt ))
  done

  if [ "${ST_STATUS[$name]}" != "complete" ]; then
    ST_STATUS[$name]=blocked; ST_ENDED[$name]="$(now)"
    say "BLOCKED at $name after $MAX_ATTEMPTS attempts. Stopping — later stages consume this one's"
    say "output, and running them now would measure the previous run and report it as this one."
    write_state blocked "$name"
    exit 1
  fi
done

say "all stages complete"
write_state complete ""
