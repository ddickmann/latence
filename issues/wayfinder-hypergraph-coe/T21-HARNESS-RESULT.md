# Ticket 21 — labeling harness: BUILT

The AFK half of ticket 21 (parts 1 + 3) is built and green. The HITL half (part 2, the
labeling itself) is yours — this file is the usage guide.

## Where it lives

`packages/latence-benchmark/src/latence_benchmark/labeling/` — new module (goldset.py,
calibration.py, server.py, ui.py, cli.py) + `packages/latence-benchmark/tests/test_labeling.py`.
One new console entry registered in the package's pyproject: **`latence-label`**. No new
dependencies (stdlib `http.server` + the already-present pyarrow via latence-core).

## Starting a session

**Gold-set mode** (~100 multi-hop questions over an exported corpus):

```bash
uv run latence-label gold --export-dir <dir-with-records.jsonl-or-.parquet> --out issues/wayfinder-hypergraph-coe/goldset/enterprise-gold.jsonl
```

Opens `http://127.0.0.1:8765/` in your browser (`--no-browser` to suppress, `--port N` to
change). Flow: pick a document in the left sidebar (`/` filters), **select text with the
mouse** in a chunk → the span is captured with exact char offsets, fill question + hop count
on the right, `Ctrl+Enter` saves. The header shows live progress + hop distribution and
turns orange when the 1-hop share exceeds 40%. Ctrl+C stops the server any time — the file
is append-only; rerunning the same command resumes with everything already labeled.

**Calibration mode** (~200 Hyperedge γ judgments, ticket 09):

```bash
uv run latence-label calibrate --hyperedges <export>/graph-hyperedges.parquet --out issues/wayfinder-hypergraph-coe/goldset/gamma-judgments.jsonl
```

One Hyperedge card at a time (tau_text + entities + relations + γ/support/arity). Keys:
`1`=correct, `2`=incorrect, `3`=partial (saves + advances), `n`=note, arrows navigate.
Resumes at the first unjudged row. Re-judging appends; the loader keeps the last word.

**Offline checks** (no server):

```bash
uv run latence-label stats --gold <gold.jsonl>                       # coverage + 1-hop guard
uv run latence-label validate --export-dir <dir> --gold <gold.jsonl> # ticket-17 gate, exit 0/1
# optionally also: --hyperedges <parquet> --calibration <judgments.jsonl>
```

## Output formats (append-only JSONL, one object per line)

Gold: `{"question_id": "q-<sha256(question)[:16]>", "question", "hop_count",
"supporting_facts": [{"document_id", "chunk_record_id", "char_start", "char_end", "text"}]}`.
Offsets are into the chunk's **exported** corpus text (`masked_content` when present, else
`content`); every span is verified `chunk_text[char_start:char_end] == text` at write time
(400 in the UI, exception in the API) and again by `validate`.

Calibration: `{"hyperedge_id", "judgment": "correct"|"incorrect"|"partial", "note"?}`.

## Tests + gates

- 22 new tests in `test_labeling.py`; full package suite: **196 passed** via
  `PYTHONHASHSEED=0 uv run pytest packages/latence-benchmark/tests/ -q`.
- `uv run ruff check packages` clean; `uv run mypy --strict` on the labeling module clean
  (latence-benchmark is not in the Makefile's `ALL_PKGS` strict set, so it is fully
  annotated but not CI-gated).

## Deviations / notes

- **Fact-count rule:** the ticket says "mark ≥2 spans", the coverage guard implies 1-hop
  questions exist. Implemented: hop_count ≥ 2 requires ≥ 2 facts; a 1-hop question may
  carry a single fact. Adjust in `goldset._required_fact_count` if you want strict ≥2.
- **Duplicate question text** is refused with an explicit message (identical text hashes to
  the same `question_id`); a correction is a resubmit with `replace` (API) — in the UI just
  reword, or fix the JSONL line by appending (last occurrence wins on load).
- Span marking supports selections **within one chunk** (a supporting fact never crosses a
  chunk boundary by design); multi-chunk facts = mark one span per chunk.
- The server binds 127.0.0.1 only. HTML mode is the only interface (terminal span-marking
  was skipped as per the ticket's own reasoning); `stats`/`validate` cover headless needs.
