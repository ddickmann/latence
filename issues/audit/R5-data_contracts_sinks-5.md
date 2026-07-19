# R5 / data_contracts_sinks-5 — the Qdrant CLI's most likely first-run failure escapes as a raw traceback

**Severity:** major
**Stream:** feat/release-fleet-integration (#178–#199), ticket #183
**Sites:**
`packages/latence-sink-qdrant/src/latence_sink_qdrant/cli.py:125-145` (`main`'s `except SinkError`),
`packages/latence-sink-qdrant/src/latence_sink_qdrant/sink.py:747-756` (`_import_qdrant` raising a
bare `ImportError`),
`packages/latence-sink-qdrant/src/latence_sink_qdrant/cli.py:95-106` (`_load_config_file`)

## What is wrong

`main()` catches only `SinkError`. Three real, reachable failures are **not** `SinkError`, so they
propagate out of `main` and the process dies with a native Python traceback printing absolute
filesystem paths, at exit code `1` (or `2`-by-accident-of-argparse, never the documented typed
exit `2`):

1. **The optional `qdrant-client` extra is not installed.** `latence-sink-qdrant/pyproject.toml`
   declares `dependencies = ["latence-core"]` with `client = ["qdrant-client>=1.10"]` as an
   *extra*, while the `latence-sink-qdrant` console script ships with the **base** install. So the
   single most likely first-run failure is exactly this one, and `sink.py:_import_qdrant` converts
   the native `ImportError` into… another bare `ImportError`.
2. `--config /missing.yaml` → raw `FileNotFoundError` from `Path.read_text`.
3. `--config bad.yaml` (malformed) → raw `yaml.parser.ParserError` / `json.JSONDecodeError`.

This violates **ADR-0034** ("failures surface as TYPED errors, never a raw native traceback"), and
contradicts two of the package's own written claims: `cli.main`'s docstring ("never raises a sink
error at the user") and `errors.py`'s ("a caller catches one named type instead of an opaque
`qdrant_client` exception"). The sibling #184 lane already does it right —
`latence_core/iceberg.py:_catalog()` turns the identical `pyiceberg` `ImportError` into a typed
`ConfigError` — so this is an inconsistency, not a design choice.

## Failure scenario

Reproduced live in the repo's own dev venv (`.venv`, which has no `qdrant_client`):

```
$ .venv/bin/python -c "from latence_sink_qdrant.cli import main; \
    print(main(['verify-absent','doc-7#chunk-0','--collection','corpus','--url','http://localhost:6333']))"
Traceback (most recent call last):
  ...
  File "/Users/dennisdickmann/Developer/latence-framework/packages/latence-sink-qdrant/src/latence_sink_qdrant/sink.py", line 755, in _import_qdrant
    raise ImportError(msg) from exc
ImportError: The 'qdrant-client' package is required to load into a live Qdrant. ...
```

Same class of escape, also reproduced:

```
--config /nonexistent.yaml  → FileNotFoundError: [Errno 2] No such file or directory: ...
--config bad.yaml           → yaml.parser.ParserError: while parsing a flow node ...
```

## Why the green gate cannot see it

The raising line carries `# pragma: no cover - exercised only without the dep`, the CI environment
installs the `[client]` extra, and no CLI test drives a **non-`SinkError`** failure: every existing
exit-2 test raises a `SinkError` from a stubbed sink.

## What to build

1. `sink.py:_import_qdrant` — raise the typed `SinkConfigError` (a `SinkError`) instead of a bare
   `ImportError`, chaining the native cause and keeping the actionable install hint verbatim. The
   library contract in `errors.py` then actually holds for every caller, not only the CLI.
2. `cli.py:_load_config_file` — convert `OSError` (missing/unreadable file) and the YAML/JSON parse
   errors into `SinkConfigError` naming the path and the reason. No raw parser text as the only
   message, no swallowed exception (chain with `from exc`).
3. Regression tests that drive `cli.main(...)` for all three paths.

## Acceptance criteria

* `cli.main([...])` **returns** `2` and prints a single `error: …` line to stderr — never a
  traceback — when (a) `qdrant_client` is not importable, (b) `--config` points at a missing file,
  (c) `--config` points at a malformed YAML/JSON file.
* The missing-dependency error message still names the remedy
  (`pip install 'latence-sink-qdrant[client]'`) and the native `ImportError` remains the
  `__cause__`.
* `QdrantSink._import_qdrant` raises a `SinkError` subclass — the typed contract holds for library
  callers, not only the CLI.
* The tests are non-vacuous: proven to FAIL on the pre-fix code with the raw
  `ImportError` / `FileNotFoundError` / `ParserError` escaping `main`.
* `make test` stays green at ≥ 2985 passed / 28 skipped. No test weakened or deleted.
