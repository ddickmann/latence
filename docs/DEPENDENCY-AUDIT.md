# Dependency-completeness audit

**Purpose.** Prove that every package under `packages/` declares every third-party
runtime dependency it actually imports, so a clean `pip install <package>` (with
its resolver, no `--no-deps`) on a fresh machine never hits a `ModuleNotFoundError`.
This is the audit that would have caught the `peft` gap the pod validation found
(gliner2's inference engine imports `peft`, which `gliner2[local]` does not declare —
fixed in commit `cdbb8c2`, before this audit).

**Method (repeatable).** For each `packages/<pkg>/src/`, every top-level third-party
module name is parsed out of the actual `import X` / `from X import …` statements
(vendored `_vendor/` trees and relative imports excluded), mapped import-name →
distribution-name where they differ (`yaml`→`pyyaml`, `PIL`→`pillow`, `docx`→
`python-docx`, `pptx`→`python-pptx`, `sentence_transformers`→`sentence-transformers`,
`presidio_analyzer`→`presidio-analyzer`, `torch_geometric`→`torch-geometric`), then
checked against that package's `[project].dependencies` + `[project.optional-dependencies]`,
counting a dep as **covered** if it is declared directly, is a documented optional
extra guarding a lazy import, or is a transitive dep of a declared `latence-*` sibling
(`latence-core` declares `pydantic`, `fsspec`, `pyarrow`, `pyyaml`, `typer` — packages
that depend on core inherit those and need not re-declare them). Script:
`scripts/dep_audit.py` (re-run to regenerate this table).

**Guarded vs unguarded.** A *module-level* import must be a hard-declared dependency.
A *nested* (inside-a-function) import is a lazy/optional load and is legitimately
covered by an optional extra with a graceful `ImportError` message — that is the
established pattern here (`airflow`, the ULTRA/PyKEEN heavy stack), not a gap.

## Result

**No genuinely-undeclared runtime dependency was found.** Every module-level
third-party import is covered by a declared dep or a `latence-core`-transitive dep.
The two items the scan flags as "not directly declared" are both **guarded (lazy)
imports that are intentional, documented design** (see notes below), not the `peft`
class of bug. The `peft` gap itself is already fixed on `wave-integration`.

## Per-package table

Covered-by: **declared** = in this package's `dependencies`; **core** = transitive via
`latence-core`; **extra[x]** = declared under optional-dependency group `x`, import is
lazy/guarded.

| Package | Third-party import | Distribution | Module-level? | Covered by |
|---|---|---|---|---|
| latence-core | `fsspec` | fsspec | yes | declared |
| latence-core | `pyarrow` | pyarrow | yes | declared |
| latence-core | `pydantic` | pydantic | yes | declared |
| latence-core | `typer` | typer | yes | declared |
| latence-core | `yaml` | pyyaml | yes | declared |
| latence-core | `torch` | torch | **no (lazy)** | intentionally undeclared — near-zero-core (ADR-0016); see note 1 |
| latence-demo | `pydantic` | pydantic | yes | core |
| latence-demo | `typer` | typer | yes | core |
| latence-disambig-embedding | *(none)* | — | — | pure `latence-core` consumer |
| latence-embedder-endpoint | `openai` | openai | lazy | declared |
| latence-embedder-st | `sentence_transformers` | sentence-transformers | lazy | declared |
| latence-extract-gliner2 | `gliner2` | gliner2 | lazy | declared (+ `peft` runtime dep, note 3) |
| latence-linkpred-ultra | `easydict` | easydict | lazy | extra[ultra] — **LGPL flag, note 4** |
| latence-linkpred-ultra | `pykeen` | pykeen | lazy | extra[pykeen] |
| latence-linkpred-ultra | `torch` | torch | lazy | extra[ultra], extra[pykeen] |
| latence-linkpred-ultra | `torch_geometric` | torch-geometric | lazy | not a plain PyPI pin — PyG wheel index, note 2 |
| latence-ner-endpoint | `openai` | openai | lazy | declared |
| latence-ner-gliner | `gliner` | gliner | lazy | declared |
| latence-parser-document | `pypdf` | pypdf | lazy | declared |
| latence-parser-endpoint | `openai` | openai | lazy | declared |
| latence-parser-glm | `PIL` | pillow | lazy | declared |
| latence-parser-glm | `pypdfium2` | pypdfium2 | lazy | declared |
| latence-parser-glm | `transformers` | transformers | lazy | declared |
| latence-parser-lighton | `PIL` | pillow | lazy | declared |
| latence-parser-lighton | `pypdfium2` | pypdfium2 | lazy | declared |
| latence-parser-lighton | `transformers` | transformers | lazy | declared |
| latence-parser-lighton-vllm | `openai` | openai | lazy | declared (+ latence-parser-lighton) |
| latence-parser-pdfplumber | `pdfplumber` | pdfplumber | lazy | declared |
| latence-parser-plaintext | *(none)* | — | — | pure `latence-core` consumer |
| latence-parser-render | `PIL` | pillow | lazy | declared |
| latence-parser-render | `docx` | python-docx | lazy | declared |
| latence-parser-render | `openpyxl` | openpyxl | lazy | declared |
| latence-parser-render | `pptx` | python-pptx | lazy | declared |
| latence-parser-render | `pypdfium2` | pypdfium2 | lazy | declared |
| latence-pii-gliner | `gliner` | gliner | lazy | declared |
| latence-pii-gliner2 | `gliner2` | gliner2 | lazy | declared (+ `peft` runtime dep, note 3) |
| latence-pii-presidio | `presidio_analyzer` | presidio-analyzer | lazy | declared (`spacy` is a presidio transitive dep) |
| latence-relation-gliner | `gliner` | gliner | lazy | declared |
| latence-relation-llm | `openai` | openai | lazy | declared |
| latence-runner-airflow | `airflow` | apache-airflow | **no (lazy)** | extra[airflow], note 2 |
| latence-schema-inducer | `openai` | openai | lazy | declared |

## Notes

1. **`latence-core` → `torch` (lazy, undeclared by design).** Imported only inside
   functions in `providers/perf.py` (dtype materialisation, `torch.compile`) and
   `tune/fingerprint.py` (version read), each with a `noqa: PLC0415` "optional heavy
   dep" marker and a graceful fallback. Core is the near-zero-dep spine (ADR-0016);
   declaring `torch` would defeat that. Not a gap.

2. **Deliberately-non-PyPI heavy deps.** `latence-runner-airflow` guards `from airflow …`
   inside `build_dag()` with an `ImportError` pointing at the `[airflow]` extra
   (`apache-airflow>=2.7`). `latence-linkpred-ultra` guards the whole ULTRA/PyKEEN stack;
   `torch-geometric` + `torch-scatter` are intentionally **not** declarable as plain PyPI
   pins — they must be installed from the PyG wheel index matched to the installed
   torch/CUDA (`torch-scatter` has no universal wheel and fails to build without torch
   present). The package's `[project.optional-dependencies]` comment documents the exact
   `-f https://data.pyg.org/whl/…` install line. Not a gap.

3. **`peft` (already fixed, the reference case).** `gliner2`'s relation-inference engine
   (`gliner2.inference.engine → gliner2.training.trainer`) imports `peft` at runtime, but
   `gliner2[local]` does not declare it — a real load fails `ModuleNotFoundError: peft`.
   `peft>=0.10` is declared on **both** `latence-extract-gliner2` and `latence-pii-gliner2`
   (commit `cdbb8c2`). This is the *hidden-transitive-import* class — invisible to a src
   import scan, only surfaced by loading real weights on the pod. No other package pulls a
   heavy dep with an equivalently-incomplete extra: `gliner` (v1), `openai`,
   `sentence-transformers`, `presidio-analyzer`, `pypdf`, `pdfplumber`, `pypdfium2`,
   `transformers`, `pillow`, `python-docx`, `python-pptx`, `openpyxl` all declare their own
   transitive runtime deps correctly via PyPI metadata.

4. **License flag — `easydict` is LGPL-3.0.** `easydict` (pulled only by the opt-in
   `latence-linkpred-ultra[ultra]` extra, used by the vendored MIT ULTRA inference subset)
   is **LGPL-3.0** on PyPI, not permissive in the MIT/BSD/Apache class. LGPL-3.0 is weak
   copyleft: dynamic import by permissively-licensed code (as here — pure-Python, no
   modification of easydict) is permitted without relicensing the framework, and the extra
   is opt-in and off the default install path. Flagged per ADR-0012 / research-diligence so
   the maintainer decides consciously (options: keep behind the opt-in extra as-is with this
   note, or replace easydict's trivial dotted-dict usage in the vendored ULTRA config with a
   permissive shim). The other ULTRA/PyKEEN extras are permissive: `torch` BSD-3-Clause,
   `torch-geometric` MIT, `torch-scatter` MIT, `pykeen` MIT (verified 2026-07-10).

## Reproduce

```
python3 scripts/dep_audit.py          # regenerate the per-package table
```
