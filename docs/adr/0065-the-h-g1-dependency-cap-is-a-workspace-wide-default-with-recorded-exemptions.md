# The H-G1 dependency cap is a workspace-wide default, with recorded exemptions

**Status:** Accepted (2026-08-25)

Every third-party requirement declared by a package under `packages/` carries an **upper bound** —
in `[project].dependencies` and in every `[project.optional-dependencies]` extra alike — unless the
pair `(package, distribution)` appears in the exemption table of
[`packages/latence-core/tests/test_dependency_caps.py`](https://github.com/ddickmann/latence/blob/main/packages/latence-core/tests/test_dependency_caps.py)
with a reason. That test is the rule; the manifest comments are commentary on it.

The scope question this decision settles is the one the convention had left open: the cap is **not**
a rule about heavy model libraries. It is a **default** that applies to every third-party pin, and
the model libraries are simply where it was applied first.

## Context

The launch-hygiene review asked for "conservative upper caps … on the model-lib deps in the provider packages that import them, so a
downstream install can't silently pull an API-breaking major", and explicitly said to "leave the
stable libs (pydantic/pyarrow/fsspec) as-is". Eleven manifests grew a comment saying exactly that.
Nothing read those comments.

A survey of the 34 package manifests found **25 uncapped third-party pins across 10 packages** in
`[project].dependencies` — and 17 more in extras, which the survey had not thought to look at.
**42 uncapped pins across 14 packages**, against 26 capped ones. The convention had not merely been
applied unevenly; it had already failed at its stated job, and the lock files say so:

| Pin | Resolved to | The crossing |
| --- | --- | --- |
| `pypdfium2>=4.0` | `5.12.1` (`deploy/env/*/requirements.lock`) | major 4 → 5 |
| `neo4j>=5.0` | `6.2.0` (`uv.lock`) | major 5 → 6 |
| `numpy>=1.24` | `2.4.6` (`uv.lock`) | major 1 → 2 |
| `apache-airflow>=2.7` | `3.3.0` (`uv.lock`) | major 2 → 3 |
| `python-pptx>=0.6.23` | `1.0.2` (`uv.lock`) | 0.x → 1.0 |

Five silent major crossings, none of them a model library, every one of them the precise event the
comment claims the convention prevents. A convention that only fires where somebody remembered to
type it is not a convention; ADR-0064 landing `protobuf`/`sentencepiece` caps in `latence-gliner25`
only because a reviewer happened to look is the same finding from the other direction.

## Decision

**1. The default is: cap it.** Every third-party requirement in every `packages/*/pyproject.toml`
carries an upper bound. Extras are in scope because `pip install latence-retrieval[graph-neo4j]` is a
downstream install with identical exposure; a gate that covered `dependencies` alone would report
green over nine uncapped pins.

**2. Exemptions are named, reasoned, and machine-read.** They live in the `EXEMPT` table of the
gate, keyed by `(package, distribution)` — not by distribution alone, because `transformers` is
correctly capped in `latence-splade` and `latence-disambig-glinker` and correctly uncapped in the two
in-process VLM Parsers. The gate checks the table in **both** directions: an uncapped pin missing
from it fails, and an entry whose pin has since been capped or deleted also fails, so an exemption
cannot outlive the reason it was written for.

**3. The cap sits at the boundary the distribution's own history supports, never at an invented
one.** For a `>=1.0` distribution that is the next major. For a `0.x` distribution it is `1.0` —
the one break every `0.x` library signals — *unless* this repository has recorded evidence that the
library breaks across `0.x` minors, which is why `gliner` is `<0.3`, `glinker` is `<0.2` and
`flashdeberta` is `<0.1` while `pdfplumber` and `peft` are `<1`. H-G1's closing instruction, "if a
cap would be guesswork about the next major's compatibility, note it and skip rather than pin
arbitrarily", is preserved exactly: the note is now an `EXEMPT` entry instead of a comment.

The five exemptions this lands with:

| Package | Distribution | Why |
| --- | --- | --- |
| `latence-core` | `fsspec` | CalVer (`2026.6.0`). The leading component is a year; `<N` would encode a date. |
| `latence-core` | `pyarrow` | The `>=15.0` floor has spanned 15 → 24 → 25 with no source change, over the stable Table/Parquet surface. A cap in the near-zero-dep base (ADR-0016) every adopter installs beside their own Arrow stack costs a real resolution conflict to buy an unobserved boundary. |
| `latence-core` | `packaging` | Same shape (23 → 26), sharper cost: the ecosystem's most ubiquitous build dependency, reached through one lazy import. A cap conflicts with pip/build/setuptools rather than protecting anything. |
| `latence-parser-lighton` | `transformers` | The compatible major is genuinely unsettled — see below. |
| `latence-parser-glm` | `transformers` | Same unsettled split, same named refusal. |

**4. `transformers` in the two in-process VLM Parsers stays uncapped, and that is load-bearing.**
`deploy/env/gpu-learned/environment.yaml` records that the `>=5` floor for LightOnOCR comes from the
issue #198 pod field evidence and "NOT from independently verified package metadata", while
`deploy/env/served-ocr-client/requirements.lock` resolves the same package next to
`transformers==4.57.6`. `<5` would contradict the field evidence; `<6` would invent a boundary
nobody has tested. That spec's `excluded_packages` block names this uncapped pin **three times** as
the reason the refusal has to be by name — "the incompatibility lives in the model code, not the
package metadata" — so capping it would not merely guess, it would falsify a recorded reason and
turn a resolution-time refusal into decoration.

**5. This narrows H-G1 §3's "leave the stable libs as-is", and says so.** `pydantic`, `pyyaml` and
`typer` are now capped. `pydantic` earns it on the merits rather than by category: the entire
contract spine — `contracts.py`, `SCHEMA_VERSION`, the validation at every Stage boundary — is
written against the 2.x model API, 2 → 3 is a signalled break, and `<3` is what essentially every
pydantic consumer already declares, so it introduces no conflict an adopter does not already have.
Of the three distributions H-G1 named, the two whose "stable" claim survives contact with the
evidence — `pyarrow` and `fsspec` — are exempt, on that evidence rather than on the label.

## Alternatives considered

**A gate that knows the heavy model libraries** (`gliner`/`gliner2`/`glinker`/`flashdeberta`/
`transformers`/`torch`) and checks only those. **Rejected: it fails open on exactly the case that
produced this ticket.** Every one of the five silent major crossings above involves a distribution
that list does not contain, so the gate would have shipped green over all of them, and would go on
passing vacuously for the next dependency somebody adds. A gate whose coverage has to be extended by
hand every time the workspace grows is the comment-only convention again, wearing a test's clothes.
The default-cap form fails *closed*: a new uncapped pin is red until somebody caps it or writes down
why they cannot.

**Cap everything, no exemptions at all.** Rejected as dishonest in two directions. It would force a
`<5` or `<6` on `transformers` in the VLM Parsers — a fabricated compatibility claim, and one that
contradicts a recorded field observation — and it would force a calendar date onto CalVer `fsspec`.
An exemption with a checkable reason is the honest encoding of "we do not know"; an invented cap is
not.

**Keep it a convention and add it to the review checklist.** Rejected on the evidence: it *was* a
convention, it *was* documented in eleven manifests, and it still admitted five silent major
crossings and 42 uncapped pins. This repository's standing answer to a claim nobody checks is
to make a test check it (AGENTS.md § *Six things that will burn you*, item 4); nothing about this
claim justifies an exception.

**Enforce it against a lock file instead of the manifests** — assert the resolved version still
matches the declared floor. Rejected: the lock is a *consequence* of the manifests, so the check
would fire only after a bad resolution had already happened, which is precisely the runtime surprise
the cap exists to convert into a resolution-time refusal. The manifest is where the claim lives, so
the manifest is where the gate reads.

## Consequences

**37 pins across 14 packages** gained an upper bound, leaving the workspace at 63 capped
third-party pins and the 5 exemptions above. `deploy/env/*/requirements.in` and `constraints.txt`
are regenerated from the manifests by `latence env render`, and `latence env verify` proves all
four environment locks still satisfy every new cap — no cap sits below a version this workspace has
already resolved.

Three places outside the manifests restated a pin and had to follow it: the `served-ocr-client`
`pillow` constraint (whose reason said in prose that there was deliberately no upper cap),
`deploy/Dockerfile`'s hand-written light-dep install line (the two OCR-client packages go in
`--no-deps`, so the image would otherwise have installed the uncapped resolution the packages now
refuse), and four `test_envspec.py` fixtures that inject drift by string-replacing a pin. Three of
those fixtures failed loudly on the cap; the fourth asserts an *absence* and so had gone silently
vacuous — worth recording, because it is the failure mode this ADR is about, one layer down.

The recurring cost is real and small: a genuine major bump now needs a one-line manifest edit with
the lock as its evidence, rather than arriving unannounced in an adopter's `pip install`. The
exemption table is the pressure valve, and it is deliberately uncomfortable to use — it wants a
reason a reviewer can check, and the gate deletes it the moment it stops being true.
