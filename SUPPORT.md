# Getting help

This project is maintained by one person. Support is **best-effort, with no SLA** — there is
no commercial support offering and no contractual response time. That is stated plainly here
so you can plan around it rather than discover it.

Pick the right channel and you will get a faster, better answer.

## Where to go

| You want to… | Go here |
| --- | --- |
| Ask how to do something, or sanity-check a design | **[Discussions](https://github.com/ddickmann/latence-framework/discussions)** |
| Report something that behaves incorrectly | **Bug report** issue template |
| Propose a feature, a Stage behaviour, or a contract change | **Feature request** issue template |
| Propose a new Provider (a model, endpoint, or cloud service) | **Provider request** issue template — it carries the licence-verification fields that gate a Provider |
| Report a suspected security vulnerability | **Privately.** [`SECURITY.md`](SECURITY.md) — never a public issue |
| Contribute code | [`CONTRIBUTING.md`](CONTRIBUTING.md) — read the CLA requirement and the gates first |

Please search existing issues and the Decision Log in `docs/adr/` before filing. A
surprising amount of "why does it work this way" already has a written answer with the
alternatives that were rejected.

## What makes a question answerable

- **The exact command and the exact output.** Not a paraphrase.
- **A minimal reproduction.** A trimmed Pipeline or stack YAML beats a description of one.
- **Your environment**: OS, Python version, `uv --version`, the framework commit, and whether
  you are on a CPU or GPU host. A surprising number of reports resolve to "the heavy Provider
  package was not installed" or "this is a GPU-only Provider on a CPU host".
- **No secrets and no real PII.** Redact before pasting. If the problem *is* about PII
  handling, describe the shape of the data rather than including any.

If the failure is order- or timing-dependent, say so — the suite is seeded and hash-stable
(`PYTHONHASHSEED=0`), so nondeterminism is a bug worth reporting on its own.

## Documentation

- **`README.md`** — what the framework is and the fastest path to a running demo.
- **`CONTEXT.md`** — the canonical glossary. If a term in an answer looks like jargon, it is
  probably defined there, precisely.
- **`docs/`** — the deeper material: deployment, tuning, running a stack or a bake-off, the
  threat model, the retrieval input spec, GPU and compatibility notes.
- **`docs/adr/`** — the Decision Log: why each seam is shaped the way it is, and what was
  rejected.

## What this project does not offer

- No paid support tier, no hosted service, no consulting attached to the open-source code.
- No maintained release branches, no LTS, and no backports — `main` and the latest release
  are what receive fixes ([`SECURITY.md`](SECURITY.md)).
- No guaranteed response time, including for security reports, where the stated windows are
  explicitly labelled targets rather than commitments.

The maintainership model, the bus factor, and what happens to adopters if the project stops
being maintained are set out honestly in [`docs/CONTINUITY.md`](docs/CONTINUITY.md). If you
are evaluating this for production, read that before you read anything else.
