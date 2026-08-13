# latence-relation-llm

An LLM-based **Relation Extraction** Provider for the
[Latence framework](../../README.md): given a document's entity mentions plus the
chunk text, it prompts an OpenAI-compatible chat model to emit typed, directed
relations between the mentions.

It registers under the `latence.providers` entry-point group as `relation.llm`, so
installing this package makes the Provider discoverable by any Pipeline with **no
change to `latence-core`** (ADR-0004). It is one of the two shipped Relation
Extraction candidates (ADR-0013 — the choice is empirical, per corpus); the other
is the deterministic, dependency-free `relation.pattern` that ships in core.

## Any OpenAI-compatible endpoint (incl. local Ollama)

The Provider talks the OpenAI chat-completions API, so the same code drives OpenAI,
vLLM, Together, or a laptop-local **Ollama** with only a `base_url` change — the
endpoint is the performance path (ADR-0007); nothing GPU is forced into first-run.

## Zero-shot

The relation types are supplied per Pipeline in the Stage `config`. No relation
label set is hardcoded. A relation naming an out-of-range mention, an unknown label,
or a self-reference is dropped, so a hallucinated reference never yields an
un-resolvable record.

## Licensing

This Provider is framework code (**PolyForm Noncommercial 1.0.0**, the repository-wide licence) and calls a user-supplied endpoint;
it bundles no model weights. The `openai` client is **Apache-2.0** (code). The key
and endpoint come from config/environment, never from code.

## Install

```bash
pip install latence-core latence-relation-llm
```

## Use in a Pipeline

```yaml
  - name: relations
    capability: relation_extraction
    provider: relation.llm
    depends_on: [chunk, entities]        # BOTH: the text/page-map AND the mentions
    config:
      labels: [works_for, located_in, founded_by]
      model: gpt-4o-mini
      base_url: http://localhost:11434/v1   # e.g. a local Ollama; omit for OpenAI
      temperature: 0.0
```
