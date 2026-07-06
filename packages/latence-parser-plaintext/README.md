# latence-parser-plaintext

A passthrough **Parser** Provider for the [Latence framework](../../README.md): it
turns `.txt`/`.md` source documents into markdown records, preserving Provenance and
Classification losslessly.

It registers itself under the `latence.providers` entry-point group as
`parser.plaintext`, so installing this package makes the Provider discoverable by any
Pipeline with **no change to `latence-core`** — the proof that Providers plug in
dynamically across package boundaries (ADR-0004).

## Install

```bash
pip install latence-core latence-parser-plaintext
```

## Use in a Pipeline

```yaml
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
```
