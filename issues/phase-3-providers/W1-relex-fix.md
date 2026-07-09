# W1-relex FIX — `relation.gliner_relex` was a FALSE GREEN (wrong real API); make it real

**Pod de-risk (2026-07-09, part of the consolidated gpu-sota sweep) proved `relation.gliner_relex`
NEVER worked with real weights.** The provider calls `model.predict_with_relations(...)` and parses
the result as a **dict** (`result.get("entities")` / `result.get("relations")`). The real
`knowledgator/gliner-relex-multi-v1.0` model (`gliner` 0.2.27) has **no such method** — it raises
`AttributeError: 'UniEncoderTokenRelexGLiNER' object has no attribute 'predict_with_relations'`. The
offline fake (`tests/test_gliner_relex_provider.py`) defined a matching `predict_with_relations`
returning a dict, so every offline test passed against a stub that mirrored a **non-existent** API —
the exact PaddleOCR false-green class (the anti-false-green rule: the stub MUST mirror the REAL model).

## The REAL API (introspected + run on the pod — pin to this)
```
GLiNER.from_pretrained("knowledgator/gliner-relex-multi-v1.0").predict_relations(
    text: str, labels: List[str], relations: List[str],
    flat_ner=True, threshold=0.5, adjacency_threshold=None,
    relation_threshold=None, multi_label=False,
) -> Tuple[List[entity_dict], List[relation_dict]]      # a TUPLE, not a dict
```
- `entity_dict`  = `{"start": int, "end": int, "text": str, "label": str, "score": float}`
  (chunk-local offsets — identical shape to `predict_entities`).
- `relation_dict`= `{"head": {"start","end","text","type","entity_idx"},
                     "tail": {"start","end","text","type","entity_idx"},
                     "relation": str, "score": float}`.

**Verified real output** for `"IBM works with Alice Johnson in Berlin. Contact her at
alice.johnson@example.com or call +1-202-555-0143 about the Acme partnership."` with
labels `[organization, person]`, relations `[works for, partner of]`, threshold 0.3:
entities = IBM(org 0.995), Alice Johnson(person 0.997), Acme(org 0.967); relations =
Alice Johnson **works for** IBM (0.96), IBM **partner of** Acme (0.79). Clean, correct, offsets exact.

## What to fix
1. **`provider.py::_extract_one`** — replace the call + parse:
   ```python
   raw_entities, raw_relations = model.predict_relations(
       chunk.content, self._labels, self._relation_labels, threshold=self._threshold,
   )
   ```
   Everything downstream is ALREADY correct against the real shape and MUST be kept: entity dicts use
   `start/end/label/score`; `_resolve_endpoint` keys head/tail by `(start,end)` (both present in the
   real relation dicts); `_relation_sort_key` reads `head["start"]`; offsets resolve through
   `OffsetIndex.resolve_span` (the same path that made `entity.gliner` pass its pod probe with exact
   round-trip). Do NOT re-architect — this is a two-line call/parse correction plus a faithful stub.
2. **The offline fake (`tests/`) MUST become faithful** — a `predict_relations(text, labels,
   relations, threshold=..., **kwargs)` that returns a **`(entities, relations)` TUPLE** of the real
   dict shapes above (head/tail with `start/end/text/type/entity_idx`, `relation`, `score`). Delete
   the old `predict_with_relations` dict fake. **A provider still calling `predict_with_relations` or
   doing `result.get(...)` on a tuple MUST FAIL** — add an anchor test that pins this (the false-green
   trap): feeding the faithful tuple-returning fake, assert ≥1 mention AND the expected relation with
   head/tail resolved to real mention record_ids; a dict-parsing provider raises/returns empty.
3. **Provider-level guard (teeth, like GLM #126):** a test that asserts the emitted `RelationMention`s
   have real endpoints (head/tail record_ids that exist among the emitted mentions) and a non-empty
   label — so a regression that silently drops all relations (e.g. mis-parsing the tuple) FAILS, not
   just the fake-shape anchor.
4. **Fix the `gpu-sota.yaml` wiring (it's broken for this stage):** the `relations` stage currently
   passes `config: {device: cuda, relations: [works_for, partner_of]}` — but the provider requires
   `labels` AND `relation_labels` (a missing-key `ConfigError` at load). Correct it to:
   ```yaml
   config:
     device: cuda
     labels: [organization, person]
     relation_labels: [works for, partner of]
   ```
   Also lower-case the `entities` stage labels (`[ORG, PERSON]` → `[organization, person]`): GLiNER
   works best with lower/title case — the pod probe scored 0.95–0.99 lower-case. (CPU stack-validate
   skips the cuda stages with-flag, so this wiring bug is invisible until a real GPU run — note it.)

## Definition of Done
- `relation.gliner_relex` uses the real `predict_relations` tuple API; the offline fake is faithful to
  it; the un-fixed dict-parse path FAILS an anchor test; a provider-level guard asserts real endpoints.
- `bash scripts/verify-local.sh` green (ruff·mypy-strict·pytest·conformance·stack-validate·bake-off;
  roster unchanged). Evidence-bearing Apache-2.0 license unchanged (still verified).
- ADR-0036 note: the pod finding (predict_with_relations was never real) + the faithful-stub rule.
- `scripts/validate-relex-on-gpu.md`: turnkey pod steps (real `predict_relations` run + the sentence
  above → the two expected relations). Real GPU numbers come from my SSH re-validation, never fabricated.

## Reviewer should check
- The stub is faithful to the REAL `predict_relations` TUPLE contract (un-fixed provider FAILS) — the
  false-green trap that this whole fix exists to close.
- The offset path is UNCHANGED (resolve_span) — it was already correct (entity.gliner proved it); only
  the model call + result-unpack change.
- `gpu-sota.yaml` relations stage now has valid `labels` + `relation_labels`; entities stage lower-cased.
- Determinism (sorted entities/relations, stable record_ids); relations with unresolved endpoints dropped.
