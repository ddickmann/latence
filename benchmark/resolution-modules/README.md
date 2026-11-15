# The older resolution modules, measured

Ticket 50 of the semantic-sidecar map, under the owner's rule of 14 Nov: an older module of the
text path is kept only where a measurement shows that it adds value.

**The rule** (fixed in `measure.py` before the run): a module is kept when it raises pairwise F1
by at least 0.02 over the same run without it and its precision does not fall; it is retired
otherwise.

```bash
PYTHONHASHSEED=0 uv run python benchmark/resolution-modules/measure.py > benchmark/resolution-modules/results.json
```

| Module | Gold | Without it (P / R / F1) | With it (P / R / F1) | F1 gain | Verdict |
|---|---|---|---|---|---|
| `disambiguation.cascade` (the string cascade) | 684 mentions of 179 organisations | 0.946 / 0.265 / 0.414 | 0.754 / 0.304 / 0.433 | +0.019 | retire |
| `disambiguation.embedding`, hashing embedder | the same | 0.946 / 0.265 / 0.414 | 0.616 / 0.265 / 0.371 | −0.043 | retire (for this embedder) |
| Type Consolidation (`TypeCanonicalizer`, lexical rungs) | 65 type labels of 16 types | 1.000 / 0.242 / 0.390 | 1.000 / 0.879 / 0.936 | +0.546 | keep |

"Without it" is, for mentions, the cascade restricted to exact normalised surfaces (the
`ExactSurfaceDisambiguator`'s behaviour) and, for type labels, labels equal once lower-cased.
`results.json` holds every number, the pair counts and the seconds; a second run gives the same
scores.

- **`gold-mentions.json`**: synthetic English documents that name organisations with noisy
  surfaces (dropped legal form, abbreviation, acronym, stem only, typo, case) and homonyms that
  differ by city or industry; each mention with its true entity. Written by the semantic
  sidecar's generator (`generate_text.py`, seed 7, 150 entities, 500 documents). Template prose
  is easier than real text.
- **`gold-types.json`**: entity-type labels as free-form extractors write them (case, plural,
  British spelling, abbreviation, acronym, punctuation), one synonym per type that no lexical
  rule can see, and two pairs that must stay apart (product/production, part/party).

**Not measured here:** the GLinker linker (`disambiguation.glinker`) and the embedding
Disambiguator on a learned embedder both need a GPU. The verdict on the embedding Disambiguator
is for the hashing embedder its default resolves to; the learned run is a live request.
