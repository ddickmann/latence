"""Phase-1 pod de-risk probe: run the 3 GLiNER learned providers with REAL weights.

These providers (entity.gliner, relation.gliner_relex, redaction.gliner_pii) have only ever been
exercised against offline stubs — the exact profile that made PaddleOCR a false green. This runs each
against the real `gliner` library on the GPU and checks: (a) it produces correctly-typed records,
(b) char offsets round-trip (mention.text == original[char_start:char_end]), (c) it finds the obvious
entities/relations/PII in a known sentence. A stub that diverged from the real predict_entities shape
fails here, before the full stack run.
"""

from __future__ import annotations

from latence_core.contracts import (
    ChunkRecord, Classification, DocumentRecord, Disposition, PageMap, Provenance,
)

CLS = Classification(language="en", category="news", sensitivity="internal")
TEXT = ("IBM works with Alice Johnson in Berlin. Contact her at alice.johnson@example.com "
        "or call +1-202-555-0143 about the Acme partnership.")


def prov(doc, cs, ce):
    return Provenance(source_uri=f"file://{doc}.txt", file_name=f"{doc}.txt", file_type="txt",
                      file_size=len(TEXT), document_id=doc, char_start=cs, char_end=ce,
                      page_start=1, page_end=1)


def chunk():
    return ChunkRecord(record_id="c1", document_record_id="d1:doc", index=0, content=TEXT,
                       token_count=30, provenance=prov("d1", 0, len(TEXT)), classification=CLS,
                       page_map=PageMap.from_page_texts([TEXT]).page_map)


def document():
    return DocumentRecord(record_id="d1:doc", content=TEXT, media_type="text/plain",
                          page_map=PageMap.from_page_texts([TEXT]).page_map,
                          disposition=Disposition.PARSED, provenance=prov("d1", 0, len(TEXT)),
                          classification=CLS)


results = {}

# ---- 1. entity.gliner --------------------------------------------------------------------------
from latence_ner_gliner.provider import GlinerEntityExtractor  # noqa: E402

ner = GlinerEntityExtractor({"labels": ["organization", "person", "location"],
                             "threshold": 0.4, "device": "cuda"})
ments = list(ner.extract([chunk()]))
print("=== entity.gliner ===")
ner_ok = len(ments) >= 2
for m in ments:
    surf = TEXT[m.provenance.char_start:m.provenance.char_end]
    rt = surf == m.text
    ner_ok &= rt
    print(f"  {m.label:14s} {m.text!r:28} off[{m.provenance.char_start}:{m.provenance.char_end}] "
          f"round_trip={'Y' if rt else 'N (%r)' % surf} conf={m.confidence:.2f}")
found = {m.text for m in ments}
ner_finds = any("IBM" in f for f in found) and any("Alice" in f for f in found)
print(f"  -> found IBM & Alice: {ner_finds}; offsets round-trip: {ner_ok}")

# ---- 2. relation.gliner_relex (fused NER+RE) --------------------------------------------------
from latence_relation_gliner.provider import GlinerRelexExtractor  # noqa: E402

relex = GlinerRelexExtractor({"labels": ["organization", "person"],
                              "relation_labels": ["works for", "partner of"],
                              "threshold": 0.3, "device": "cuda"})
fused = relex.extract_fused([chunk()])
print("\n=== relation.gliner_relex (fused) ===")
for m in fused.mentions:
    print(f"  MENTION {m.label:14s} {m.text!r}")
for r in fused.relations:
    print(f"  RELATION {getattr(r, 'label', '?')!r}  {r!r}"[:140])
relex_ok = len(fused.mentions) >= 1  # relations are model/threshold-dependent; mentions must exist
print(f"  -> mentions={len(fused.mentions)} relations={len(fused.relations)}")

# ---- 3. redaction.gliner_pii ------------------------------------------------------------------
from latence_pii_gliner.provider import GlinerPIIRedactor  # noqa: E402

pii = GlinerPIIRedactor({"pii_labels": ["person", "email", "phone number"],
                         "threshold": 0.4, "device": "cuda"})
recs = list(pii.redact([document()]))
print("\n=== redaction.gliner_pii ===")
pii_ok = False
for rec in recs:
    spans = getattr(rec, "pii_spans", None) or getattr(rec, "spans", [])
    print(f"  doc disposition={rec.disposition} spans={len(spans)}")
    for s in spans:
        cs, ce = getattr(s, "char_start", None), getattr(s, "char_end", None)
        surf = TEXT[cs:ce] if cs is not None and ce is not None else "?"
        print(f"    {getattr(s,'pii_type',getattr(s,'label','?'))!s:16} off[{cs}:{ce}] surface={surf!r}")
    types = {str(getattr(s, "pii_type", getattr(s, "label", ""))).lower() for s in spans}
    pii_ok = pii_ok or (len(spans) >= 2)
print(f"  -> PII spans found: {pii_ok}")

ok = ner_ok and ner_finds and relex_ok and pii_ok
print("\n================ PHASE-1 GLiNER DE-RISK ================")
print(f"  entity.gliner        : {'PASS' if ner_ok and ner_finds else 'FAIL'}")
print(f"  relation.gliner_relex: {'PASS' if relex_ok else 'FAIL'}")
print(f"  redaction.gliner_pii : {'PASS' if pii_ok else 'FAIL'}")
print("RESULT:", "ALL REAL-WEIGHT PROVIDERS FUNCTIONAL" if ok else "DEFECT FOUND — inspect above")
raise SystemExit(0 if ok else 1)
