# Quality Report — latence-setup-stack (run-0001)

- documents: 82
- duration: 1742.891s
- contracts complete: PASS
  - provenance 17386/17386, classification 17386/17386, offsets aligned 17386/17386
- provenance drift: clean (7130 lookups, max 0 chars)

## Stages
- source [source/source.local_folder]: 0 in -> 82 out, 4.368s
- intake_screen [intake_screening/screening.intake_signature]: 82 in -> 82 out, 0.215s
- parse [parse/parser.plaintext]: 82 in -> 82 out, 0.062s
- chunk [chunk/chunk.markdown]: 82 in -> 3565 out, 3.913s
- content_screen [content_screening/screening.content_keyword]: 3565 in -> 3565 out, 0.267s
- induce [schema_induction/label_inducer.llm]: 3565 in -> 3565 out, 79.497s
- extract [fused_entity_relation/fused_entity_relation.gliner2]: 3565 in -> 58299 out, 265.950s
- header_refs [fused_entity_relation/fused_entity_relation.header_refs]: 3565 in -> 207 out, 0.434s
- inline_refs [fused_entity_relation/fused_entity_relation.inline_refs]: 3565 in -> 2172 out, 0.567s
- redact [redaction/redaction.gliner2]: 3565 in -> 3565 out, 64.413s
- profiling [profiling/profiling.statistical]: 58381 in -> 83 out, 2.117s
- disambiguate [disambiguation/disambiguation.glinker]: 60678 in -> 13738 out, 1274.379s
- graph [graph_assembly/graph.canonical]: 13738 in -> 13738 out, 1.407s
- complete [graph_completion/graph_completion.ultra]: 13738 in -> 0 out, 0.880s
- enrich [context_enrichment/context.kg_header]: 17303 in -> 3565 out, 3.529s
- export_kg [export/export.knowledge_graph]: 13738 in -> 13738 out, 1.443s
- export_corpus [export/export.jsonl_parquet]: 3565 in -> 3565 out, 39.450s

## Stage metrics present: Parse, Chunk, Screening, Entities, Relations, Redaction, Profiling, Disambiguation, Graph, ContextEnrichment

## Context enrichment
- chunks enriched: 3535, avg neighbors/chunk: 5.042, hub entities capped: 263
