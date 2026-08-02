# 05 — Grilling: entity type-assignment errors

Type: grilling
Status: open
Blocked by: 01

## Question

Pattern 4, ~29/250 cases in recurring shapes: languages/demonyms → LOCATION (`Greek`,
`French`, `American`, `Chamorro`), people → ORGANIZATION/LOCATION (`Lord North`,
`Meat Loaf`, `Dōkyō`), places/things → PERSON (`John W. Ferguson House`), works → wrong slot
(`Game of Thrones` → ORGANIZATION, `June` (Carter) → DATE). Downstream this is what feeds
`country of` a demonym — so ticket 02's slot-type table depends on types being right.

Levers to arbitrate with data: (a) schema label *definitions/descriptions* passed to GLiNER2
(cheapest, zero code), (b) a demonym/language gazetteer post-check (closed word class,
DE/EN), (c) type-vote across a surface's mentions at disambiguation time (the type a surface
takes most often wins), (d) accept-and-record (types are advisory metadata — measure whether
type errors actually damage retrieval/consumers before spending).

## Outcome

Ruling + (if levers a–c win) the implementing task scoped; measured type-error delta on the
judged sample.
