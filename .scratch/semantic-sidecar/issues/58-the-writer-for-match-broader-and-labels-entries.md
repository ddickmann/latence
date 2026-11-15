# Build: the proposal writer writes `match`, `broader` and `labels` entries (`layer-proposal/3`)

Type: build
Mode: AFK
Status: open
Blocked by: 57 (the crosswalk names the conformed key a match is written to); the host's 304
(decided and built on its side, 15 Nov)
Emerges from: the host's drafted ticket B-2 (15 Nov); ADR-0070 (the proposal file is the one
seam); 13 (every human decision a link with its actor)

## Question

The host's tool reads three new entries in a file of format `layer-proposal/3` (its 304, the
owner's decisions of 15 Nov: a reviewer approves a cross-domain match and every named domain's
data owner certifies it; a close match is never placed). The writer writes `spellings`,
`names` and `link` today (`layer-proposal/1` and `/2`).

## The shapes (the host's 304, as decided)

```
format: layer-proposal/3
- match: {key: dimension:<ns>.<local>, concepts: [{domain, object, column}, ...], close: false}
  origin: {element, claim, decision: {verdict: approved, by, at}, certified: [{domain, by, at}, ...]}
- broader: {object, column, value: <code>, broader: [<code>...], support: {rows, share}}
- labels: {object, column, language, names: <name>, synonyms: [...]}
```

A close match is written with `close: true` or left out. The host answers with a receipt
(`layer-receipt/1`, `kind` per entry).

## What is asked

`latence-proposal-writer` writes the three entries from approved (and, for a match, certified)
decisions only; the schema and the receipt reader follow; the writer's `SCHEMA_VERSION` is
bumped.

## Done when

Each shape round-trips into a released library version on the host's tool, as ticket 40 did
(where the host's tool cannot run here, the round trip is a live request and the writer is
tested against the shapes).
