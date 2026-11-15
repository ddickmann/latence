# Where `estate-semantics.json` comes from

`estate-semantics.json` is the host's alias table of the vehicle estate's semantic library, in
its published format `metis.alias/1`, copied byte for byte as a data file. No code of the host
is in this repository.

- Repository: `ddickmann/metis`
- Commit: `e4b71867c57a537bb4c7203b801df487f3cdd788` (15 Nov 2026)
- Path there: `tools/dev/tests/fixtures/alias/estate-semantics.json`
- Written there by the host's command `metis-dev concepts alias --library estate-semantics
  --out <file>` (its ticket 326), from the library's catalog, without a live import.
- Its `version` is the digest of its table, which the builder checks on reading
  (`latence_ontology.crosswalk.read_alias_tables`); its `catalog` is the catalog version it was
  read from.

To refresh it, copy the file from a newer commit of the host and change the commit above in the
same change.
