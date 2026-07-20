# R6-connectors_ingestion-10 — `SharePointSource` builds `source_uri` without percent-encoding, destroying the Graph identity

**Severity:** major
**Area:** connectors / ingestion
**Site:** `packages/latence-source-sharepoint/src/latence_source_sharepoint/provider.py` (`SharePointSource._source_uri`)
**Related:** ticket #182 (SharePoint Source Provider), ADR-0021/0031 (provenance is true)

## What is wrong

`_source_uri` builds the Provenance URI by raw f-string interpolation:

```python
return f"sharepoint://{self._drive_id}{item.drive_path}?itemId={item.item_id}"
```

Neither the drive path nor the Graph item id is percent-encoded. SharePoint Online has
permitted `#`, `%` and `&` in file and folder names since 2017, and spaces have always been
legal, so the interpolated values routinely contain characters that are *URI syntax*.

## Failure scenario

A single file named `/Q1 #final.md` in drive `drive-1` yields:

```
sharepoint://drive-1/Q1 #final.md?itemId=item-Q1 #final.md
```

`urllib.parse.urlsplit` reads that as `path='/Q1 '`, `query=''`,
`fragment='final.md?itemId=item-Q1 #final.md'` — the `#` opened a fragment, so
`parse_qs(parts.query).get("itemId")` is `None`. **The stable Graph identity is
unrecoverable by any URI parser.** Nothing raises.

Two further corruptions from the same root cause:

* a literal `%` in a name is worse than lost — `/Report%20final.md` round-trips through
  `unquote` as `Report final.md`, silently naming a *different* item;
* unescaped spaces make the value not a valid URI at all.

The corrupted URI is stamped on every `ParserInput` and inherited unchanged by every
downstream record into the exported corpus, so the damage is durable.

This contradicts the property the package asserts in `provider.py:14-16` and
`README.md:93-96` ("the stable Graph identity that resolves back to the document") and breaks
the README live-smoke step that instructs the operator to paste the `itemId` into
`GET /drives/{drive_id}/items/{itemId}`.

The offline suite is blind to it: every fixture path in `testing.py` and all four test modules
uses only `[A-Za-z0-9./-]` names, and
`test_lists_and_fetches_documents_with_sharepoint_provenance` asserts only
`"itemId=" in source_uri` — a substring check that passes on the corrupted value.

Note the Graph *client* already gets this right (`graph.py` quotes every path and item id it
puts in a request URL); only the Provenance URI does not.

## What to build

* Percent-encode every interpolated component of `source_uri`:
  * drive id — `quote(safe="")` (it is one opaque path segment);
  * drive path — `quote(safe="/")`, so `/` stays a separator and `#`, `%`, ` `, `?`, `&` are escaped;
  * item id — `quote(safe="")`, so it survives as a single query value.
* The encoding must be the standard one, so `urlsplit` + `parse_qs` + `unquote` recover the
  exact drive path and the exact Graph item id.
* URIs made of only unreserved characters must be **byte-identical** to before (no churn on
  existing corpora).
* Document the encoding in the package README next to the live-smoke instruction.

## Acceptance criteria

1. For a document whose name contains `#`, a space and `%`, `urlsplit(source_uri).fragment`
   is empty and `parse_qs(urlsplit(source_uri).query)["itemId"][0]` equals the Graph item id
   exactly (after `unquote`).
2. `unquote(urlsplit(source_uri).path)` equals the item's drive-relative path exactly, for the
   same document.
3. A path with a literal `%` does not decode to a different name (`Report%20final.md` stays
   `Report%20final.md`, not `Report final.md`).
4. The existing exact-URI assertion
   (`sharepoint://drive-1/Handbook/onboarding.md?itemId=item-Handbook-onboarding.md`) still
   holds unchanged.
5. A regression test in `packages/latence-source-sharepoint/tests/test_sharepoint_source.py`
   fails on the old code and passes on the fix.
6. `make test` stays green at or above 3320 passed / 29 skipped.
