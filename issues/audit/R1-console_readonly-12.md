# R1 — the console's "fully offline" audit is blind to protocol-relative URLs (`//host`)

- **Severity:** major
- **Dimension:** console read-only / offline guarantee / test quality (hollow audit)
- **Branch:** feat/release-fleet-integration
- **Tickets:** #188 AC 4 (read-only + offline), docs/getting-started/run-console.md ("It never phones home")
- **Status:** fixed

## What is wrong

`packages/latence-console/src/latence_console/render.py:82`

```python
_EXTERNAL = re.compile(r"(?i)\b(?:https?:)?//(?!127\.0\.0\.1|localhost)[a-z0-9.-]+")
```

`external_references()` (render.py:86) is **the** audit four console test modules use to prove the
rendered bytes are fully offline (`tests/test_browse.py:258`, `tests/test_e2e_console.py:73,104`,
`tests/test_quality_report_view.py:145` — all `assert render.external_references(...) == []`).

The scheme-optional branch is **structurally dead**. When the optional `(?:https?:)?` group matches
empty, the match must begin at `/`, and the leading `\b` then requires a *word* character
immediately before that `/`. In every context where a protocol-relative URL actually appears —
`src="//…"`, `href="//…"`, `url(//…)`, or bare text — the preceding character is `"`, `(` or a
space, all non-word, so `\b` never holds. The regex therefore only ever matches `http://` /
`https://`. It also misses every other scheme (`s3://`, `ftp://`, `//[2001:db8::1]`).

## Failure scenario

Verified against the real module before the fix:

```
'<img src="//cdn.example.com/a.png">'   -> []
'<a href="//evil.example.com/x">'       -> []
'@import url(//fonts.example.com/f.css)'-> []
'text //evil.example.com'               -> []
'https://evil.example.com'              -> ['https://evil.example.com']   # only this is caught
```

Consequence: adding a protocol-relative CDN font, analytics beacon or image to `_STYLE` or any
template in `render.py` passes **every** console test, including the e2e assertion that the served
demo pages are fully offline. The documented guarantee ("empty ⇒ the page is fully offline";
"It never phones home") is unenforced for the most common off-machine URL form. The page CSP
(`default-src 'none'`) still blocks sub-resource fetches at runtime, but not `<a href="//host">`
navigation — and the build-time audit that is supposed to catch this is hollow.

## What to build

Make `external_references` catch every off-machine URL form in the emitted bytes, with the
loopback exclusion preserved:

1. Match an authority-form URL — optional `scheme:` (any scheme, not just http/https) followed by
   `//` and a host — with **no leading `\b`**, so the scheme-less (protocol-relative) form is
   reachable. Single-slash same-origin paths (`href="/run/abc"`) must still not match.
2. Support bracketed IPv6 authorities (`//[2001:db8::1]/x`).
3. Move the loopback exclusion out of the regex into an explicit host set
   (`127.0.0.1`, `localhost`, `::1`/`[::1]`), so `localhost.evil.example.com` is *not* silently
   whitelisted by a prefix match and the rule is readable.
4. Keep the return contract identical: the full matched reference text, in document order,
   `[]` ⇒ fully offline.

## Acceptance criteria

- [x] `external_references` reports protocol-relative references in attribute, CSS `url(...)`,
      `@import` and bare-text positions.
- [x] Non-http schemes with an authority (`s3://`, `ftp://`) are reported; bracketed IPv6 hosts are
      reported.
- [x] Loopback (`http://127.0.0.1:PORT/`, `//localhost/x`, `//[::1]/x`) is still **not** reported;
      same-origin single-slash paths (`/run/abc`, `file:///tmp/x`) are still **not** reported.
- [x] A non-vacuous regression test fails on the pre-fix regex and passes on the fix
      (`tests/test_offline_audit.py`).
- [x] The four existing offline assertions stay green (rendered pages are genuinely clean).
- [x] `make test` stays at or above 2985 passed / 28 skipped; ruff + mypy --strict clean.
