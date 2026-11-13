"""Short forms of names (acronyms, initialisms) and where their meaning can be read from.

A short form such as "IBM", "MB" or "VW" shares almost no characters with the name it stands
for, so string similarity cannot retrieve it, and it is ambiguous on its own: in an estate of a
few thousand organisations a two-letter form fits eight of them on average. What resolves it in
practice is evidence outside the string. This module gathers that evidence:

- ``scope``    the full name is used in the same thread (document, mail thread, folder) and
               this short form fits it;
- ``defined``  some document introduces it: "Lumenstone Foods Corp (LF)" or the reverse
               (Schwartz and Hearst, 2003, "A simple algorithm for identifying abbreviation
               definitions in biomedical text");
- ``alias``    the master data carries it as a known short name;
- ``initials`` nothing but the letters: every anchor whose name can be abbreviated to it.

None of these decides alone: a corpus-wide definition of "LF" does not make every "LF" that
organisation. The candidates of all routes are ranked together by a small model fitted on
reviewer labels (run_b.py), over this evidence, the anchor's popularity and the context. Every
candidate keeps its evidence, so a link can say what it rests on.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable

LEGAL_FORMS = {"ltd", "limited", "inc", "incorporated", "llc", "corp", "corporation", "plc", "co", "company", "gmbh", "ag", "sa", "bv", "nv"}
STOP = {"of", "and", "the", "for", "de", "der", "und", "&"}
ROUTES = ("scope", "defined", "alias", "initials")
_SMALL = {"a", "an", "and", "the", "of", "or", "in", "on", "for", "to", "with", "as", "by", "at", "from", "is", "are", "such"}
_PART = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")  # CamelCase and digit boundaries


def norm_short(s: str) -> str:
    """"I.B.M." and "ibm" and "H&M" as keys: letters and digits only, upper case."""
    return re.sub(r"[^A-Za-z0-9]", "", s).upper()


def is_short_form(surface: str) -> bool:
    """Looks like an acronym: one token, at most six letters or digits, and written in capitals
    (dots, an ampersand and one lower-case letter, as in "PwC", are allowed)."""
    s = surface.strip()
    if not s or " " in s:
        return False
    key = norm_short(s)
    if not 1 <= len(key) <= 6 or not key[0].isalpha():
        return False
    letters = [c for c in s if c.isalpha()]
    return len(letters) >= 2 and sum(c.isupper() for c in letters) >= len(letters) - 1 or (len(letters) == 1 and letters[0].isupper())


def _words(name: str) -> list[str]:
    return [w for w in re.split(r"[\s/]+", re.sub(r"[.,()]", " ", name)) if w]


def _subsequences(initials: list[str]) -> set[str]:
    """Initialisms often skip a word ("Orbitbrook Kestrel Engineering" -> "OE"): every ordered
    selection of the initials that keeps the first one. Only for names of up to five words."""
    if len(initials) < 2:
        return set()
    if len(initials) > 5:
        return {"".join(initials)}
    first, rest = initials[0], initials[1:]
    return {first + "".join(c for k, c in enumerate(rest) if mask >> k & 1) for mask in range(1, 2 ** len(rest))}


def short_forms(name: str) -> set[str]:
    """The short forms a name can be abbreviated to, by letters alone.

    Initials of the words, with and without the legal form; without small words and numbers,
    also with words skipped; and the initials of the parts of hyphenated and CamelCase words
    ("Mercedes-Benz" -> "MB", "PricewaterhouseCoopers" -> "PC"). A compound written as one
    lower-case word ("Volkswagen" -> "VW") cannot be split by letters; that short form has to
    come from a definition, an alias column or a reviewer."""
    words = _words(name)
    core = [w for w in words if w.lower() not in LEGAL_FORMS]
    content = [w for w in core if w.lower() not in STOP and not w.isdigit()]
    out: set[str] = set()
    for ws in (words, core, content):
        if len(ws) >= 2:
            out.add(norm_short("".join(w[0] for w in ws)))
        parts = [p for w in ws for h in w.split("-") for p in _PART.findall(h)]
        if len(parts) >= 2 and len(parts) != len(ws):
            out.add(norm_short("".join(p[0] for p in parts)))
    out |= {norm_short(x) for x in _subsequences([w[0] for w in content if w[0].isalnum()])}
    return {k for k in out if 2 <= len(k) <= 6 and k[0].isalpha()}


def _long_form_before(text: str, end: int, short: str) -> tuple[int, int] | None:
    """Schwartz-Hearst: walk the short form from its last character backwards through the text
    that ends at ``end``; every character has to be found in order, the first one at the start
    of a word. Returns the span of the shortest long form that fits."""
    key = norm_short(short)
    i, j = end - 1, len(key) - 1
    limit = max(0, end - 12 * len(key) - 40)
    while j >= 0:
        c = key[j].lower()
        while i >= limit and (text[i].lower() != c or (j == 0 and i > 0 and text[i - 1].isalnum())):
            i -= 1
        if i < limit:
            return None
        j -= 1
        i -= 1
    start = i + 1
    long = text[start:end]
    if end - start <= len(key) or len(key) < 2:
        return None
    # Seen on real filings: a "long form" that runs across a quote, a bracket or a line break
    # belongs to something else, and one that starts with a small word is a piece of a sentence.
    if re.search(r"[\u201c\u201d\"()\n]", long) or long.split()[0].lower() in _SMALL:
        return None
    return (start, end)


_QUOTED = re.compile(r"[\u201c\"]([^\u201c\u201d\"]{1,40})[\u201d\"]")
_CONNECT = {"of", "and", "&", "de", "the", "for", "du", "la"}


def _name_before(text: str, end: int) -> tuple[int, int] | None:
    """The name that ends at ``end``: the run of capitalised words (with small connecting words
    and a legal form after a comma) directly before it."""
    start = end
    for m in reversed(list(re.finditer(r"\S+", text[:end]))):
        word = m.group().strip(",.;:")
        if not word:
            continue
        if word[0].isupper() or word[0].isdigit() or word.lower() in _CONNECT:
            start = m.start()
        else:
            break
    words = text[start:end].split()
    while words and words[0].lower() in _CONNECT:  # a name does not begin with "The" or "of"
        start += len(words[0]) + 1
        words = words[1:]
    return (start, end) if words else None


def _quoted_definitions(text: str, inner: str, before_end: int) -> list[dict[str, object]]:
    """Short names a filing gives in quotes after a name: ``Name Inc. ("NI" or the "Company")``.
    An acronym counts when its letters fit the text before the bracket. Another quoted term
    counts when it is the beginning of the name ("Apple" for "Apple Inc."). What is introduced
    with "the" and does not fit as an acronym is a defined term of the document, not a name."""
    found: list[dict[str, object]] = []
    for q in _QUOTED.finditer(inner):
        term = q.group(1).strip()
        after_the = re.search(r"\bthe\s*$", inner[: q.start()], re.I) is not None
        span: tuple[int, int] | None = None
        if is_short_form(term):
            span = _long_form_before(text, before_end, term)
            if span and not text[span[0]].isupper():
                span = None
        elif not after_the:
            name = _name_before(text, before_end)
            if name:
                words = [w.strip(",.") for w in text[name[0] : name[1]].split()]
                given = term.split()
                if given and len(given) < len(words) and [w.lower() for w in words[: len(given)]] == [w.lower() for w in given]:
                    span = name
        if span:
            found.append({"short": term, "long": text[span[0] : span[1]], "start": span[0], "end": span[1]})
    return found


def mine_definitions(text: str) -> list[dict[str, object]]:
    """Definitions a text gives: "long form (SF)", "SF (long form)", and the quoted form of
    company filings, ``Long Name Inc. ("LN")``.

    Each result has ``short`` (as written), ``long`` and the long form's ``start`` and ``end``
    in the text."""
    found: list[dict[str, object]] = []
    for m in re.finditer(r"\(([^()]{1,60})\)", text):
        inner = m.group(1).strip()
        before = text[: m.start()].rstrip()
        if _QUOTED.search(inner):  # Name ("SF" or the "Company")
            found.extend(_quoted_definitions(text, inner, len(before)))
            continue
        if is_short_form(inner):  # long form (SF)
            span = _long_form_before(text, len(before), inner)
            if span:
                found.append({"short": inner, "long": text[span[0] : span[1]], "start": span[0], "end": span[1]})
        else:  # SF (long form)
            last = re.search(r"(\S+)$", before)
            if last and is_short_form(last.group(1)):
                span = _long_form_before(inner, len(inner), last.group(1))
                if span and span[0] == 0:
                    found.append({"short": last.group(1), "long": inner, "start": m.start(1), "end": m.start(1) + len(inner)})
    return found


class AliasTable:
    """Short form -> the anchors it may stand for, each with the evidence for it.

    Evidence kinds, see the module text: ``scope`` (per thread), ``defined``, ``alias`` and
    ``initials`` (whole corpus). Nothing is decided here.
    """

    def __init__(self, names: Iterable[str]) -> None:
        self.forms = [short_forms(n) for n in names]
        self._initials: dict[str, set[int]] = defaultdict(set)
        for j, forms in enumerate(self.forms):
            for key in forms:
                self._initials[key].add(j)
        self._alias: dict[str, Counter[int]] = defaultdict(Counter)
        self._defined: dict[str, Counter[int]] = defaultdict(Counter)
        self._scoped: dict[str, Counter[int]] = defaultdict(Counter)

    def add_alias(self, short: str, anchor: int) -> None:
        self._alias[norm_short(short)][anchor] += 1

    def add_definition(self, short: str, anchor: int) -> None:
        self._defined[norm_short(short)][anchor] += 1

    def add_in_scope(self, scope: str, anchor: int) -> None:
        """A full name was linked to ``anchor`` inside ``scope`` (a thread)."""
        self._scoped[scope][anchor] += 1

    def candidates(self, short: str, scope: str | None = None) -> dict[int, dict[str, int]]:
        """Every anchor some evidence proposes for this short form: anchor -> {kind: count}."""
        key = norm_short(short)
        out: dict[int, dict[str, int]] = defaultdict(dict)
        if scope is not None:
            for a, n in self._scoped.get(scope, {}).items():
                if key in self.forms[a]:
                    out[a]["scope"] = n
        for a, n in self._defined.get(key, {}).items():
            out[a]["defined"] = n
        for a, n in self._alias.get(key, {}).items():
            out[a]["alias"] = n
        for a in self._initials.get(key, ()):
            out[a]["initials"] = 1
        return dict(out)


def strongest(evidence: dict[str, int]) -> str:
    """The strongest kind of evidence a candidate has, for reporting what a link rests on."""
    return next((r for r in ROUTES if r in evidence), "none")
