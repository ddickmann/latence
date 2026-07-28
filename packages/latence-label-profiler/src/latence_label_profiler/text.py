"""Tokenization, stopword segmentation, and the canonical label normal form.

Everything here is pure, deterministic, and dependency-free. There is no POS tagger, no lemmatizer,
no model and no downloaded resource — a corpus profiler that needed a network fetch to start would
fail the ticket's zero-network bar on its first call.

The two shapes downstream needs:

* **Content runs** — maximal runs of content words between stopwords, punctuation and bare numbers.
  This is RAKE's candidate-phrase step (Rose, Engel, Cramer & Cowley, 2010), and it is a noun-phrase
  proxy that costs nothing: in English and German, function words are exactly the phrase boundaries.
* **Gaps** — the material *between* two content runs inside one sentence, which is where a relation
  phrase lives (Fader, Soderland & Etzioni, 2011, ReVerb: relation = the ``V | VP | VW*P`` span
  between two argument phrases).

A **bare number is a boundary, not a token**. ``purchase order number 4711`` must mine the type
``purchase order number``, never ``number 4711`` — the number is an instance, and instances are the
thing a *type* space must not contain.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

#: Mirrors ``latence_schema_inducer.sanitize.MAX_LABEL_LEN``: a label longer than this is truncated.
#: Duplicated rather than imported because this package is deliberately dependency-free; the
#: equivalence is not asserted by comment but by ``test_mining.py``, which round-trips our output
#: through the REAL ``sanitize_label`` and requires it to be a fixed point.
MAX_LABEL_LEN = 64

#: Unicode categories stripped from a label: C0/C1 controls + DEL (``Cc``) and the zero-width /
#: bidi-override format characters (``Cf``) an injection payload can hide in.
_CONTROL_CATEGORIES = frozenset({"Cc", "Cf"})
#: A run of separators — whitespace, ``_`` or ``-`` — collapses to one space in the normal form.
_SEPARATOR_RUN = re.compile(r"[\s_-]+")

#: One token: a word-ish run, or a run of punctuation.
_TOKEN_RE = re.compile(r"[\w'’‐-―-]+|[^\s\w]+", re.UNICODE)
#: Sentence terminators. Deliberately crude — a profiler samples, it does not parse.
_SENTENCE_BREAK = re.compile(r"[.!?;:…\n\r]+")

WORD = "word"
STOP = "stop"
NUMBER = "number"
PUNCT = "punct"

# --------------------------------------------------------------------------------------
# Stopwords. English + German function words, hand-listed. Function words are not a
# copyrightable creative work and this list is not copied from any package's data file, which is
# what keeps the zero-dependency, zero-download posture honest (ADR-0012: we verify what we adopt,
# and here we adopt nothing). German is included because DECISIONS.md D5/T-WIZ-2 says this corpus
# will contain German, whose compounding is the reason `whole_word=False` exists downstream.
# --------------------------------------------------------------------------------------
_EN_STOPWORDS = """
a about above after again against all also am an and any are as at be because been before being
below between both but by can cannot could did do does doing down during each few for from further
had has have having he her here hers herself him himself his how i if in into is it its itself just
me more most my myself no nor not now of off on once only or other our ours ourselves out over own
same she should so some such than that the their theirs them themselves then there these they this
those through to too under until up very was we were what when where which while who whom why will
with would you your yours yourself yourselves shall may might must upon per via etc
""".split()  # noqa: SIM905 - a readable word list beats a 200-element literal

_DE_STOPWORDS = """
aber alle allem allen aller alles als also am an andere anderem anderen anderer anderes auch auf
aus bei beim bin bis bist da damit dann der den des dem die das dass daß derselbe dazu dein deine
dem den denn denen der deren dessen dich dir doch dort du durch ein eine einem einen einer eines
einige er es etwas euch euer für gegen gewesen hab habe haben hat hatte hatten hier hin hinter ich
ihm ihn ihnen ihr ihre ihrem ihren ihrer ihres im in indem ins ist jede jedem jeden jeder jedes
jener jenes jetzt kann kein keine können könnte machen man manche mein meine mit muss musste nach
nicht nichts noch nun nur ob oder ohne sehr sein seine seinem seinen seiner seines selbst sich sie
sind so solche soll sollte sondern sonst über um und uns unse unser unter viel vom von vor war
waren warst was weg weil weiter welche wenn werde werden wie wieder wir wird wirst wo wollen wollte
würde würden zu zum zur zwar zwischen
""".split()  # noqa: SIM905 - as above

#: The default stopword set: English + German function words.
DEFAULT_STOPWORDS: frozenset[str] = frozenset(_EN_STOPWORDS) | frozenset(_DE_STOPWORDS)

#: Prepositions/particles that may TRAIL a relation phrase (ReVerb's ``W*P`` tail): ``classified
#: under``, ``issued by``, ``maps to``. They are stopwords, so they never start a content run — the
#: relation miner re-admits them deliberately, only as a tail.
RELATION_TAIL_WORDS: frozenset[str] = frozenset(
    """
    by to of for with from in on at into onto under over about against through between as
    von zu für mit aus in an auf unter über durch zwischen gegen als bei nach
    """.split()  # noqa: SIM905 - as above
)

#: Verbs that carry a relation but have no usable morphology (copulas + high-frequency auxiliaries),
#: English and German. Everything else is caught by :data:`_VERB_SUFFIXES`.
CLOSED_VERBS: frozenset[str] = frozenset(
    """
    is are was were be been being has have had contains includes requires maps refers relates
    ist sind war waren sein gewesen hat haben hatte hatten wird werden wurde wurden enthält
    """.split()  # noqa: SIM905 - as above
)

#: Morphological verb evidence, in the absence of a POS tagger. Deliberately conservative: a bare
#: ``-s`` is excluded because it is far more often a plural noun than a verb, and a false relation
#: type is worse than a missing one (it burns one of the <= N label slots on noise).
_VERB_SUFFIXES: tuple[str, ...] = ("ed", "ing", "ates", "ised", "ized", "ifies", "fies", "iert")


@dataclass(frozen=True)
class Token:
    """One token: its casefolded form, its original surface, and what kind of boundary it is."""

    text: str
    raw: str
    kind: str

    @property
    def is_content(self) -> bool:
        return self.kind == WORD


@dataclass(frozen=True)
class Sentence:
    """One sentence's tokens, plus the index spans of its maximal content runs."""

    tokens: tuple[Token, ...]
    runs: tuple[tuple[int, int], ...]

    def run_tokens(self, span: tuple[int, int]) -> tuple[Token, ...]:
        return self.tokens[span[0] : span[1]]

    def gap_tokens(self, left: tuple[int, int], right: tuple[int, int]) -> tuple[Token, ...]:
        return self.tokens[left[1] : right[0]]


@dataclass(frozen=True)
class TokenizedText:
    """One text's sentences and the flat list of its content runs (the RAKE candidate phrases)."""

    sentences: tuple[Sentence, ...]

    @property
    def runs(self) -> tuple[tuple[Token, ...], ...]:
        return tuple(
            sentence.run_tokens(span) for sentence in self.sentences for span in sentence.runs
        )


def _strip_controls(text: str) -> str:
    return "".join(
        ch for ch in text if ch.isspace() or unicodedata.category(ch) not in _CONTROL_CATEGORIES
    )


def canonical_label(raw: str) -> str | None:
    """The canonical normal form — the SAME one ``latence_schema_inducer.sanitize.sanitize_label``
    produces, so an emitted candidate is a fixed point of the downstream sanitizer rather than a
    near-miss that silently renames itself between the profiler and the inducer.

    Strips control/format characters, collapses every separator run (whitespace, ``_``, ``-``) to a
    single space, casefolds, and caps at :data:`MAX_LABEL_LEN`. Returns ``None`` for an unusable
    candidate so the caller drops it rather than emitting a placeholder.
    """
    text = _SEPARATOR_RUN.sub(" ", _strip_controls(raw)).strip()
    if not text:
        return None
    text = text.casefold()
    if len(text) > MAX_LABEL_LEN:
        text = text[:MAX_LABEL_LEN].strip()
    return text or None


def classify(raw: str, stopwords: frozenset[str]) -> str:
    """Which kind of token ``raw`` is — the whole segmentation decision, in one place."""
    if not raw:
        return PUNCT
    lowered = raw.casefold()
    if not any(ch.isalnum() for ch in raw):
        return PUNCT
    if any(ch.isdigit() for ch in raw):
        # An alphanumeric with a digit in it ("4711", "CN8471", "2024-Q3") is an INSTANCE. Types are
        # what we are mining; letting one through would put a document's own identifiers into the
        # corpus-level type space.
        return NUMBER
    if len(lowered) < 2 or lowered in stopwords:
        return STOP
    return WORD


def tokenize(text: str, *, stopwords: frozenset[str] = DEFAULT_STOPWORDS) -> TokenizedText:
    """Split ``text`` into sentences, tokens, and maximal content runs. Pure and total."""
    sentences: list[Sentence] = []
    for raw_sentence in _SENTENCE_BREAK.split(text):
        if not raw_sentence.strip():
            continue
        tokens = tuple(
            Token(text=match.group(0).casefold(), raw=match.group(0),
                  kind=classify(match.group(0), stopwords))
            for match in _TOKEN_RE.finditer(raw_sentence)
        )
        if not tokens:
            continue
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for index, token in enumerate(tokens):
            if token.is_content:
                if start is None:
                    start = index
            elif start is not None:
                runs.append((start, index))
                start = None
        if start is not None:
            runs.append((start, len(tokens)))
        sentences.append(Sentence(tokens=tokens, runs=tuple(runs)))
    return TokenizedText(sentences=tuple(sentences))


def is_verb_like(token: Token) -> bool:
    """Verb evidence without a POS tagger: a closed set plus conservative suffix morphology.

    This is the weakest link in the whole profiler and it is reported as such: the relation slot's
    recall is bounded by this heuristic, so the artifact carries a per-slot note rather than letting
    a thin relation pool read as "this corpus has few relations".
    """
    if token.kind not in (WORD, STOP):
        return False
    if token.text in CLOSED_VERBS:
        return True
    return len(token.text) > 4 and token.text.endswith(_VERB_SUFFIXES)


def subsequences(tokens: tuple[str, ...], *, max_len: int) -> tuple[tuple[str, ...], ...]:
    """Every contiguous sub-sequence of ``tokens`` up to ``max_len``, in a deterministic order.

    This is not an arbitrary generalization step — it is the DOWNSTREAM MATCHER'S SEMANTICS written
    out. ``LexiconLabelInducer`` compiles each cue to ``\\b(?:cue)\\b`` and asks whether it occurs
    in the chunk, so a cue fires on a surface exactly when its token sequence is a contiguous
    sub-sequence of that surface. Defining coverage this way means the coverage number the profiler
    reports is the coverage the inducer will actually deliver, not an optimistic proxy for it.

    Ordered by start index then by length, so the enumeration never depends on set iteration order.
    """
    out: list[tuple[str, ...]] = []
    for start in range(len(tokens)):
        for end in range(start + 1, min(start + max_len, len(tokens)) + 1):
            out.append(tokens[start:end])
    return tuple(out)
