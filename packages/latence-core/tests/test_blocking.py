"""The shared blocking primitive (``latence_core.blocking``): tokens, keys, closure.

Wave 1-D of the 2026-08 architecture pass: the blocking-key tokenizer (previously the
private ``disambiguation._tokens``) and the transitive blocking-neighborhood closure
(previously embedded in ``compute_affected_set``, reachable only by faking mention
records) got a public home. These tests pin:

* closure correctness on plain ``(id, blocking_text)`` pairs — no pseudo-mentions;
* key-function equivalence to the old tokenizer on representative inputs (including the
  NFKC/casefold confusable folding the cascade depends on), plus the alias identity that
  keeps the engine's behavioural surface byte-identical;
* set-equality between ``compute_affected_set`` and the closure it now delegates to.
"""

from __future__ import annotations

from latence_core import disambiguation
from latence_core.blocking import blocking_keys, blocking_tokens, closure
from latence_core.state.incremental import MentionView, compute_affected_set

# -- the key function ---------------------------------------------------------


def test_blocking_tokens_representative_inputs() -> None:
    """The old ``_tokens`` behaviour, pinned on representative surfaces."""
    assert blocking_tokens("International Business Machines") == [
        "international",
        "business",
        "machines",
    ]
    assert blocking_tokens("  Acme,  Corp.  ") == ["acme", "corp"]
    # NFKC compatibility folding: fullwidth Latin and ligatures fold to ASCII twins.
    assert blocking_tokens("ＡＣＭＥ") == ["acme"]
    assert blocking_tokens("ﬁle") == ["file"]
    # Unicode-correct casefold: ß → ss.
    assert blocking_tokens("Straße") == ["strasse"]
    # Underscore is excluded from word runs; digits are kept; non-Latin tokenises.
    assert blocking_tokens("a_b c3 東京") == ["a", "b", "c3", "東京"]
    assert blocking_tokens("") == []
    assert blocking_tokens("   \n ") == []


def test_blocking_tokens_is_the_disambiguation_tokenizer() -> None:
    """The private alias in the cascade engine is the SAME function object (no drift)."""
    assert disambiguation._tokens is blocking_tokens  # noqa: SLF001


def test_blocking_keys_are_the_distinct_tokens() -> None:
    assert blocking_keys("Acme Acme Corp") == {"acme", "corp"}
    assert blocking_keys("") == set()


# -- the closure --------------------------------------------------------------


def test_closure_direct_neighbors_via_shared_key() -> None:
    items = [
        ("r1", "International Business Machines"),
        ("r2", "Business Analytics"),  # shares "business" with r1
        ("r3", "Globex"),  # disjoint block
    ]
    assert closure(items, ["r1"]) == {"r1", "r2"}


def test_closure_is_transitive_bridge_two_blocks_away() -> None:
    """A bridge two hops out is caught: seed → shared key → neighbor → its keys → next."""
    items = [
        ("a", "alpha beta"),
        ("b", "beta gamma"),  # bridge: shares beta with a, gamma with c
        ("c", "gamma delta"),
        ("d", "epsilon"),  # untouched block
    ]
    assert closure(items, ["a"]) == {"a", "b", "c"}
    # Seeding the far end walks the same neighborhood backwards.
    assert closure(items, ["c"]) == {"a", "b", "c"}
    assert closure(items, ["d"]) == {"d"}


def test_closure_keyless_and_absent_seeds_stay_members_only() -> None:
    items = [("r1", ""), ("r2", "acme")]
    # A keyless seed closes over nothing beyond itself.
    assert closure(items, ["r1"]) == {"r1"}
    # A seed id absent from items is still a member (and pulls in nothing).
    assert closure(items, ["ghost"]) == {"ghost"}
    # No seeds ⇒ empty closure.
    assert closure(items, []) == set()


def test_closure_does_not_leak_across_disjoint_blocks() -> None:
    items = [("r1", "acme corp"), ("r2", "acme inc"), ("r3", "globex ltd")]
    assert closure(items, ["r2"]) == {"r1", "r2"}


# -- delegation parity: compute_affected_set == the shared closure ------------


def test_compute_affected_set_matches_closure_on_id_text_pairs() -> None:
    """The mention framing delegates to the same closure — set-equal, no pseudo-shape."""
    mentions = [
        MentionView("m1", "doc-a", "International Business Machines", "ORG", 1.0),
        MentionView("m2", "doc-b", "Business Analytics", "ORG", 1.0),
        MentionView("m3", "doc-b", "Analytics Cloud", "PRODUCT", 0.9),
        MentionView("m4", "doc-c", "Globex", "ORG", 1.0),
    ]
    affected = compute_affected_set(mentions, ["doc-a"])
    seed_ids = {m.mention_id for m in mentions if m.document_id == "doc-a"}
    expected = closure(((m.mention_id, m.text) for m in mentions), seed_ids)
    assert affected.mention_ids == expected == {"m1", "m2", "m3"}
    # The reported keys are exactly the affected mentions' keys.
    assert affected.blocking_keys == {
        k for m in mentions if m.mention_id in expected for k in blocking_keys(m.text)
    }
    assert not affected.is_full
