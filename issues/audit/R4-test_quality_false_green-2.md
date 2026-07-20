# R4 — [major] The SharePoint determinism regression test is hollow (false green)

**Area:** `packages/latence-source-sharepoint`
**Introduced by:** b9ea8ef (R3 SharePoint determinism fix)
**Severity:** major — the fix itself is still guarded by a sibling test, but the test that
advertises itself as *the* guard cannot fail, and the sort key it names is pinned by nothing.

## What was wrong

Commit b9ea8ef fixed a real determinism defect: Graph specifies no ordering for a `driveItem`
children collection absent `$orderby` (which the connector never sends), and
`GraphClient.list_files` never sorted. The fix added
`files.sort(key=lambda item: item.item_id)` in `graph.py:list_files`, and — correctly —
changed the shipped fake so `FakeGraphApi._children_of` returns files **descending** by path,
on the stated reasoning that a pre-sorted fake "would prove a property the real dependency does
not deliver".

The accompanying regression test then defined its adversarial double **relatively**:

```python
class _UnorderedGraphApi(FakeGraphApi):
    def _children_of(self, folder):
        return list(reversed(super()._children_of(folder)))
```

For the test's own fixture — four files `/a.md /b.md /c.md /d.md`, all at the drive root, no
subfolders — the base fake yields `['d','c','b','a']` and the reversal yields `['a','b','c','d']`:
**strictly ascending**, i.e. exactly the order the client is supposed to impose. The two
reversals cancel and the double hands the client the answer.

Two independent false greens followed:

1. Deleting `files.sort(...)` left `test_files_are_sorted_by_item_id_even_when_graph_returns_them_unordered`
   **passing** (only its two siblings went red).
2. Replacing the sort key with `item.name` left the entire 62-test SharePoint suite **green**,
   so the `item_id` claim asserted in the test name, in `graph.py`'s comment ("a rename reorders
   names, not ids") and in `list_files`' docstring was pinned by nothing in the tree.

Root cause of (2): the fake derived every item id from the path (`item-a.md`), so id order and
name order were identical by construction and no test could tell the two keys apart.

## Failure scenario

A future edit that narrows the sort to only the multi-folder path, or that switches the key to
`name` — reintroducing precisely the rename-instability the connector's comment rejects — ships
green, and the test whose name advertises it as the guard for exactly that is the one that
cannot fail.

## What was built

- `FakeFile` gains an optional `item_id: str | None = None` — an **opaque** id override, so a
  fixture can model real Graph ids (`01BYE5RZ…`), which carry no relation to a name.
  `FakeGraphApi.item_id()` consults it, defaulting to the readable path-derived id.
- `_UnorderedGraphApi` now states its order **absolutely** (descending by id) rather than
  relatively ("reverse whatever the base did"), so it can never again cancel out against the
  base fake's order.
- The regression test uses ids that disagree with the names (`alpha→01QQ`, `bravo→01AA`,
  `charlie→01ZZ`, `delta→01MM`) and asserts the id order, the resulting name order, and that
  the name order is *not* sorted. It also guards the guard: it asserts the double's served
  order is `['01ZZ','01QQ','01MM','01AA']` before exercising the client.

No production change was required — the fix in `graph.py` was correct; only its proof was not.

## Acceptance criteria

- [x] Deleting `files.sort(key=lambda item: item.item_id)` from `graph.py:list_files` fails
      `test_files_are_sorted_by_item_id_even_when_graph_returns_them_unordered`.
- [x] Changing the sort key to `item.name` fails the same test.
- [x] The adversarial double provably serves a non-ascending order (asserted in-test).
- [x] No test weakened or deleted; the SharePoint suite stays at 62 passed.
- [x] `make test` stays green at or above 2985 passed / 28 skipped.
