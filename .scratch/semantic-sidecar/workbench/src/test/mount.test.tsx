// The claim "a host can mount this" as a test: the workbench inside a router it does not own,
// at a path it did not choose, with the host's chrome and the host's own controls.
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes, useLocation } from "react-router";
import { describe, expect, it, vi } from "vitest";

import { ResolutionWorkbench, StaticGraphSource, StaticReviewSource, memoryStore, type ButtonProps, type Crumb, type HostChrome, type ListKeys, type Notice } from "../index";
import { graphExport, reviewExport } from "./fixtures";

const actor = { id: "t", name: "Tester" };

function Where() {
  return <output data-testid="where">{useLocation().pathname}</output>;
}

/** A host shell: owns J/K/Esc the way a real one does, and records what it is told. */
function hostChrome() {
  const seen = { crumbs: [] as Crumb[][], notices: [] as Notice[], recounts: 0, keys: {} as ListKeys };
  const chrome: HostChrome = {
    useCrumbs: (crumbs) => void seen.crumbs.push(crumbs),
    useListKeys: (keys) => void (seen.keys = keys),
    recount: () => void (seen.recounts += 1),
    notify: (n) => void seen.notices.push(n),
  };
  return { chrome, seen };
}

function mount(path: string, options: { chrome?: Partial<HostChrome>; controls?: Parameters<typeof ResolutionWorkbench>[0]["controls"]; graph?: boolean } = {}) {
  const store = memoryStore();
  const reviewSource = new StaticReviewSource(reviewExport, { actor, store });
  const graphSource = options.graph === false ? undefined : new StaticGraphSource(graphExport);
  const view = render(
    <MemoryRouter initialEntries={[path]}>
      <Where />
      <Routes>
        <Route
          path="/x/y/*"
          element={<ResolutionWorkbench reviewSource={reviewSource} {...(graphSource ? { graphSource } : {})} {...(options.chrome ? { chrome: options.chrome } : {})} {...(options.controls ? { controls: options.controls } : {})} />}
        />
      </Routes>
    </MemoryRouter>,
  );
  return { ...view, reviewSource };
}

const where = () => screen.getByTestId("where").textContent;

describe("mounted in a host's router", () => {
  it("keeps the open pair in the host's URL, under the path the host chose", async () => {
    const user = userEvent.setup();
    mount("/x/y");
    const rows = await screen.findAllByTestId("queue-row");
    expect(rows).toHaveLength(3);
    expect(rows[0]).toHaveAttribute("href", "/x/y/review/r1~r2");
    await user.click(rows[0]!);
    expect(where()).toBe("/x/y/review/r1~r2");
    expect(await screen.findByTestId("evidence")).toHaveTextContent("Acme Capital Inc");
    expect(screen.getAllByTestId("evidence-row")).toHaveLength(3);
    await user.click(screen.getByTestId("back-to-queue"));
    expect(where()).toBe("/x/y/review");
  });

  it("records one decision per key, asks why for 'different', and moves to the next open pair", async () => {
    const user = userEvent.setup();
    const { chrome, seen } = hostChrome();
    const { reviewSource } = mount("/x/y/review/r1~r2", { chrome });
    await screen.findByTestId("evidence");

    await user.keyboard("a");
    await waitFor(() => expect(where()).toBe("/x/y/review/r2~r3"));
    expect(seen.notices.at(-1)?.title).toBe("Recorded: same");
    expect(seen.recounts).toBe(1);

    // "Different" needs a reason before it may be recorded; Ctrl+Enter records it.
    await user.keyboard("r");
    const sheet = await screen.findByTestId("decision-sheet");
    expect(within(sheet).getByTestId("sheet-confirm")).toBeDisabled();
    await user.keyboard("{Control>}{Enter}{/Control}");
    expect(screen.getByTestId("decision-sheet")).toBeInTheDocument();
    await user.click(within(sheet).getByTestId("reason-business"));
    await user.keyboard("{Control>}{Enter}{/Control}");
    await waitFor(() => expect(where()).toBe("/x/y/review/r4~r5"));

    await user.keyboard("u");
    await waitFor(() => expect(seen.recounts).toBe(3));
    // Nothing is left open, so the workbench stays on the last pair.
    expect(where()).toBe("/x/y/review/r4~r5");

    const ledger = await reviewSource.loadLedger();
    expect(ledger.decisions.map((d) => [d.item, d.verdict, d.reason])).toEqual([
      ["r1~r2", "same", undefined], ["r2~r3", "different", "business"], ["r4~r5", "unsure", undefined],
    ]);
    expect(ledger.decisions.every((d) => d.actor.name === "Tester" && !Number.isNaN(Date.parse(d.at)))).toBe(true);
    expect(screen.getByTestId("qa-open")).toHaveTextContent("0");
    expect(screen.getByTestId("qa-decided")).toHaveTextContent("3");
  });

  it("changes a recorded decision only with a note, and keeps the earlier one in the history", async () => {
    const user = userEvent.setup();
    const { reviewSource } = mount("/x/y/review/r4~r5");
    await screen.findByTestId("evidence");
    await user.click(screen.getByTestId("decide-same"));
    await waitFor(() => expect(where()).toBe("/x/y/review/r1~r2"));
    await user.click(screen.getByTestId("filter-all"));
    await user.click(screen.getAllByTestId("queue-row")[2]!);
    await screen.findByTestId("history");

    await user.click(screen.getByTestId("decide-unsure"));
    const sheet = await screen.findByTestId("decision-sheet");
    expect(within(sheet).getByTestId("sheet-confirm")).toBeDisabled();
    await user.type(within(sheet).getByRole("textbox"), "the city is missing on one side");
    await user.click(within(sheet).getByTestId("sheet-confirm"));
    await waitFor(async () => expect((await reviewSource.loadLedger()).decisions).toHaveLength(2));
    const ledger = await reviewSource.loadLedger();
    expect(ledger.decisions.map((d) => d.verdict)).toEqual(["same", "unsure"]);
    expect(ledger.decisions[1]?.note).toBe("the city is missing on one side");
  });

  it("shows a suggestion with its grounds, takes it with Y, and records whether the reviewer agreed", async () => {
    const user = userEvent.setup();
    const { reviewSource } = mount("/x/y/review/r1~r2");
    await screen.findByTestId("evidence");
    const card = screen.getByTestId("suggestion");
    expect(card).toHaveTextContent("Different");
    expect(card).toHaveTextContent("41 of 44 were different");
    expect(card).toHaveTextContent("Industry differs");
    expect(card).toHaveTextContent("Name not the same");
    expect(screen.getByTestId("suggestion-trust")).toHaveTextContent("Checked on 300 decided pairs the suggestion had not seen: right in 96 of 100.");
    expect(screen.getByTestId("suggestion-trust")).toHaveTextContent("Calling every pair the same would be right in 70 of 100.");
    expect(screen.getByTestId("qa-agreed")).toHaveTextContent("–");

    // Y takes it: "different" is recorded without the reason dialog, with the suggestion as the reason.
    await user.keyboard("y");
    await waitFor(() => expect(where()).toBe("/x/y/review/r2~r3"));
    expect(screen.queryByTestId("decision-sheet")).toBeNull();
    expect(screen.getByTestId("qa-agreed")).toHaveTextContent("1 of 1");
    // This pair has no suggestion: no card, and Y does nothing.
    expect(screen.queryByTestId("suggestion")).toBeNull();
    await user.keyboard("y");
    expect(where()).toBe("/x/y/review/r2~r3");

    // Going against a suggestion is recorded as that.
    await user.click(screen.getByTestId("filter-all"));
    await user.click(screen.getAllByTestId("queue-row")[2]!);
    expect(await screen.findByTestId("suggestion")).toHaveTextContent("29 of 30 were the same");
    await user.keyboard("r");
    await user.click(within(await screen.findByTestId("decision-sheet")).getByTestId("reason-location"));
    await user.keyboard("{Control>}{Enter}{/Control}");
    await waitFor(() => expect(screen.getByTestId("qa-agreed")).toHaveTextContent("1 of 2"));

    const ledger = await reviewSource.loadLedger();
    expect(ledger.decisions.map((d) => [d.item, d.verdict, d.reason, d.suggested])).toEqual([
      ["r1~r2", "different", "as_suggested", "different"],
      ["r4~r5", "different", "location", "same"],
    ]);
  });

  it("decides a whole group as one with M, and takes it back with undo", async () => {
    const user = userEvent.setup();
    const { reviewSource } = mount("/x/y/review/r1~r2");
    await screen.findByTestId("evidence");
    await user.keyboard("m");
    const sheet = await screen.findByTestId("merge-sheet");
    expect(within(sheet).getByTestId("plan-outcome")).toHaveTextContent("2 open pairs become Same.");
    await user.keyboard("{Control>}{Enter}{/Control}");
    // Both pairs of the group are decided, so the workbench moves to the other group.
    await waitFor(() => expect(where()).toBe("/x/y/review/r4~r5"));
    expect(screen.getByTestId("qa-decided")).toHaveTextContent("2");
    expect(screen.getByTestId("notice")).toHaveTextContent("Group decided as one organisation: 2 pairs recorded");

    await user.click(screen.getByTestId("notice-action"));
    await waitFor(() => expect(screen.getByTestId("qa-decided")).toHaveTextContent("0"));
    const ledger = await reviewSource.loadLedger();
    expect(ledger.decisions.map((d) => [d.item, d.verdict, d.withdraws === true])).toEqual([
      ["r1~r2", "same", false], ["r2~r3", "same", false], ["r1~r2", "same", true], ["r2~r3", "same", true],
    ]);
    expect(new Set(ledger.decisions.map((d) => d.batch)).size).toBe(1);
  });

  it("splits a group with S: letters per record, the pairs follow, and a separated machine link is reported", async () => {
    const user = userEvent.setup();
    const { reviewSource } = mount("/x/y/review/r1~r2");
    await screen.findByTestId("evidence");
    await user.keyboard("s");
    const sheet = await screen.findByTestId("split-sheet");
    // The starting point follows the machine's accepted link: r1 and r3 together, r2 apart.
    expect(within(sheet).getByTestId("part-r1-A")).toHaveAttribute("aria-pressed", "true");
    expect(within(sheet).getByTestId("part-r3-A")).toHaveAttribute("aria-pressed", "true");
    expect(within(sheet).getByTestId("part-r2-B")).toHaveAttribute("aria-pressed", "true");
    expect(within(sheet).getByTestId("plan-outcome")).toHaveTextContent("2 organisations. Open pairs that become Same: 0. Open pairs that become Different: 2.");
    expect(within(sheet).queryByTestId("plan-separates")).toBeNull();

    // Move r3 over to r2: that cuts the link the machine accepted.
    await user.click(within(sheet).getByTestId("part-r3-B"));
    expect(within(sheet).getByTestId("plan-outcome")).toHaveTextContent("Open pairs that become Same: 1. Open pairs that become Different: 1.");
    expect(within(sheet).getByTestId("plan-separates")).toHaveTextContent("separates 1 pair the machine accepted");

    // One letter for everything is not a split.
    await user.click(within(sheet).getByTestId("part-r1-B"));
    expect(within(sheet).getByTestId("sheet-confirm")).toBeDisabled();
    await user.click(within(sheet).getByTestId("part-r1-A"));

    await user.click(within(sheet).getByTestId("sheet-confirm"));
    await waitFor(async () => expect((await reviewSource.loadLedger()).intents).toHaveLength(1));
    const ledger = await reviewSource.loadLedger();
    expect(ledger.decisions.map((d) => [d.item, d.verdict, d.reason])).toEqual([
      ["r1~r2", "different", "group_split"], ["r2~r3", "same", undefined],
    ]);
    expect(ledger.intents[0]?.parts).toEqual([["r1"], ["r2", "r3"]]);
    expect(ledger.intents[0]?.separates_accepted).toEqual([["r1", "r3"]]);
  });

  it("shows the group decision on a pair it decided, with an undo that stays", async () => {
    const user = userEvent.setup();
    mount("/x/y/review/r1~r2");
    await screen.findByTestId("evidence");
    await user.click(screen.getByTestId("group-merge"));
    await user.click(within(await screen.findByTestId("merge-sheet")).getByTestId("sheet-confirm"));
    await waitFor(() => expect(where()).toBe("/x/y/review/r4~r5"));
    await user.click(screen.getByTestId("filter-all"));
    await user.click(screen.getAllByTestId("queue-row")[0]!);
    expect(await screen.findByTestId("history")).toHaveTextContent("by group decision");
    expect(screen.getByTestId("group-recorded")).toHaveTextContent("2 pairs recorded");
    await user.click(screen.getByTestId("group-undo"));
    await waitFor(() => expect(screen.getByTestId("group-undone")).toBeInTheDocument());
    expect(screen.getByTestId("history")).toHaveTextContent("withdrawn");
    expect(screen.getByTestId("no-decision")).toBeInTheDocument();
  });

  it("leaves J, K and Esc to the host and tells it the breadcrumb", async () => {
    const { chrome, seen } = hostChrome();
    mount("/x/y/review/r1~r2", { chrome });
    await screen.findByTestId("evidence");
    expect(seen.crumbs.at(-1)).toEqual([{ label: "Review", to: "/x/y/review" }, { label: "Acme Capital Inc · Acme Consulting Inc" }]);

    // The workbench itself does not listen for these keys when the host supplies them.
    fireEvent.keyDown(window, { key: "j" });
    expect(where()).toBe("/x/y/review/r1~r2");
    // The host does, and calls what the workbench gave it.
    seen.keys.next?.();
    await waitFor(() => expect(where()).toBe("/x/y/review/r2~r3"));
    seen.keys.previous?.();
    await waitFor(() => expect(where()).toBe("/x/y/review/r1~r2"));
    seen.keys.close?.();
    await waitFor(() => expect(where()).toBe("/x/y/review"));
  });

  it("handles J, K and Esc itself when no host does", async () => {
    mount("/x/y/review");
    await screen.findAllByTestId("queue-row");
    fireEvent.keyDown(window, { key: "j" });
    await waitFor(() => expect(where()).toBe("/x/y/review/r1~r2"));
    fireEvent.keyDown(window, { key: "j" });
    await waitFor(() => expect(where()).toBe("/x/y/review/r2~r3"));
    fireEvent.keyDown(window, { key: "k" });
    await waitFor(() => expect(where()).toBe("/x/y/review/r1~r2"));
    fireEvent.keyDown(window, { key: "Escape" });
    await waitFor(() => expect(where()).toBe("/x/y/review"));
  });

  it("renders the host's own controls in place of Carbon's", async () => {
    const HostButton = vi.fn((props: ButtonProps) => (
      <button type="button" className="host-btn" data-testid={props.testId} onClick={props.onClick} disabled={props.disabled}>{props.children}</button>
    ));
    mount("/x/y/review/r1~r2", { controls: { Button: HostButton } });
    await screen.findByTestId("evidence");
    expect(screen.getByTestId("decide-same")).toHaveClass("host-btn");
    expect(HostButton.mock.calls.some(([p]) => p.kbd === "A" && p.kind === "primary")).toBe(true);
  });

  it("walks the graph: schema, a concept's instances, a node, and from an open link into its review", async () => {
    const user = userEvent.setup();
    mount("/x/y/graph");
    expect(await screen.findByTestId("graph-home")).toHaveTextContent("5 records from 4 source systems describe 3 organisations");
    expect(screen.getByTestId("concept-organisation")).toHaveAttribute("aria-pressed", "true");
    await user.click(screen.getByTestId("concept-city"));
    const detail = screen.getByTestId("concept-detail");
    await waitFor(() => expect(within(detail).getAllByTestId("node-link").map((a) => a.textContent)).toEqual(["Leeds", "Bristol"]));

    await user.type(screen.getByRole("textbox"), "acme");
    const hits = await screen.findByTestId("search-hits");
    await user.click(within(hits).getAllByTestId("node-link")[0]!);
    expect(where()).toBe("/x/y/graph/org%3Ar1");
    const node = await screen.findByTestId("graph-node");
    expect(node).toHaveTextContent("Backed by 2 records from 2 source systems");
    expect(within(screen.getByTestId("focus-graph")).getAllByTestId("focus-node")).toHaveLength(5);
    const open = screen.getByTestId("group-possibly_same");
    expect(open).toHaveTextContent("0 of 2 pairs decided");
    await user.click(within(open).getByTestId("review-link"));
    expect(where()).toBe("/x/y/review/r1~r2");

    // And back: a record in the evidence links to the organisation it supports.
    await user.click(await screen.findByTestId("open-graph-right"));
    expect(where()).toBe("/x/y/graph/org%3Ar2");
    expect(await screen.findByTestId("graph-node")).toHaveTextContent("Acme Consulting Inc");
  });

  it("is the review screen alone when the host passes no graph", async () => {
    mount("/x/y/graph", { graph: false });
    expect(await screen.findAllByTestId("queue-row")).toHaveLength(3);
    expect(screen.queryByTestId("tab-graph")).toBeNull();
  });
});
