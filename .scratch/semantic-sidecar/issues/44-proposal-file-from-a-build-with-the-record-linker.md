# Task: the proposal file from a fresh build with the record linker

Type: task
Mode: HITL (it needs the machine that holds the linker's environment)
Status: open: the code part is built and tested (14 Nov); the run with the linker's environment is the owner's (live request below)
Blocked by: nothing
Emerges from: 40 ("Not done")

## Question

The trial of 12 Nov wrote the proposal file twice: from the export kept on 11 Nov, which holds
the machine's 938 links, and from a fresh build without the record linker, because the
linker's environment was not on the machine that built the writer. Both files are the same
bytes, since no link is ever written into a proposal. A fresh build with the linker has not
been put through the command.

## Goal

One fresh build of the vehicle estate with the record linker on the development machine, the
proposal file written from it, compared byte for byte with the file of 12 Nov.

## Done check

- The comparison's result is written here.
- If the bytes differ, the entry that differs is named and has a ticket of its own.

## Result (14 Nov): the code part

**The build path with the linker** is tested through the real call, not a replaced function
(`estate/tests/test_linker_path.py`, 2 tests). The build writes the records file, runs
`link_records.py` in the linker's folder with that folder's own `.venv/bin/python`, reads its
`resolution/3` answer and takes it into the export; the proposal file written from that build
is compared byte for byte with the file from a build of the same estate without the linker,
with the same decisions (two spellings, the dealers, and a person's link of a dealer to a
seeded account, which the second build passes to the linker as a label).

- The linker's environment (Splink and its stack) is not on this machine, so a stand-in runs in
  its place. It is held to the linker's command line: its argument parser is the one
  `spike/link_records.py` declares, read from that file at test time, so an argument the build
  passes and the linker does not take stops it with exit 2 as it stops the linker (the second
  test shows it). It checks that the records file has exactly the columns the build declares,
  and answers from the truth column.
- **Result:** the files are the same bytes, three entries each; the person's link is in the
  export as a link decided by them and in neither file (the account is seeded).
- **No defect was found on the path**, so no fix and no red test: both tests passed on the
  tree as it was once their own mistakes were corrected (a truth for every account, the
  build's summary read as it is). They guard the path; they do not prove a change.

**The host's ticket 282** (the coordinator's contract update, built here as part of the writer):

- (a) The page's downloaded lines carry `at` (an ISO 8601 time in UTC, after `by`). The
  command takes the day from it, so no day option is needed. `--decided-on` stays for lines
  without a day (an older page, a line by hand) and **never replaces a day a line states**:
  taking "only as an override" to mean the second way would put a day on a person's decision
  that the person did not state, which ticket 40 rules out. Tested both ways.
- (b) The export marks a stated master list the estate file names under `propose.names`, and
  that is not seeded, `"proposal": "names"` (on the vehicle estate: `concept:internal.dealers`
  only; the seeded service accounts are named nowhere and would not be marked). The mark is no
  part of the claim and no assertion, so every decision made before stays valid. The writer
  (`latence-proposal-writer`) accepts an approval of a marked list without being told the list.
- **`graph-export/3` keeps its number**: the mark is optional and additive, and the host's
  page refuses any export whose `schema` is not exactly `graph-export/3` (its `data.ts`), so a
  new number would stop the page until the host changes in step. The same choice as for
  `in_layer` in ticket 40. The export schema states the field. If the host's side wants a new
  number all the same, it is one constant and the schema's title here, changed together.
- **Red first:** `test_a_page_download_alone_writes_the_file_with_no_day_option_and_no_line_written_by_hand`
  (estate; red: the export carried no mark) and
  `test_a_download_of_the_page_alone_writes_the_file_with_no_day_given_and_no_list_named`
  (package, on a fixture download in the page's format, `tests/fixtures/page-decisions.jsonl`;
  red: the dealers were counted as not carried).
- **The vehicle estate, from a download in the page's format:** a fresh build (without the
  linker) of the mirror the host's tools write, a download holding the Kia and Chrysler pairs
  and the dealers approved and the Mazda pair rejected, each with `at` and its claim, no day
  option and no line by hand: the file is **byte for byte the file of 12 Nov**'s trial (same
  person id and day), all three entries with the day stated by the decision.

**Live request (the owner's machine, which holds or can install the linker's environment):**

```bash
cd $LATENCE/.scratch/semantic-sidecar/estate
python3.12 -m venv ../spike/.venv && ../spike/.venv/bin/pip install -q -r ../spike/requirements-link.txt
.venv/bin/pip install -q -r requirements.txt            # now also installs latence-core and latence-proposal-writer
# blocks 2 and 4 of HANDOFF.md as written (the build with --resolver ../spike), then the page's
# decisions.jsonl into out/vehicles, then:
.venv/bin/python -m estate propose --config estates/vehicles.yaml --out out/vehicles --file /tmp/with-linker.yaml
mkdir -p out/vehicles-plain && cp out/vehicles/decisions.jsonl out/vehicles-plain/
PYTHONHASHSEED=0 .venv/bin/python -m estate build --config estates/vehicles.yaml \
    --duckdb /tmp/vehicles_seeded.duckdb --metric-views /tmp/metric_views.txt \
    --verified $METIS/tools/synth/estates/catalog.yaml \
    --class-answer estates/vehicles.classes_answer.gpt-6-luna.json --usage /tmp/usage.jsonl --out out/vehicles-plain
.venv/bin/python -m estate propose --config estates/vehicles.yaml --out out/vehicles-plain --file /tmp/without-linker.yaml
cmp /tmp/with-linker.yaml /tmp/without-linker.yaml && echo same
```

Report back `same`, or the output of `diff` and the two reports `propose` printed. Expected:
`same`, and the build's summary saying 938 of 1,274 accounts linked.
