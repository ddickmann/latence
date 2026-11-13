"""The pipeline still gives the result it gave before pairs and clusters became arrays.

``results/reference/estate_12k.npz`` was written by the earlier implementation (candidate pairs
as a frame of string ids, neighbours in a dictionary of dictionaries, strong rejects as sets of
cannot-links) on the seeded 12k estate. This reruns the current code on the same estate and
requires the same candidate pairs, the same probability (to float noise) and band for each, the
same band thresholds, and the same two partitions of the records into clusters.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from equivalence import compare
from generate import generate
from run_a import run

REFERENCE = Path(__file__).parent / "results" / "reference" / "estate_12k.npz"


@pytest.mark.filterwarnings("ignore")
def test_same_pairs_bands_and_clusters_as_the_recorded_run(tmp_path: Path) -> None:
    estate = tmp_path / "estate.parquet"
    generate(4000, seed=11).to_parquet(estate, index=False)
    dump = tmp_path / "now.npz"
    run("estate", estate, 2000, 3000, 7, baseline_max=0, dump=dump)
    assert compare(str(REFERENCE), str(dump), p_tolerance=1e-9) == []


def test_the_comparison_can_fail(tmp_path: Path) -> None:
    """A comparison that cannot say "different" proves nothing: move one record, flip one band."""
    import numpy as np

    ref = dict(np.load(REFERENCE))
    moved = dict(ref)
    moved["machine"] = ref["machine"].copy()
    names, sizes = np.unique(ref["machine"], return_counts=True)
    shared = int(np.where(ref["machine"] == names[sizes > 1][0])[0][0])  # a record that has company
    moved["machine"][shared] = "somewhere-else"
    moved["band"] = ref["band"].copy()
    moved["band"][0] = "accept" if ref["band"][0] != "accept" else "reject"
    other = tmp_path / "moved.npz"
    np.savez_compressed(other, **moved)
    found = compare(str(REFERENCE), str(other))
    assert any(f.startswith("machine clustering") for f in found)
    assert any(f.startswith("band") for f in found)
