"""The pure-PyTorch ``rspmm`` fallback: self-contained ULTRA with no C++/CUDA build.

ULTRA's fused message-passing calls a compiled ``generalized_rspmm`` kernel (a build artifact not
vendored). The vendored subset dispatches to that kernel when present and otherwise to a pure-PyTorch
:func:`_rspmm_pytorch` of the SAME primitive, so the plugin runs on any CPU/GPU host with no build
step and no upstream ULTRA repo. These tests pin the primitive's exact semantics (a hand-computed
reference) and that the dispatcher falls back instead of raising. Numerical equivalence to the real
CUDA kernel is additionally proven on the GPU pod (identical held-out MRR/Hits@k with and without the
kernel — see scripts/validate-linkpred-on-gpu.md).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from latence_linkpred_ultra._vendor.ultra.rspmm import (  # noqa: E402
    _rspmm_pytorch,
    generalized_rspmm,
)


def test_rspmm_pytorch_matches_hand_computed_reference() -> None:
    """``output[node_in] = Σ_e edge_weight[e] * (relation[type_e] ⊙ input[node_out_e])``.

    The upstream CUDA kernel CSR-sorts edges by ``node_in`` (row=node_in, col=node_out), so the
    aggregation reads ``input`` at ``node_out`` and writes at ``node_in`` — the convention this
    reference locks. 3 nodes, 2 relations, dim 2; two edges both out of node 0.
    """
    edge_index = torch.tensor([[0, 0], [1, 2]])  # node_in=[0,0], node_out=[1,2]
    edge_type = torch.tensor([0, 1])
    edge_weight = torch.tensor([1.0, 2.0])
    relation = torch.tensor([[1.0, 1.0], [2.0, 0.0]])          # [num_rel=2, dim=2]
    inp = torch.tensor([[0.0, 0.0], [3.0, 4.0], [5.0, 6.0]])   # [num_node=3, dim=2]

    out = _rspmm_pytorch(edge_index, edge_type, edge_weight, relation, inp, sum="add", mul="mul")

    # node 0: e0 = 1*( [1,1]*[3,4] ) = [3,4];  e1 = 2*( [2,0]*[5,6] ) = [20,0]  → [23,4]
    # nodes 1,2: no outgoing edge → [0,0]
    expected = torch.tensor([[23.0, 4.0], [0.0, 0.0], [0.0, 0.0]])
    assert torch.allclose(out, expected), out


def test_rspmm_add_mul_op_is_transe_style() -> None:
    """``mul='add'`` (TransE message) adds relation to the read input before weighting."""
    edge_index = torch.tensor([[0], [1]])
    out = _rspmm_pytorch(
        edge_index, torch.tensor([0]), torch.tensor([3.0]),
        torch.tensor([[2.0, 2.0]]), torch.tensor([[0.0, 0.0], [1.0, 5.0]]),
        sum="add", mul="add",
    )
    # node 0: 3 * ( [2,2] + input[node_out=1]=[1,5] ) = 3*[3,7] = [9,21]
    assert torch.allclose(out[0], torch.tensor([9.0, 21.0])), out


def test_generalized_rspmm_falls_back_not_raises_without_kernel() -> None:
    """With no compiled kernel on the path, the dispatcher uses the pure impl — never an ImportError."""
    edge_index = torch.tensor([[0], [1]])
    out = generalized_rspmm(
        edge_index, torch.tensor([0]), torch.tensor([1.0]),
        torch.tensor([[2.0, 3.0]]), torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        sum="add", mul="mul",
    )
    assert out.shape == (2, 2)
    assert torch.allclose(out[0], torch.tensor([2.0, 3.0])), out  # rel[0] * input[node_out=1]
