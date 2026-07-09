# ruff: noqa
"""``rspmm`` — the relational sparse-matrix-product kernel ULTRA's fused message-passing calls.

Upstream ULTRA ships ``ultra/rspmm/rspmm.py``, which JIT-compiles a C++/CUDA extension
(``generalized_rspmm``) on first use. That extension is a **build artifact**, not vendored here
(we vendor only the Python inference subset — models/layers/tasks/base_nbfnet).

``layers.GeneralizedRelationalConv.message_and_aggregate`` does ``from .rspmm import
generalized_rspmm``. To make the vendored subset **self-contained** — it runs on any CPU/GPU host
with NO C++/CUDA build step and NO upstream ULTRA repo on the path — this module dispatches
``generalized_rspmm`` to the genuine compiled kernel when it is importable (fast; e.g. a host that
also has the real ``ultra`` package), and otherwise to :func:`_rspmm_pytorch`, a pure-PyTorch
implementation of the *same* primitive. The fallback is **numerically identical** to the fused kernel
(verified on the GPU pod: byte-for-byte the same held-out MRR/Hits@k and predicted edges with and
without the compiled kernel), because it replaces ONLY the sparse-matmul primitive — ULTRA's own
boundary/degree/PNA math in ``layers.message_and_aggregate`` is untouched. It is slower on very large
graphs; enterprise KGs are small enough that this is a non-issue, and the compiled kernel still wins
wherever it exists.
"""

from __future__ import annotations

from typing import Any


def _kernel_available() -> bool:
    """True iff the genuine compiled upstream ``ultra.rspmm`` kernel is importable."""
    try:
        from ultra.rspmm.rspmm import generalized_rspmm as _real  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _rspmm_pytorch(edge_index: Any, edge_type: Any, edge_weight: Any, relation: Any, input: Any, sum: str = "add", mul: str = "mul") -> Any:
    """A pure-PyTorch equivalent of the fused ``generalized_rspmm`` primitive.

    Computes exactly what the CUDA kernel computes — for each edge ``e = (in, type, out)`` the message
    ``edge_weight[e] * (relation[type] {mul} input[in])`` reduced (``sum``/``max``/``min``) into
    ``out`` — with plain ``index_add_`` / ``scatter_reduce``. No C++/CUDA build, runs on CPU or GPU.
    It replaces ONLY the sparse-matmul primitive; ULTRA's own boundary/degree/PNA math in
    ``layers.message_and_aggregate`` is untouched, so the result is numerically identical to the fused
    kernel (verified on the pod: same held-out MRR/Hits@k, same predicted edges, with and without the
    compiled kernel). ``relation``/``input`` are the ``[num_relation | num_node, batch*dim]`` flattened
    tensors the layer passes; ``edge_index`` is ``[2, E]`` (in-row, out-row).
    """
    import torch

    # rspmm convention (verified from the upstream CUDA source): edges are CSR-sorted by ``node_in``,
    # so ``row = node_in`` (the WRITTEN/output node) and ``col = node_out`` (the READ/input node):
    # ``output[node_in] = reduce_e  edge_weight[e] * (relation[type_e] {mul} input[node_out_e])``.
    # (Aggregation flows node_out → node_in — the reverse of the naive reading.)
    node_in, node_out = edge_index[0], edge_index[1]
    rel = relation.index_select(0, edge_type)          # [E, F]
    x = input.index_select(0, node_out)                # read at col = node_out
    if mul == "mul":
        message = rel * x
    elif mul == "add":
        message = rel + x
    else:  # pragma: no cover - ULTRA only emits 'mul'/'add'
        raise ValueError(f"Unknown rspmm mul op {mul!r}")
    message = message * edge_weight.unsqueeze(-1)       # [E, F]

    num_node, feat = input.shape
    out_index = node_in.unsqueeze(-1).expand(-1, feat)  # write at row = node_in
    if sum == "add":
        out = input.new_zeros(num_node, feat)
        out.scatter_add_(0, out_index, message)
    elif sum in ("max", "min"):
        fill = float("-inf") if sum == "max" else float("inf")
        out = input.new_full((num_node, feat), fill)
        out.scatter_reduce_(
            0, out_index, message, reduce="amax" if sum == "max" else "amin", include_self=True
        )
        # A node with no incoming edge keeps the ±inf sentinel; the fused kernel yields 0 there.
        out = torch.where(torch.isinf(out), torch.zeros_like(out), out)
    else:  # pragma: no cover - ULTRA only emits 'add'/'max'/'min'
        raise ValueError(f"Unknown rspmm sum op {sum!r}")
    return out


def generalized_rspmm(*args: Any, **kwargs: Any) -> Any:
    """The fused relational sparse-matmul: genuine compiled kernel if available, else pure PyTorch.

    Prefers the real ``ultra.rspmm`` extension (fast, e.g. the warm pod at ``/root/ULTRA``); when it is
    not built, falls back to :func:`_rspmm_pytorch` — a numerically-identical pure-PyTorch
    implementation of the same primitive — so the vendored ULTRA subset runs self-contained on any
    CPU/GPU host with NO C++/CUDA build step and NO upstream ULTRA repo on the path. The fallback is
    slower on very large graphs but exact; enterprise-scale KGs are small enough that it is fine, and
    the compiled kernel still takes precedence wherever it exists.
    """
    try:
        from ultra.rspmm.rspmm import generalized_rspmm as _real
    except Exception:  # noqa: BLE001 - kernel not built on this host → portable pure-torch path
        return _rspmm_pytorch(*args, **kwargs)
    return _real(*args, **kwargs)
