# ruff: noqa
"""``rspmm`` — the relational sparse-matrix-product kernel ULTRA's fused message-passing calls.

Upstream ULTRA ships ``ultra/rspmm/rspmm.py``, which JIT-compiles a C++/CUDA extension
(``generalized_rspmm``) on first use. That extension is a **build artifact**, not vendored here
(we vendor only the Python inference subset — models/layers/tasks/base_nbfnet). On the warm pod at
``/root/ULTRA`` the real extension is already built and takes precedence; this module only matters
when a host imports the vendored subset without the compiled upstream kernel on the path.

``layers.GeneralizedRelationalConv.message_and_aggregate`` does ``from .rspmm import
generalized_rspmm``. To keep the vendored subset faithful AND honest, we re-export the real
upstream kernel when the genuine ``ultra`` package (with its compiled extension) is importable, and
otherwise raise a clear, actionable error the moment the fused path is actually invoked — never a
silent wrong answer. The non-fused message/aggregate path (which does NOT touch this module) is used
whenever ``edge_weight.requires_grad`` is True or the message function is ``rotate`` (see
``layers.GeneralizedRelationalConv.propagate``).
"""

from __future__ import annotations


def generalized_rspmm(*args, **kwargs):  # pragma: no cover - exercised only on a real fused run
    """Dispatch to the genuine upstream ``rspmm`` kernel, or fail loudly if it is not built.

    Tries the real ``ultra.rspmm`` (the compiled extension on the warm pod) first; if it is not
    importable, raises a clear ImportError telling the operator to build ULTRA's ``rspmm`` extension
    (``python -m ultra.rspmm`` in the upstream repo) or install the genuine ``ultra`` package. This
    is only ever reached on a real fused message-passing run — the CPU verify lane never imports the
    vendored subset (the real-inference test is dep+CUDA-gated).
    """
    try:
        from ultra.rspmm.rspmm import generalized_rspmm as _real
    except Exception as exc:  # noqa: BLE001 - re-raise as an actionable, typed message
        msg = (
            "ULTRA's fused 'rspmm' kernel is not available. The vendored inference subset ships "
            "only the Python model code (models/layers/tasks/base_nbfnet), not the C++/CUDA rspmm "
            "extension (a build artifact). Build it from the upstream ULTRA repo (its rspmm compiles "
            "on first use) or install the genuine 'ultra' package so 'ultra.rspmm.rspmm' is "
            "importable. On the validation pod at /root/ULTRA it is already built."
        )
        raise ImportError(msg) from exc
    return _real(*args, **kwargs)
