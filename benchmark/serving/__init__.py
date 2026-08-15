# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""Serving-path validation harnesses (wayfinder-enterprise-serving).

The S5 bench measures the retrieval stack over a brute-force dense first stage; production serves
the same stack through a real ANN engine behind the ``RetrievalBackend`` seam. The harnesses here
prove — against live dependencies, never fakes — that the served path returns the benched path's
numbers (:mod:`benchmark.serving.parity`).
"""
