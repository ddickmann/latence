# ruff: noqa
"""Vendored MIT ULTRA inference subset (DeepGraphLearning/ULTRA, ICLR'24, commit 427966a).

The minimal subset needed to run ULTRA *inference* (zero-shot link prediction) from a pretrained
checkpoint: ``models.py`` (``Ultra`` + ``RelNBFNet`` / ``EntityNBFNet``), ``base_nbfnet.py``
(``BaseNBFNet``), ``layers.py`` (``GeneralizedRelationalConv``), and ``tasks.py``
(``build_relation_graph`` / ``all_negative`` / ``compute_ranking`` + their helpers). Copyright (c)
2023 MilaGraph; MIT License (see ``LICENSE`` in this directory and the package NOTICE /
THIRD-PARTY-LICENSES). The files are verbatim upstream except a single import in ``models.py`` made
relative so the subset resolves inside this vendored package — the algorithm is byte-for-byte
upstream. NO model weights are vendored: the checkpoint is a config path (default ``ultra_3g``).

Importing this subpackage pulls torch / torch-geometric / torch-scatter (the ``[ultra]`` extra), so
it is imported LAZILY by :class:`~latence_linkpred_ultra.scorer.UltraScorer` at first scoring call,
never at provider discovery time (ADR-0016 keeps discovery torch-free).
"""
