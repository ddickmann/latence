"""Enterprise GLinker Disambiguation Provider for the Latence framework (ADR-0046).

Registers ``disambiguation.glinker`` under the ``latence.providers`` entry-point group (ADR-0004):
the neural GLinker L2->L3->L4->L0 entity-linking pipeline + the audited resolver cascade, ported
from the private ``latenceai-dataset-intelligence`` repo, behind the framework's single
:class:`~latence_core.capability.Disambiguator` seam — a drop-in for ``disambiguation.cascade`` /
``disambiguation.embedding`` with the identical ``DisambiguationRecord`` contract.
"""

from latence_disambig_glinker.provider import GLinkerDisambiguator

__all__ = ["GLinkerDisambiguator"]
