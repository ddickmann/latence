"""GLiNER 2.5 Providers for the Latence framework (ADR-0064).

One package, two entry points — ``fused_entity_relation.gliner25`` and ``redaction.gliner25`` —
because GLiNER 2.5 is a single checkpoint family serving both Capabilities over one dependency
pin, one compat gate and one loader seam.
"""

from latence_gliner25.extract import Gliner25FusedExtractor
from latence_gliner25.redact import Gliner25PIIRedactor

__all__ = ["Gliner25FusedExtractor", "Gliner25PIIRedactor"]
