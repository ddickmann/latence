"""latence-pii-presidio — a real Microsoft Presidio PII Redaction Provider (W1-pii, #95).

Registers ``redaction.presidio``, an :class:`~latence_core.providers.adapter.AdapterBase`
subclass satisfying the :class:`~latence_core.capability.PIIDetector` Protocol. A true
drop-in for the in-core ``redaction.hybrid_rule`` and the learned ``redaction.gliner_pii``:
same :class:`~latence_core.contracts.RedactionRecord` contract, the same sensitivity-keyed
policy machinery, and the same H-C1 sensitive-doc no-op floor — swap the provider in a stack
with no other config change.
"""

from __future__ import annotations

from latence_pii_presidio.provider import PresidioRedactor

__all__ = ["PresidioRedactor"]
