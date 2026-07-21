"""Tokenizer-config shape shim for the default PII checkpoint (dogfood-2b-process, ADR-0034).

The **field failure**: ``latence process`` died entering the mandatory redact stage with

.. code-block:: text

    ProviderError: redaction.gliner2: provider failure in 'load_model' (native AttributeError)
    root: AttributeError: 'list' object has no attribute 'keys'

raised deep inside ``transformers`` (4.57.x) while ``GLiNER2.from_pretrained`` builds the
DeBERTa-v2 tokenizer of the default checkpoint ``fastino/GLiNER2-Guardrails-PII-Multi``.

The **root cause** is a checkpoint/library shape drift, not a bug in this framework. That
checkpoint's ``tokenizer_config.json`` stores ``extra_special_tokens`` as a **list** of the 10
GLiNER2 structural tokens::

    "extra_special_tokens": ["[SEP_STRUCT]", "[SEP_TEXT]", "[P]", "[C]", "[E]", "[R]",
                             "[L]", "[EXAMPLE]", "[OUTPUT]", "[DESCRIPTION]"]

— the shape an older ``transformers`` accepted (its ``_set_model_specific_special_tokens`` type
hint still reads ``list[str]``). ``transformers`` 4.57 refactored that field to a **dict**
``{name: token}`` and its loader now calls ``special_tokens.keys()`` / ``.items()`` on the value,
so a *list* raises the ``AttributeError`` above at load — killing the whole pipeline before any run
dir is written. The sibling extract checkpoint ``fastino/gliner2-multi-v1`` stores
``extra_special_tokens`` as ``{}`` (already the dict shape), which is exactly why only this PII
checkpoint is affected.

The **fix** is a byte-identical-output shim: coerce the list to the dict shape 4.57 requires,
``{token: token}``, *before* the load. All 10 tokens are ALREADY registered as ``special`` added
tokens in the checkpoint's ``tokenizer.json`` (the fast tokenizer's source of truth), so the
model's tokenization is unchanged — the shim only fixes the *shape* of a redundant config field so
the loader stops crashing and the tokens remain model-specific special tokens (present in
``all_special_tokens``, as GLiNER2 intends). This mirrors the established precedent
``latence_disambig_glinker...GLinkerBackend._patch_tokenizer_config`` (patch a shape-incompatible
field in the cached ``tokenizer_config.json`` before the framework reads it).

**Guarded, best-effort, never fatal** (ADR-0034 posture — no *new* silent error, only the removal
of a crash): a missing ``huggingface_hub`` (dep absent), an offline/unresolvable download, a
malformed config, or a read-only cache all log at ``debug`` and return ``False`` — the load then
proceeds exactly as before (and, on 4.57, fails with the same typed ``ProviderError`` the operator
already saw, never a worse or hidden one). **Idempotent**: it rewrites only when the value is
actually a *list*, so a config already in dict shape (the extract checkpoint, or a re-run after the
first patch) is a no-op.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

_log = logging.getLogger("latence.provider.gliner2_pii")

#: The tokenizer-config file whose ``extra_special_tokens`` shape 4.57 rejects as a list.
_TOKENIZER_CONFIG = "tokenizer_config.json"


def coerce_list_extra_special_tokens(model_id: str, *, provider_name: str) -> bool:
    """Coerce a list-valued ``extra_special_tokens`` to a dict in the checkpoint's cached config.

    Returns ``True`` iff the cached ``tokenizer_config.json`` for ``model_id`` was rewritten (its
    ``extra_special_tokens`` was a list and is now the ``{token: token}`` dict shape that
    ``transformers`` >= 4.57 requires). Returns ``False`` — never raises — when there is nothing to
    do (already a dict / absent) or the shim cannot run (no ``huggingface_hub``, offline,
    unreadable, or a read-only cache), so the caller's load proceeds unchanged.
    """
    try:
        import json  # noqa: PLC0415 - stdlib, imported on use
        import os  # noqa: PLC0415
        import shutil  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        from huggingface_hub import hf_hub_download  # noqa: PLC0415 - heavy dep, resolved on use
    except ImportError:
        # ``huggingface_hub`` is not installed (it ships transitively with ``gliner2[local]``); with
        # no gliner2 stack there is no load to protect. The load path surfaces the missing dep.
        _log.debug(
            "%s: huggingface_hub unavailable; skipping the extra_special_tokens shape shim",
            provider_name,
        )
        return False

    try:
        config_path = hf_hub_download(model_id, _TOKENIZER_CONFIG)
    except Exception as exc:  # noqa: BLE001 - offline / private / unresolved: best-effort, never fatal
        _log.debug(
            "%s: could not resolve %s for %s; skipping the shape shim (%s)",
            provider_name,
            _TOKENIZER_CONFIG,
            model_id,
            exc,
        )
        return False

    try:
        with open(config_path, encoding="utf-8") as handle:
            config: dict[str, Any] = json.load(handle)
    except (OSError, ValueError) as exc:
        _log.debug(
            "%s: could not read %s; skipping the shape shim (%s)", provider_name, config_path, exc
        )
        return False

    extra = config.get("extra_special_tokens")
    if not isinstance(extra, list):
        # Already a dict (the extract checkpoint, or a re-run after the first patch), or absent:
        # nothing to fix. Idempotent no-op — the load proceeds unchanged.
        return False

    # Preserve every string token as a model-specific special token keyed by itself — the exact dict
    # shape transformers 4.57 iterates, and byte-identical in effect (the tokens are already special
    # in tokenizer.json). Non-str entries (there are none in the real checkpoint) are dropped rather
    # than smuggled through the isinstance(value, (str, AddedToken)) check the loader would reject.
    coerced = {token: token for token in extra if isinstance(token, str)}
    config["extra_special_tokens"] = coerced

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json", dir=os.path.dirname(config_path))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        # Atomic same-filesystem rename over the cached file (replacing the snapshot symlink with a
        # real, patched file); readers see either the old or the new config, never a torn one.
        shutil.move(tmp_path, config_path)
    except OSError as exc:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        _log.debug(
            "%s: could not write the patched %s (read-only cache?); skipping the shape shim (%s)",
            provider_name,
            _TOKENIZER_CONFIG,
            exc,
        )
        return False

    _log.info(
        "%s: coerced list-valued extra_special_tokens (%d tokens) to the dict shape transformers "
        ">= 4.57 requires in %s's cached %s — byte-identical tokenization, load no longer crashes",
        provider_name,
        len(coerced),
        model_id,
        _TOKENIZER_CONFIG,
    )
    return True
