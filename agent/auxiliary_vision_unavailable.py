"""Fallback when the explicitly configured vision provider cannot build a client.

The failed provider's model ID must never be carried to another provider: re-resolving through
task config re-read ``auxiliary.vision.model`` and sent it to the auto chain, where a different
provider rejected it. Prefer the task's ``fallback_chain`` (provider, model) pairs, then the auto
vision route with no model pin so the destination picks its own model.
"""
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def fallback_for_unavailable_vision(
    task: Optional[str], failed_provider: str, main_runtime: Optional[dict[str, Any]], async_mode: bool,
) -> tuple[Optional[str], Optional[Any], Optional[str]]:
    """Return ``(effective_provider, client, model)``; client is None when nothing is reachable."""
    from agent.auxiliary_client import (
        _normalize_main_runtime, _to_async_client, _try_configured_fallback_for_unavailable_client,
        _vision_auto_route,
    )
    client, model, label = _try_configured_fallback_for_unavailable_client(task, failed_provider)
    if client is not None:
        if async_mode:
            client, model = _to_async_client(client, model or "", is_vision=True)
        return label, client, model
    logger.warning("Vision provider %s unavailable, falling back to auto vision backends", failed_provider)
    return _vision_auto_route(_normalize_main_runtime(main_runtime), None, None, async_mode)
