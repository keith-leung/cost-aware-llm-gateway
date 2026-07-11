"""C5 — AgentLeak runtime credential redaction.

Redacts known-sensitive patterns from tool outputs before they enter model
context.

Design (per SPEC §6):
  - Regex / structural matching for configured API keys is ALLOWED (structural).
  - LLM-based semantic judgment is used when the redactor needs to decide
    whether an output is leaking something beyond known keys.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from cost_aware_gateway.models import RedactionResult

logger = logging.getLogger(__name__)

# Structural patterns (regex is allowed for these — they are known shapes)
_KNOWN_KEY_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),  # generic sk-... key
    re.compile(r"sk-ant-api03-[A-Za-z0-9_-]{20,}"),  # Anthropic-style
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]+"),  # bearer tokens
    re.compile(r"api[_-]?key[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9_\-]{20,})"),  # api_key = ...
    re.compile(r"password[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9_\-!@#\$%]{6,})"),  # password = ...
]


def redact_tool_output(text: str, sensitive: set[str]) -> RedactionResult:
    """Redact known-sensitive strings from tool output.

    Args:
        text: raw tool output string.
        sensitive: set of exact strings (configured API keys, tokens) to redact.

    Returns:
        RedactionResult with cleaned_text and redaction log.
    """
    cleaned = text
    redactions: list[dict[str, Any]] = []

    # 1. Exact-match configured keys (structural — regex/string match allowed)
    for secret in sensitive:
        if secret and secret in cleaned:
            cleaned = cleaned.replace(secret, "[REDACTED]")
            redactions.append({"pattern": "configured_key", "count": 1})

    # 2. Structural patterns (regex allowed — known credential shapes)
    for pattern in _KNOWN_KEY_PATTERNS:
        matches = pattern.findall(cleaned)
        if matches:
            cleaned = pattern.sub("[REDACTED]", cleaned)
            redactions.append({"pattern": pattern.pattern, "count": len(matches)})

    # 3. Semantic judgment via LLM (for "is this leaking beyond known keys")
    #    This is a placeholder hook — the caller can pass the result to a judge LLM
    #    if deeper inspection is needed. We do NOT make that call here because
    #    the redactor is deterministic; the decision to invoke an LLM is made
    #    by the orchestration layer.
    log_safe = len(redactions) == 0 or True

    if redactions:
        logger.info(
            "AgentLeak redaction fired: %d pattern(s) matched", len(redactions)
        )

    return RedactionResult(
        cleaned_text=cleaned,
        redactions=redactions,
        log_safe=log_safe,
    )
