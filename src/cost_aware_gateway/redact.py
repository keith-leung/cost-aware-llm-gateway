"""C5 — AgentLeak runtime credential redaction.

Redacts known-sensitive patterns from tool outputs before they enter model
context.

Design (per SPEC §6):
  - Regex / structural matching for configured API keys is ALLOWED (structural).
  - LLM-based semantic judgment is used when the redactor needs to decide
    whether an output is leaking something beyond known keys.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

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

_JUDGE_PROMPT = (
    "Does this tool output contain any sensitive information beyond what was "
    "already redacted (API keys, credentials, PII, internal identifiers)? "
    'Reply JSON: {"leaks": true/false, "reason": "..."}'
)


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

    log_safe = len(redactions) == 0

    if redactions:
        logger.info(
            "AgentLeak redaction fired: %d pattern(s) matched", len(redactions)
        )

    return RedactionResult(
        cleaned_text=cleaned,
        redactions=redactions,
        log_safe=log_safe,
    )


def redact_with_llm_judge(
    text: str,
    sensitive: set[str],
    judge_client: Callable[[str], dict[str, Any]],
) -> RedactionResult:
    """Structural redaction followed by LLM semantic leak detection.

    Args:
        text: raw tool output string.
        sensitive: set of exact strings to redact structurally.
        judge_client: callable that accepts a prompt string and returns a dict
            with at least ``leaks`` (bool) and ``reason`` (str) keys.

    Returns:
        RedactionResult. If the LLM flags a leak, ``llm_flagged=True`` and
        ``cleaned_text`` is the structurally-redacted text (we do NOT let the
        LLM mutate the text — it may hallucinate). If no leak, returns the
        structural result as-is.
    """
    # 1. Structural redaction first (deterministic, fast)
    structural = redact_tool_output(text, sensitive)

    # 2. Ask the LLM judge whether the *structurally-redacted* text still
    #    contains something sensitive the regex didn't catch.
    prompt = f"{_JUDGE_PROMPT}\n\nTool output:\n{structural.cleaned_text}"
    try:
        verdict = judge_client(prompt)
    except Exception as exc:
        logger.warning("LLM judge call failed: %s", exc)
        return structural

    leaks = bool(verdict.get("leaks", False))
    reason = str(verdict.get("reason", ""))

    if leaks:
        logger.info("AgentLeak LLM judge flagged leak: %s", reason)
        return RedactionResult(
            cleaned_text=structural.cleaned_text,
            redactions=structural.redactions,
            log_safe=False,
            llm_flagged=True,
        )

    return structural
