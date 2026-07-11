"""Tests for C5 — AgentLeak runtime credential redaction."""

from __future__ import annotations

import pytest

from cost_aware_gateway.redact import redact_tool_output
from cost_aware_gateway.models import RedactionResult


class TestRedaction:
    def test_configured_key_stripped(self) -> None:
        key = "sk-abcdef1234567890abcdef1234567890"
        text = f"Here is your key: {key}. Use it."
        result = redact_tool_output(text, sensitive={key})
        assert key not in result.cleaned_text
        assert "[REDACTED]" in result.cleaned_text
        assert len(result.redactions) > 0

    def test_clean_passthrough(self) -> None:
        text = "Hello, world!"
        result = redact_tool_output(text, sensitive={"sk-notintext"})
        assert result.cleaned_text == text
        assert result.redactions == []

    def test_generic_sk_pattern(self) -> None:
        text = "key: sk-1234567890abcdefABCDEF"
        result = redact_tool_output(text, sensitive=set())
        assert "sk-1234567890abcdefABCDEF" not in result.cleaned_text

    def test_bearer_token_pattern(self) -> None:
        text = "Authorization: Bearer abcdef1234567890"
        result = redact_tool_output(text, sensitive=set())
        assert "abcdef1234567890" not in result.cleaned_text

    def test_multiple_redactions(self) -> None:
        key = "sk-aaaabbbbccccddddeeee"
        text = f"{key} and {key} again"
        result = redact_tool_output(text, sensitive={key})
        assert result.cleaned_text.count("[REDACTED]") == 2
