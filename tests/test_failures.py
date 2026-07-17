"""Failure-path verification for C1/C2/C5.

These tests verify the gateway actually REJECTS / BLOCKS / REDACTS when
expected, not just that success paths return True.
"""

from __future__ import annotations

import time

import fakeredis
import pytest

from cost_aware_gateway.budget import BudgetTracker
from cost_aware_gateway.models import BudgetExceededError, CircuitState
from cost_aware_gateway.breaker import CircuitBreaker
from cost_aware_gateway.redact import redact_tool_output


@pytest.fixture()
def fake_budget():
    client = fakeredis.FakeRedis(decode_responses=True)
    return BudgetTracker(redis_url="redis://localhost:6379/0", default_limit=1000, default_window=3600.0, _redis_client=client)


@pytest.fixture()
def fake_breaker():
    client = fakeredis.FakeRedis(decode_responses=True)
    return CircuitBreaker(name="fail-provider", redis_url="redis://localhost:6379/0", failure_threshold=2, recovery_seconds=1, _redis_client=client)


class TestBudgetFailurePaths:
    def test_budget_exhaustion_blocks_reserve(self, fake_budget: BudgetTracker) -> None:
        """Once limit is hit, further reserves must raise BudgetExceededError."""
        fake_budget.reset("u1")
        fake_budget.set_budget("u1", limit_tokens=100)
        fake_budget.check_and_reserve("u1", 60)
        fake_budget.check_and_reserve("u1", 40)  # exactly hits limit
        with pytest.raises(BudgetExceededError):
            fake_budget.check_and_reserve("u1", 1)  # must fail

    def test_window_reset_clears_spend(self, fake_budget: BudgetTracker) -> None:
        """After window expires, reserved/spent must reset to 0."""
        fake_budget.reset("u2")
        fake_budget.set_budget("u2", limit_tokens=100, window_seconds=1)
        fake_budget.check_and_reserve("u2", 100)
        time.sleep(1.1)
        fake_budget.check_and_reserve("u2", 50)  # should not raise after reset
        status = fake_budget.status("u2")
        assert status is not None
        assert status["reserved"] == 50
        assert status["spent"] == 0


class TestBreakerFailurePaths:
    def test_open_breaker_blocks_all_calls(self, fake_breaker: CircuitBreaker) -> None:
        """When OPEN, allow() must return False until recovery window passes."""
        fake_breaker.reset()
        fake_breaker.record_failure()
        fake_breaker.record_failure()
        assert fake_breaker.state == CircuitState.OPEN
        # Immediately after trip, all calls blocked
        assert fake_breaker.allow() is False
        assert fake_breaker.allow() is False
        assert fake_breaker.allow() is False

    def test_half_open_probe_serialization(self, fake_breaker: CircuitBreaker) -> None:
        """Only one caller should get through during half-open probe."""
        fake_breaker.reset()
        fake_breaker.record_failure()
        fake_breaker.record_failure()
        assert fake_breaker.state == CircuitState.OPEN
        time.sleep(1.1)
        # First allow() may transition to half_open; subsequent ones should block
        first = fake_breaker.allow()
        second = fake_breaker.allow()
        # At least one should be blocked if first succeeded
        if first:
            assert second is False, "half-open probe not serialized: both calls allowed"


class TestRedactionFailurePaths:
    def test_api_key_not_leaked_in_output(self) -> None:
        """Configured API key must never appear in redacted output."""
        key = "sk-abcdef1234567890abcdef1234567890"
        text = f"error: call failed with key={key}"
        result = redact_tool_output(text, sensitive={key})
        assert key not in result.cleaned_text
        assert "[REDACTED]" in result.cleaned_text

    def test_multiple_leaks_all_redacted(self) -> None:
        """Every occurrence of the secret must be redacted."""
        key = "sk-aaaabbbbccccddddeeee"
        text = f"{key} {key} {key}"
        result = redact_tool_output(text, sensitive={key})
        assert result.cleaned_text.count("[REDACTED]") == 3
        assert key not in result.cleaned_text

    def test_generic_sk_pattern_caught(self) -> None:
        """Even without configured key, generic sk-... pattern must be caught."""
        text = " leaked: sk-1234567890abcdefABCDEF "
        result = redact_tool_output(text, sensitive=set())
        assert "sk-1234567890abcdefABCDEF" not in result.cleaned_text
