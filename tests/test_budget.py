"""Tests for C1 — Redis-backed per-user token budget."""

from __future__ import annotations

import threading

import fakeredis
import pytest

from cost_aware_gateway.budget import BudgetTracker
from cost_aware_gateway.models import BudgetExceededError


@pytest.fixture()
def fake_budget():
    client = fakeredis.FakeRedis(decode_responses=True)
    return BudgetTracker(redis_url="redis://localhost:6379/0", default_limit=1000, default_window=3600.0, _redis_client=client)


class TestBudgetTracker:
    def test_reserve_and_reconcile(self, fake_budget: BudgetTracker) -> None:
        fake_budget.reset("u1")
        fake_budget.set_budget("u1", limit_tokens=1000)
        fake_budget.check_and_reserve("u1", 500)
        fake_budget.record_actual("u1", 500, 300)
        status = fake_budget.status("u1")
        assert status is not None
        assert status["spent"] == 300
        assert status["reserved"] == 0

    def test_budget_exceeded(self, fake_budget: BudgetTracker) -> None:
        fake_budget.reset("u2")
        fake_budget.set_budget("u2", limit_tokens=100)
        fake_budget.check_and_reserve("u2", 60)
        with pytest.raises(BudgetExceededError):
            fake_budget.check_and_reserve("u2", 50)

    def test_window_reset(self, fake_budget: BudgetTracker) -> None:
        fake_budget.reset("u3")
        fake_budget.set_budget("u3", limit_tokens=100, window_seconds=1)
        fake_budget.check_and_reserve("u3", 100)
        import time
        time.sleep(1.1)
        fake_budget.check_and_reserve("u3", 50)
        status = fake_budget.status("u3")
        assert status is not None
        assert status["reserved"] == 50

    def test_concurrent_reserve_no_overspend(self, fake_budget: BudgetTracker) -> None:
        """Concurrent reserves must not overspend the limit."""
        fake_budget.reset("u4")
        fake_budget.set_budget("u4", limit_tokens=1000)
        errors: list[Exception] = []

        def reserve() -> None:
            try:
                fake_budget.check_and_reserve("u4", 100)
            except BudgetExceededError:
                pass
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reserve) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        status = fake_budget.status("u4")
        assert status is not None
        # At most 10 reservations of 100 can fit in 1000
        assert status["reserved"] <= 1000, f"overspend detected: reserved={status['reserved']}"
        assert not errors, f"unexpected errors: {errors}"
