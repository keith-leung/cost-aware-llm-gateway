"""Tests for C1 — Redis-backed per-user token budget."""

from __future__ import annotations

import threading

import fakeredis
import pytest
import redis

from cost_aware_gateway.budget import BudgetTracker
from cost_aware_gateway.models import BudgetExceededError


def _real_redis_available() -> bool:
    try:
        client = redis.from_url("redis://localhost:6379/0", socket_timeout=2)
        client.ping()
        client.close()
        return True
    except Exception:
        return False


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

    def test_concurrent_reserve_exact_counts(self, fake_budget: BudgetTracker) -> None:
        """Prove exactly floor(1000/150)=6 succeed and 4 fail, final reserved=900."""
        fake_budget.reset("u5")
        fake_budget.set_budget("u5", limit_tokens=1000)
        successes = 0
        failures = 0
        errors: list[Exception] = []

        def reserve() -> None:
            nonlocal successes, failures
            try:
                fake_budget.check_and_reserve("u5", 150)
                successes += 1
            except BudgetExceededError:
                failures += 1
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reserve) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        status = fake_budget.status("u5")
        assert status is not None
        assert successes == 6, f"expected 6 successes, got {successes}"
        assert failures == 4, f"expected 4 failures, got {failures}"
        assert status["reserved"] == 900, f"expected reserved=900, got {status['reserved']}"
        assert not errors, f"unexpected errors: {errors}"


@pytest.mark.skipif(not _real_redis_available(), reason="Real Redis not available at localhost:6379/0")
class TestRealRedisBudget:
    """Budget atomicity proof against real Redis (not fakeredis)."""

    def test_concurrent_reserve_real_redis(self) -> None:
        client = redis.from_url("redis://localhost:6379/0", decode_responses=True)
        budget = BudgetTracker(redis_url="redis://localhost:6379/0", default_limit=1000, default_window=3600.0, _redis_client=client)
        user = "stress-user"
        budget.reset(user)
        budget.set_budget(user, limit_tokens=1000)

        errors: list[Exception] = []

        def reserve() -> None:
            try:
                budget.check_and_reserve(user, 100)
            except BudgetExceededError:
                pass
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reserve) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        status = budget.status(user)
        assert status is not None
        assert status["reserved"] <= 1000, f"overspend detected: reserved={status['reserved']}"
        assert not errors, f"unexpected errors: {errors}"
        budget.reset(user)
