"""Tests for C2 — Redis-backed 3-state circuit breaker."""

from __future__ import annotations

import time

import fakeredis
import pytest

from cost_aware_gateway.breaker import CircuitBreaker
from cost_aware_gateway.models import CircuitState


@pytest.fixture()
def redis_client():
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture()
def breaker(redis_client):
    b = CircuitBreaker(
        name="test-provider",
        redis_url="redis://localhost:6379/0",
        failure_threshold=2,
        recovery_seconds=1,
        _redis_client=redis_client,
    )
    b.reset()
    return b


class TestCircuitBreaker:
    def test_initial_state_closed(self, breaker: CircuitBreaker) -> None:
        assert breaker.state == CircuitState.CLOSED

    def test_trip_to_open(self, breaker: CircuitBreaker) -> None:
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

    def test_allow_blocks_when_open(self, breaker: CircuitBreaker) -> None:
        breaker.record_failure()
        breaker.record_failure()
        assert not breaker.allow()

    def test_recovery_to_half_open(self, breaker: CircuitBreaker) -> None:
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        time.sleep(1.1)
        # After recovery window, allow should try to acquire probe lock
        allowed = breaker.allow()
        assert allowed is True or breaker.state == CircuitState.HALF_OPEN

    def test_success_closes_breaker(self, breaker: CircuitBreaker) -> None:
        breaker.record_failure()
        breaker.record_failure()
        time.sleep(1.1)
        breaker.allow()
        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

    def test_shared_state_across_instances(self, breaker: CircuitBreaker, redis_client) -> None:
        b2 = CircuitBreaker(
            name="test-provider",
            redis_url="redis://localhost:6379/0",
            failure_threshold=2,
            recovery_seconds=1,
            _redis_client=redis_client,
        )
        breaker.record_failure()
        breaker.record_failure()
        assert b2.state == CircuitState.OPEN
