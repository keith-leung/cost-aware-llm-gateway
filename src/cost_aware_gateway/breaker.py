"""C2 — Hand-rolled Redis-backed 3-state circuit breaker.

State machine: closed → open (after failure_threshold Redis-aggregated
failures) → half_open (after recovery_seconds) → closed on probe success /
re-open on probe failure.

All state lives in Redis so multiple CircuitBreaker instances (workers) share
the same failure counts, state, and opened_at timestamp. The open/close
decision is driven entirely by Redis-aggregated counts, not process-local state.

Half-open single-probe serialization uses a Redis SETNX lock.

(R4 note: pybreaker was removed — its listener + _breaker.call had become dead
code once record_failure/record_success were rewired to drive Redis directly.
A hand-rolled 3-state machine on Redis is more honest than importing pybreaker
as a name-drop while not actually using its decision logic.)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import redis

from cost_aware_gateway.models import CircuitState

logger = logging.getLogger(__name__)

_STATE_KEY = "cow:breaker:{name}:state"
_FAILURES_KEY = "cow:breaker:{name}:failures"
_OPENED_AT_KEY = "cow:breaker:{name}:opened_at"
_HALF_OPEN_LOCK_KEY = "cow:breaker:{name}:half_open_lock"


class RedisBreakerStorage:
    """Redis-backed storage for breaker state.

    Stores state, failure count, opened_at timestamp, and a SETNX lock
    for half-open single-probe serialization.
    """

    def __init__(self, redis_client: redis.Redis, name: str) -> None:
        self._redis = redis_client
        self._name = name
        self._state_key = _STATE_KEY.format(name=name)
        self._failures_key = _FAILURES_KEY.format(name=name)
        self._opened_at_key = _OPENED_AT_KEY.format(name=name)
        self._lock_key = _HALF_OPEN_LOCK_KEY.format(name=name)
        self._lock_token = f"{time.time()}_{threading.get_native_id()}"

    def get_state(self) -> str:
        raw = self._redis.get(self._state_key)
        if raw is None:
            return "closed"
        return raw

    def set_state(self, state: CircuitState | str) -> None:
        if isinstance(state, CircuitState):
            state = state.value
        self._redis.set(self._state_key, state)

    def increment_failure_count(self) -> int:
        return self._redis.incr(self._failures_key)

    def reset_failure_count(self) -> None:
        self._redis.delete(self._failures_key)

    def get_failure_count(self) -> int:
        val = self._redis.get(self._failures_key)
        return int(val) if val is not None else 0

    def set_opened_at(self, timestamp: float) -> None:
        self._redis.set(self._opened_at_key, timestamp)

    def get_opened_at(self) -> float | None:
        val = self._redis.get(self._opened_at_key)
        return float(val) if val is not None else None

    def acquire_half_open_lock(self, expire: float = 30.0) -> bool:
        """SETNX lock for single-probe serialization in half-open state."""
        acquired = self._redis.set(
            self._lock_key, self._lock_token, nx=True, ex=int(expire)
        )
        return bool(acquired)

    def release_half_open_lock(self) -> None:
        """Release the half-open lock if we hold it.

        Uses a Lua compare-and-delete for atomicity on real Redis. Falls back
        to GET-check-DEL if EVAL is unavailable (fakeredis).
        """
        script = """
        if redis.call('GET', KEYS[1]) == ARGV[1] then
            return redis.call('DEL', KEYS[1])
        end
        return 0
        """
        try:
            self._redis.eval(script, 1, self._lock_key, self._lock_token)
        except Exception:
            current = self._redis.get(self._lock_key)
            if current == self._lock_token:
                self._redis.delete(self._lock_key)

    def clear(self) -> None:
        """Reset all state (test helper)."""
        self._redis.delete(
            self._state_key,
            self._failures_key,
            self._opened_at_key,
            self._lock_key,
        )


class CircuitBreaker:
    """Hand-rolled Redis-backed 3-state circuit breaker.

    State machine: closed -> open (after failure_threshold Redis-aggregated
    failures) -> half_open (after recovery_seconds) -> closed on probe
    success / re-open on probe failure.

    The open/close decision is driven entirely by Redis-aggregated failure
    counts — every record_failure does an atomic INCR on the Redis counter,
    and when it reaches failure_threshold, sets state=OPEN in Redis. This
    makes the breaker genuinely cross-worker: 3 workers each failing once
    will trip a threshold=3 breaker because Redis sees count=3.

    Atomicity mechanisms:
      - Failure counting: Redis INCR (atomic across workers).
      - Half-open single-probe: SETNX lock.
    """

    def __init__(
        self,
        name: str,
        redis_url: str,
        failure_threshold: int = 3,
        recovery_seconds: float = 30.0,
        _redis_client: redis.Redis | None = None,
    ) -> None:
        self._name = name
        self._failure_threshold = failure_threshold
        self._recovery_seconds = recovery_seconds
        if _redis_client is not None:
            self._redis = _redis_client
        else:
            self._redis = redis.from_url(redis_url, decode_responses=True)
        self._storage = RedisBreakerStorage(self._redis, name)
        self._lock = threading.Lock()

    # --- Public API ---

    @property
    def state(self) -> CircuitState:
        """Current breaker state, read from Redis."""
        raw = self._storage.get_state()
        if isinstance(raw, CircuitState):
            return raw
        try:
            return CircuitState(raw)
        except ValueError:
            return CircuitState.CLOSED

    def allow(self) -> bool:
        """Return True if a call is allowed through the breaker."""
        with self._lock:
            st = self._storage.get_state()
            if st == "closed":
                return True
            if st == "open":
                opened_at = self._storage.get_opened_at()
                if opened_at is None:
                    return True
                if time.time() - opened_at >= self._recovery_seconds:
                    if self._storage.acquire_half_open_lock(
                        expire=self._recovery_seconds
                    ):
                        self._storage.set_state(CircuitState.HALF_OPEN)
                        logger.debug(
                            "Breaker %s transitioned to half_open (probe acquired)",
                            self._name,
                        )
                        return True
                    return False
                return False
            # half_open: only allow if we hold the probe lock
            return self._storage.acquire_half_open_lock(
                expire=self._recovery_seconds
            )

    def record_success(self) -> None:
        """Record a successful call: clear Redis failure count + close breaker."""
        with self._lock:
            self._storage.reset_failure_count()
            self._storage.set_state(CircuitState.CLOSED)
            self._storage.release_half_open_lock()

    def record_failure(self) -> None:
        """Record a failed call via Redis-aggregated failure count.

        Increments the Redis failure counter atomically; when it reaches
        failure_threshold, sets state=OPEN + opened_at in Redis.
        """
        with self._lock:
            count = self._storage.increment_failure_count()
            if count >= self._failure_threshold:
                self._storage.set_state(CircuitState.OPEN)
                self._storage.set_opened_at(time.time())

    def reset(self) -> None:
        """Force reset the breaker."""
        with self._lock:
            pass
        self._storage.clear()

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(name={self._name!r}, state={self.state.value}, "
            f"threshold={self._failure_threshold}, recovery={self._recovery_seconds}s)"
        )
