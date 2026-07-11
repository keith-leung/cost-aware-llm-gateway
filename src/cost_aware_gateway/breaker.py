"""C2 — Redis-backed 3-state circuit breaker using pybreaker.

Uses pybreaker as the 3-state machine core (closed/open/half-open) with a
custom Redis storage backend.  The half-open single-probe serialization is
guarded by a Redis SETNX lock — pybreaker does NOT primary-source-confirm
this guarantee, so we build it ourselves.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

import pybreaker
import redis

from cost_aware_gateway.models import CircuitState

logger = logging.getLogger(__name__)

_STATE_KEY = "cow:breaker:{name}:state"
_FAILURES_KEY = "cow:breaker:{name}:failures"
_OPENED_AT_KEY = "cow:breaker:{name}:opened_at"
_HALF_OPEN_LOCK_KEY = "cow:breaker:{name}:half_open_lock"


class RedisBreakerStorage:
    """Redis-backed storage for pybreaker state.

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
        """SETNX lock for single-probe serialization in half-open state.

        Only one worker across all instances can acquire this lock.
        """
        acquired = self._redis.set(
            self._lock_key, self._lock_token, nx=True, ex=int(expire)
        )
        return bool(acquired)

    def release_half_open_lock(self) -> None:
        """Release the half-open lock if we hold it."""
        script = """
        if redis.call('GET', KEYS[1]) == ARGV[1] then
            return redis.call('DEL', KEYS[1])
        end
        return 0
        """
        self._redis.eval(script, 1, self._lock_key, self._lock_token)

    def clear(self) -> None:
        """Reset all state (test helper)."""
        self._redis.delete(
            self._state_key,
            self._failures_key,
            self._opened_at_key,
            self._lock_key,
        )


class _BreakerListener(pybreaker.CircuitBreakerListener):
    """Listener that syncs pybreaker state transitions to Redis."""

    def __init__(self, storage: RedisBreakerStorage) -> None:
        self._storage = storage

    def state_change(
        self,
        cb: pybreaker.CircuitBreaker,
        old_state: pybreaker.CircuitBreakerState,
        new_state: pybreaker.CircuitBreakerState,
    ) -> None:
        try:
            state = CircuitState(new_state.name)
            self._storage.set_state(state)
            if state == CircuitState.OPEN:
                self._storage.set_opened_at(time.time())
        except ValueError:
            pass

    def failure(self, cb: pybreaker.CircuitBreaker, exc: BaseException) -> None:
        self._storage.increment_failure_count()

    def success(self, cb: pybreaker.CircuitBreaker) -> None:
        self._storage.reset_failure_count()


class CircuitBreaker:
    """Redis-backed 3-state circuit breaker built on pybreaker.

    State machine: closed -> open (after failure_threshold failures) ->
    half_open (after recovery_seconds) -> closed on probe success /
    re-open on probe failure.

    Atomicity mechanisms:
      - State transitions use pybreaker's listener callbacks backed by Redis.
      - Half-open single-probe serialization uses SETNX (documented above).
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
        self._breaker = self._build_breaker()

    def _build_breaker(self) -> pybreaker.CircuitBreaker:
        listener = _BreakerListener(self._storage)
        return pybreaker.CircuitBreaker(
            fail_max=self._failure_threshold,
            reset_timeout=self._recovery_seconds,
            name=self._name,
            listeners=[listener],
        )

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
                    # Transition to half-open via SETNX single-probe
                    if self._storage.acquire_half_open_lock(
                        expire=self._recovery_seconds
                    ):
                        self._storage.set_state(CircuitState.HALF_OPEN)
                        logger.debug(
                            "Breaker %s transitioned to half_open (probe acquired)",
                            self._name,
                        )
                        return True
                    # Another worker is probing; stay open
                    return False
                return False
            # half_open: only allow if we hold the probe lock
            return self._storage.acquire_half_open_lock(
                expire=self._recovery_seconds
            )

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._breaker.call(lambda: None)

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            try:
                self._breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
            except pybreaker.CircuitBreakerError:
                pass
            except RuntimeError:
                pass

    def reset(self) -> None:
        """Force reset the breaker."""
        with self._lock:
            self._breaker.close()
        self._storage.clear()

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(name={self._name!r}, state={self.state.value}, "
            f"threshold={self._failure_threshold}, recovery={self._recovery_seconds}s)"
        )
