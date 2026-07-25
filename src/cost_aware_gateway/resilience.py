"""M4 — Resilience: cross-provider failover + soft-cap degradation.

Two mechanisms layered ON TOP of the existing breaker/budget, not replacing
the DoW hard defenses:

  * ``FailoverRouter`` — iterates providers in order. If the primary
    provider's breaker is open, it fails over to the next provider whose
    breaker is closed. Only when ALL providers' breakers are open does it
    reject (preserving the "no available → reject" floor).

  * ``SoftCapRouter`` — when the target tier's estimated cost exceeds the
    remaining budget, it degrades to a cheaper tier that fits (with a
    degradation warning). Only when even the cheapest tier doesn't fit
    does it raise ``BudgetExceededError`` (preserving the DoW floor).

Both use injectable breaker-state + budget-status for deterministic tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from cost_aware_gateway.config import GatewayConfig
from cost_aware_gateway.cost import CostCalculator
from cost_aware_gateway.models import GatewayReply, BudgetExceededError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Breaker-state protocol (injectable for tests; production uses real
# CircuitBreaker).
# ---------------------------------------------------------------------------

class BreakerGate(Protocol):
    """Minimal breaker interface needed by FailoverRouter."""
    def allow(self) -> bool: ...
    def record_success(self) -> None: ...
    def record_failure(self) -> None: ...


# ---------------------------------------------------------------------------
# Failover result.
# ---------------------------------------------------------------------------

@dataclass
class FailoverResult:
    """Outcome of a failover call."""
    reply: GatewayReply
    provider_used: str
    providers_tried: list[str]
    failed_over: bool


class FailoverRouter:
    """Cross-provider failover: primary breaker open → try backup.

    Iterates ``providers`` in order. For each, checks ``breaker_state[provider].allow()``.
    First provider whose breaker allows → calls ``call_fn(provider, ...)``.
    If all are open → raises RuntimeError (no available provider).

    This preserves the floor: "all open → reject". The existing breaker
    tests (single-provider) are unchanged because FailoverRouter is a
    new layer, not a modification of CircuitBreaker.
    """

    def __init__(
        self,
        *,
        providers: list[str],
        breaker_states: dict[str, BreakerGate],
        call_fn: Callable[..., GatewayReply],
        config: GatewayConfig,
    ) -> None:
        self._providers = providers
        self._breaker_states = breaker_states
        self._call_fn = call_fn
        self._config = config

    def call(
        self,
        *,
        tier: str,
        system: str,
        user_message: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> FailoverResult:
        """Try providers in order; fail over on open breaker."""
        tried: list[str] = []
        for provider in self._providers:
            tried.append(provider)
            breaker = self._breaker_states.get(provider)
            if breaker is None or not breaker.allow():
                logger.warning("provider %s breaker open, trying next", provider)
                continue
            # Breaker allows → call.
            try:
                reply = self._call_fn(
                    provider=provider, tier=tier,
                    system=system, user_message=user_message,
                    max_tokens=max_tokens, temperature=temperature,
                )
                breaker.record_success()
                return FailoverResult(
                    reply=reply,
                    provider_used=provider,
                    providers_tried=tried,
                    failed_over=(provider != self._providers[0]),
                )
            except Exception:
                breaker.record_failure()
                logger.warning("provider %s call failed, trying next", provider)
                continue

        # All providers exhausted.
        raise RuntimeError(
            f"All providers have open breakers or failed: tried={tried}"
        )


# ---------------------------------------------------------------------------
# Soft-cap degradation.
# ---------------------------------------------------------------------------

@dataclass
class SoftCapResult:
    """Outcome of a soft-cap call."""
    reply: GatewayReply
    tier_used: str
    requested_tier: str
    degraded: bool           # True if served by a cheaper tier than requested
    degradation_reason: str = ""


class SoftCapRouter:
    """Budget-aware tier degradation: expensive too costly → degrade to cheap.

    Before calling the model, estimates the cost of the requested tier. If
    it exceeds the remaining budget, tries cheaper tiers in order until one
    fits. Only when even the cheapest tier doesn't fit does it raise
    ``BudgetExceededError`` (preserving the DoW floor).

    ``budget_remaining_usd`` is an injectable callable so tests can set
    exact remaining amounts. ``call_fn`` is the model-call stub.
    """

    def __init__(
        self,
        *,
        cost_calc: CostCalculator,
        provider: str,
        tiers_cheapest_first: list[str],
        budget_remaining_usd: Callable[[], float],
        call_fn: Callable[..., GatewayReply],
        est_input_tokens: int = 100,
        est_output_tokens: int = 50,
    ) -> None:
        self._calc = cost_calc
        self._provider = provider
        self._tiers = tiers_cheapest_first  # e.g. ["low", "medium", "high"]
        self._budget_remaining = budget_remaining_usd
        self._call_fn = call_fn
        self._est_input = est_input_tokens
        self._est_output = est_output_tokens

    def call(
        self,
        *,
        requested_tier: str,
        system: str,
        user_message: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> SoftCapResult:
        """Serve the request, degrading tier if budget is tight."""
        remaining = self._budget_remaining()

        # Check if the requested tier fits.
        requested_cost = self._calc.cost_usd(
            self._provider, requested_tier,
            self._est_input, self._est_output,
        )
        if requested_cost <= remaining:
            # Requested tier fits → serve it.
            reply = self._call_fn(
                provider=self._provider, tier=requested_tier,
                system=system, user_message=user_message,
                max_tokens=max_tokens, temperature=temperature,
            )
            return SoftCapResult(
                reply=reply, tier_used=requested_tier,
                requested_tier=requested_tier, degraded=False,
            )

        # Requested tier too expensive → try cheaper tiers.
        # Find tiers cheaper than requested, in cheapest-first order.
        requested_idx = self._tiers.index(requested_tier) if requested_tier in self._tiers else len(self._tiers)
        cheaper_tiers = self._tiers[:requested_idx]

        for cheaper in cheaper_tiers:
            cheaper_cost = self._calc.cost_usd(
                self._provider, cheaper,
                self._est_input, self._est_output,
            )
            if cheaper_cost <= remaining:
                # This cheaper tier fits → degrade to it.
                reply = self._call_fn(
                    provider=self._provider, tier=cheaper,
                    system=system, user_message=user_message,
                    max_tokens=max_tokens, temperature=temperature,
                )
                return SoftCapResult(
                    reply=reply, tier_used=cheaper,
                    requested_tier=requested_tier, degraded=True,
                    degradation_reason=(
                        f"requested tier '{requested_tier}' "
                        f"(${requested_cost:.6f}) exceeds remaining budget "
                        f"(${remaining:.6f}); degraded to '{cheaper}' "
                        f"(${cheaper_cost:.6f})"
                    ),
                )

        # Even the cheapest tier doesn't fit → hard reject.
        cheapest_cost = self._calc.cost_usd(
            self._provider, self._tiers[0],
            self._est_input, self._est_output,
        ) if self._tiers else requested_cost
        raise BudgetExceededError(
            user_id="soft-cap",
            spent=0.0,
            limit=remaining,
        )
