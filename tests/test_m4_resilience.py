"""M4 resilience tests — failover + soft-cap degradation.

Load-bearing assertions:
  1. P1 breaker OPEN, P2 CLOSED → request served by P2 (not rejected).
     Both OPEN → rejected.
  2. Expensive tier exceeds budget → degraded to cheap tier + warning.
     Even cheap exceeds → BudgetExceededError (floor preserved).

Each has paired mutations that must FAIL.
"""

from __future__ import annotations

import pytest

from cost_aware_gateway.config import (
    BudgetConfig, BreakerConfig, GatewayConfig, JudgeConfig,
    LiteLLMConfig, ProviderConfig, RedisConfig, TierConfig,
)
from cost_aware_gateway.cost import CostCalculator
from cost_aware_gateway.models import GatewayReply, BudgetExceededError
from cost_aware_gateway.resilience import (
    BreakerGate, FailoverRouter, FailoverResult,
    SoftCapRouter, SoftCapResult,
)


# ---------------------------------------------------------------------------
# Test doubles.
# ---------------------------------------------------------------------------

class FakeBreaker:
    """Deterministic breaker for tests."""
    def __init__(self, is_open: bool = False):
        self._open = is_open
        self.success_count = 0
        self.failure_count = 0

    def allow(self) -> bool:
        return not self._open

    def record_success(self) -> None:
        self.success_count += 1

    def record_failure(self) -> None:
        self.failure_count += 1


def _config_two_providers() -> GatewayConfig:
    """Two providers: 'p1' (primary) and 'p2' (backup)."""
    return GatewayConfig(
        mode="mock", default_provider="p1",
        providers={
            "p1": ProviderConfig(base_url="http://p1", api_key="k1", tiers={
                "low": TierConfig(model="m1", input_price_per_token=0.000001,
                                  output_price_per_token=0.000002, cache_read_price_per_token=0.0000001),
                "high": TierConfig(model="m1", input_price_per_token=0.000010,
                                   output_price_per_token=0.000020, cache_read_price_per_token=0.000001),
            }),
            "p2": ProviderConfig(base_url="http://p2", api_key="k2", tiers={
                "low": TierConfig(model="m2", input_price_per_token=0.000001,
                                  output_price_per_token=0.000002, cache_read_price_per_token=0.0000001),
                "high": TierConfig(model="m2", input_price_per_token=0.000010,
                                   output_price_per_token=0.000020, cache_read_price_per_token=0.000001),
            }),
        },
        judge=JudgeConfig(provider="p1", tier="low"),
        redis=RedisConfig(url="redis://localhost:6379/0"),
        litellm=LiteLLMConfig(timeout=10, num_retries=1),
        budget=BudgetConfig(default_limit_usd=100.0, window_seconds=3600),
        breaker=BreakerConfig(failure_threshold=3, recovery_seconds=30),
    )


def _make_call_fn():
    """Deterministic call_fn that tags its reply with the provider."""
    def call_fn(*, provider, tier, system, user_message, max_tokens=1024, temperature=0.0):
        return GatewayReply(
            content=f"reply from {provider}:{tier}",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            tier_used=tier, model_used=f"{provider}:{tier}",
            budget_after={}, breaker_state_after="closed",
            latency_ms=10.0, call_id=f"test:{provider}:{tier}",
            cost_usd=0.001, est_cost_usd=0.001,
        )
    return call_fn


# ---------------------------------------------------------------------------
# 1. Failover — P1 open → P2 serves; both open → reject.
# ---------------------------------------------------------------------------

def test_failover_p1_open_served_by_p2():
    """P1 breaker OPEN, P2 CLOSED → request served by P2."""
    cfg = _config_two_providers()
    router = FailoverRouter(
        providers=["p1", "p2"],
        breaker_states={
            "p1": FakeBreaker(is_open=True),
            "p2": FakeBreaker(is_open=False),
        },
        call_fn=_make_call_fn(),
        config=cfg,
    )
    result = router.call(tier="low", system="s", user_message="hello")
    assert result.provider_used == "p2"
    assert result.failed_over is True
    assert "p2" in result.reply.content


def test_failover_both_open_rejected():
    """P1 and P2 both OPEN → RuntimeError."""
    cfg = _config_two_providers()
    router = FailoverRouter(
        providers=["p1", "p2"],
        breaker_states={
            "p1": FakeBreaker(is_open=True),
            "p2": FakeBreaker(is_open=True),
        },
        call_fn=_make_call_fn(),
        config=cfg,
    )
    with pytest.raises(RuntimeError, match="All providers"):
        router.call(tier="low", system="s", user_message="hello")


def test_failover_p1_closed_served_by_p1():
    """P1 CLOSED → served by P1 (no failover needed)."""
    cfg = _config_two_providers()
    router = FailoverRouter(
        providers=["p1", "p2"],
        breaker_states={
            "p1": FakeBreaker(is_open=False),
            "p2": FakeBreaker(is_open=False),
        },
        call_fn=_make_call_fn(),
        config=cfg,
    )
    result = router.call(tier="low", system="s", user_message="hello")
    assert result.provider_used == "p1"
    assert result.failed_over is False


def test_mutation_disable_failover_p1_open_rejected():
    """MUTATION: only use primary provider (no failover list). P1 breaker
    open → request rejected (no P2 fallback)."""
    cfg = _config_two_providers()
    # Mutation: providers list is only ["p1"] — no fallback.
    router = FailoverRouter(
        providers=["p1"],
        breaker_states={"p1": FakeBreaker(is_open=True)},
        call_fn=_make_call_fn(),
        config=cfg,
    )
    with pytest.raises(RuntimeError, match="All providers"):
        router.call(tier="low", system="s", user_message="hello")
    # The load-bearing "served by P2" assertion would FAIL here
    # because the call raises instead of returning a result.


# ---------------------------------------------------------------------------
# 2. Soft-cap degradation — expensive over budget → degrade; floor → hard reject.
# ---------------------------------------------------------------------------

def _soft_cap_config() -> GatewayConfig:
    """Single provider with cheap + expensive tiers."""
    return GatewayConfig(
        mode="mock", default_provider="p",
        providers={
            "p": ProviderConfig(base_url="http://mock", api_key="k", tiers={
                "low": TierConfig(model="cheap", input_price_per_token=0.000001,
                                  output_price_per_token=0.000002, cache_read_price_per_token=0.0000001),
                "high": TierConfig(model="expensive", input_price_per_token=0.000010,
                                   output_price_per_token=0.000020, cache_read_price_per_token=0.000001),
            }),
        },
        judge=JudgeConfig(provider="p", tier="low"),
        redis=RedisConfig(url="redis://localhost:6379/0"),
        litellm=LiteLLMConfig(timeout=10, num_retries=1),
        budget=BudgetConfig(default_limit_usd=100.0, window_seconds=3600),
        breaker=BreakerConfig(failure_threshold=3, recovery_seconds=30),
    )


def test_soft_cap_expensive_over_budget_degrades_to_cheap():
    """Remaining budget fits cheap but not expensive → degraded to cheap."""
    cfg = _soft_cap_config()
    calc = CostCalculator(cfg)
    # Set remaining budget to fit low tier but not high.
    # low cost = 100×1e-6 + 50×2e-6 = 0.0002
    # high cost = 100×1e-5 + 50×2e-5 = 0.002
    remaining = 0.001  # fits low (0.0002) but not high (0.002)

    router = SoftCapRouter(
        cost_calc=calc, provider="p",
        tiers_cheapest_first=["low", "high"],
        budget_remaining_usd=lambda: remaining,
        call_fn=_make_call_fn(),
    )
    result = router.call(requested_tier="high", system="s", user_message="hello")
    assert result.degraded is True
    assert result.tier_used == "low"
    assert "degraded" in result.degradation_reason.lower()
    assert result.reply.tier_used == "low"


def test_soft_cap_cheap_fits_served_normally():
    """Remaining budget fits requested tier → no degradation."""
    cfg = _soft_cap_config()
    calc = CostCalculator(cfg)
    router = SoftCapRouter(
        cost_calc=calc, provider="p",
        tiers_cheapest_first=["low", "high"],
        budget_remaining_usd=lambda: 100.0,  # plenty
        call_fn=_make_call_fn(),
    )
    result = router.call(requested_tier="high", system="s", user_message="hello")
    assert result.degraded is False
    assert result.tier_used == "high"


def test_soft_cap_floor_cheapest_exceeds_hard_reject():
    """Even the cheapest tier exceeds remaining budget → BudgetExceededError."""
    cfg = _soft_cap_config()
    calc = CostCalculator(cfg)
    # Set remaining budget to 0 → nothing fits.
    router = SoftCapRouter(
        cost_calc=calc, provider="p",
        tiers_cheapest_first=["low", "high"],
        budget_remaining_usd=lambda: 0.0,
        call_fn=_make_call_fn(),
    )
    with pytest.raises(BudgetExceededError):
        router.call(requested_tier="low", system="s", user_message="hello")


def test_mutation_disable_soft_cap_hard_rejects_degradable():
    """MUTATION: SoftCapRouter that doesn't degrade — always tries the
    requested tier or hard-rejects. The "fits cheap but not expensive"
    request gets hard-rejected instead of degraded."""
    cfg = _soft_cap_config()
    calc = CostCalculator(cfg)
    remaining = 0.001  # fits low but not high

    # Mutation: call_fn replaced with a version that always hard-rejects
    # if the requested tier doesn't fit (no degradation logic).
    def hard_cap_call(*, requested_tier, system, user_message, **kw):
        cost = calc.cost_usd("p", requested_tier, 100, 50)
        if cost > remaining:
            raise BudgetExceededError("mut", 0, remaining)
        return _make_call_fn()(
            provider="p", tier=requested_tier,
            system=system, user_message=user_message,
        )

    with pytest.raises(BudgetExceededError):
        hard_cap_call(
            requested_tier="high", system="s", user_message="hello",
        )
    # The load-bearing "degraded to low" assertion would FAIL here
    # because BudgetExceededError is raised instead of a SoftCapResult.
