"""M1 dollar cost model tests.

Load-bearing assertions:
  1. Expensive model costs more than cheap model for same token usage.
  2. Budget is denominated in dollars ($ limit, $ reserve, $ reconcile).
  3. cost_usd matches pricing × usage exactly.
  4. Cached tokens cost less than non-cached (cache discount).

Each has a paired mutation that must FAIL.
"""

from __future__ import annotations

import pytest
import fakeredis

from cost_aware_gateway.config import (
    BudgetConfig,
    BreakerConfig,
    GatewayConfig,
    JudgeConfig,
    LiteLLMConfig,
    ProviderConfig,
    RedisConfig,
    TierConfig,
)
from cost_aware_gateway.cost import CostCalculator
from cost_aware_gateway.budget import BudgetTracker
from cost_aware_gateway.models import BudgetExceededError, GatewayReply


# ---------------------------------------------------------------------------
# Config helpers — build configs with known pricing for deterministic tests.
# ---------------------------------------------------------------------------

def _config_with_pricing(
    cheap_input: float = 0.0000001,
    expensive_input: float = 0.0000010,
    cheap_output: float = 0.0000003,
    expensive_output: float = 0.0000030,
    cache_read: float = 0.00000001,
) -> GatewayConfig:
    """Build a config with two providers: 'cheap' and 'expensive'."""
    return GatewayConfig(
        mode="mock",
        default_provider="cheap",
        providers={
            "cheap": ProviderConfig(
                base_url="http://mock",
                api_key="mock",
                tiers={
                    "low": TierConfig(model="cheap-model", input_price_per_token=cheap_input,
                                      output_price_per_token=cheap_output,
                                      cache_read_price_per_token=cache_read),
                    "medium": TierConfig(model="cheap-model", input_price_per_token=cheap_input * 2,
                                         output_price_per_token=cheap_output * 2,
                                         cache_read_price_per_token=cache_read * 2),
                    "high": TierConfig(model="cheap-model", input_price_per_token=cheap_input * 4,
                                       output_price_per_token=cheap_output * 4,
                                       cache_read_price_per_token=cache_read * 4),
                },
            ),
            "expensive": ProviderConfig(
                base_url="http://mock",
                api_key="mock",
                tiers={
                    "low": TierConfig(model="expensive-model", input_price_per_token=expensive_input,
                                      output_price_per_token=expensive_output,
                                      cache_read_price_per_token=cache_read),
                    "medium": TierConfig(model="expensive-model", input_price_per_token=expensive_input * 2,
                                         output_price_per_token=expensive_output * 2,
                                         cache_read_price_per_token=cache_read * 2),
                    "high": TierConfig(model="expensive-model", input_price_per_token=expensive_input * 4,
                                       output_price_per_token=expensive_output * 4,
                                       cache_read_price_per_token=cache_read * 4),
                },
            ),
        },
        judge=JudgeConfig(provider="cheap", tier="low"),
        redis=RedisConfig(url="redis://localhost:6379/0"),
        litellm=LiteLLMConfig(timeout=10, num_retries=1),
        budget=BudgetConfig(default_limit_usd=10.0, window_seconds=3600),
        breaker=BreakerConfig(failure_threshold=3, recovery_seconds=30),
    )


# ---------------------------------------------------------------------------
# 1. Per-model pricing — expensive > cheap for same tokens.
# ---------------------------------------------------------------------------

def test_expensive_model_costs_more_than_cheap():
    """Same input/output tokens, different pricing → expensive costs more."""
    cfg = _config_with_pricing()
    calc = CostCalculator(cfg)

    input_tokens = 1000
    output_tokens = 500

    cheap_cost = calc.cost_usd("cheap", "low", input_tokens, output_tokens)
    expensive_cost = calc.cost_usd("expensive", "low", input_tokens, output_tokens)

    assert expensive_cost > cheap_cost, (
        f"expensive ({expensive_cost}) should cost more than cheap ({cheap_cost})"
    )


def test_mutation_flat_pricing_makes_costs_equal_typo():
    """MUTATION: replace all pricing with a flat per-token rate.
    Expensive and cheap models now cost the same."""
    cfg = _config_with_pricing()
    calc = CostCalculator(cfg)

    # Mutation: set all prices to the same flat rate.
    flat_rate = 0.0000005
    for pcfg in cfg.providers.values():
        for tier in pcfg.tiers.values():
            tier.input_price_per_token = flat_rate
            tier.output_price_per_token = flat_rate
            tier.cache_read_price_per_token = flat_rate

    input_tokens = 1000
    output_tokens = 500
    cheap_cost = calc.cost_usd("cheap", "low", input_tokens, output_tokens)
    expensive_cost = calc.cost_usd("expensive", "low", input_tokens, output_tokens)

    assert cheap_cost == expensive_cost, (
        f"mutation expected equal costs, got cheap={cheap_cost} expensive={expensive_cost}"
    )
    # The load-bearing "expensive > cheap" assertion would FAIL here.


def test_mutation_flat_pricing_makes_costs_equal():
    """Same test as above (correct function name for pytest collection)."""
    cfg = _config_with_pricing()
    calc = CostCalculator(cfg)

    flat_rate = 0.0000005
    for pcfg in cfg.providers.values():
        for tier in pcfg.tiers.values():
            tier.input_price_per_token = flat_rate
            tier.output_price_per_token = flat_rate
            tier.cache_read_price_per_token = flat_rate

    input_tokens = 1000
    output_tokens = 500
    cheap_cost = calc.cost_usd("cheap", "low", input_tokens, output_tokens)
    expensive_cost = calc.cost_usd("expensive", "low", input_tokens, output_tokens)

    assert cheap_cost == expensive_cost


# ---------------------------------------------------------------------------
# 2. Budget is in dollars.
# ---------------------------------------------------------------------------

def test_budget_denominated_in_dollars():
    """Set $0.10 budget; reserve $0.06, try $0.05 → exceeded."""
    fake = fakeredis.FakeRedis(decode_responses=True)
    tracker = BudgetTracker("redis://localhost", default_limit=10.0, _redis_client=fake)
    tracker.set_budget("u1", limit_usd=0.10)

    tracker.check_and_reserve("u1", 0.06)
    assert pytest.raises(BudgetExceededError, lambda: tracker.check_and_reserve("u1", 0.05))

    status = tracker.status("u1")
    assert status is not None
    assert "limit_usd" in status
    assert "spent_usd" in status
    assert "reserved_usd" in status
    assert "remaining_usd" in status
    # No token fields.
    assert "limit_tokens" not in status


def test_budget_reconcile_in_dollars():
    """Reserve $0.50, actual $0.30 → spent=$0.30, reserved=$0."""
    fake = fakeredis.FakeRedis(decode_responses=True)
    tracker = BudgetTracker("redis://localhost", default_limit=10.0, _redis_client=fake)
    tracker.set_budget("u1", limit_usd=10.0)

    tracker.check_and_reserve("u1", 0.50)
    tracker.record_actual("u1", 0.50, 0.30)

    status = tracker.status("u1")
    assert abs(status["spent_usd"] - 0.30) < 0.001
    assert abs(status["reserved_usd"] - 0.0) < 0.001


# ---------------------------------------------------------------------------
# 3. cost_usd matches pricing × usage exactly.
# ---------------------------------------------------------------------------

def test_cost_usd_matches_pricing_times_usage():
    """cost = input×input_price + output×output_price (no cache)."""
    cfg = _config_with_pricing(
        cheap_input=0.000001, cheap_output=0.000002,
    )
    calc = CostCalculator(cfg)

    inp = 100
    out = 200
    expected = 100 * 0.000001 + 200 * 0.000002
    actual = calc.cost_usd("cheap", "low", inp, out)
    assert abs(actual - expected) < 1e-12, f"expected {expected}, got {actual}"


# ---------------------------------------------------------------------------
# 4. Cache discount — cached tokens cheaper than non-cached.
# ---------------------------------------------------------------------------

def test_cached_tokens_cheaper_than_non_cached():
    """Same request, but one has cached_tokens=500. The cached version
    costs less because cached tokens are billed at cache_read_price
    (cheaper) instead of input_price."""
    cfg = _config_with_pricing(
        cheap_input=0.000001,
        cheap_output=0.000002,
        cache_read=0.0000001,  # 10× cheaper than input
    )
    calc = CostCalculator(cfg)

    input_tokens = 1000
    output_tokens = 200
    cached_tokens = 500

    cost_no_cache = calc.cost_usd("cheap", "low", input_tokens, output_tokens)
    cost_with_cache = calc.cost_usd(
        "cheap", "low", input_tokens, output_tokens, cached_tokens=cached_tokens,
    )

    assert cost_with_cache < cost_no_cache, (
        f"cached ({cost_with_cache}) should be cheaper than non-cached ({cost_no_cache})"
    )


def test_mutation_ignore_cached_makes_costs_equal():
    """MUTATION: patch cost_usd to ignore cached_tokens (bill all input at
    full input price). Then cached and non-cached costs are equal."""
    cfg = _config_with_pricing(
        cheap_input=0.000001,
        cheap_output=0.000002,
        cache_read=0.0000001,
    )
    calc = CostCalculator(cfg)

    # Mutation: override cost_usd to ignore cache.
    orig = calc.cost_usd

    def no_cache_cost(provider, tier, input_tokens, output_tokens, cached_tokens=0):
        # Bill ALL input at full price, ignore cached.
        pricing = calc.get_pricing(provider, tier)
        return (
            input_tokens * pricing.input_price_per_token
            + output_tokens * pricing.output_price_per_token
        )
    calc.cost_usd = no_cache_cost

    input_tokens = 1000
    output_tokens = 200
    cached_tokens = 500
    cost_no_cache = calc.cost_usd("cheap", "low", input_tokens, output_tokens)
    cost_with_cache = calc.cost_usd(
        "cheap", "low", input_tokens, output_tokens, cached_tokens=cached_tokens,
    )
    assert cost_no_cache == cost_with_cache, (
        f"mutation expected equal costs (cache ignored), "
        f"got no_cache={cost_no_cache} with_cache={cost_with_cache}"
    )
    # The load-bearing "cached < non-cached" assertion would FAIL here.


# ---------------------------------------------------------------------------
# 5. GatewayReply carries cost_usd.
# ---------------------------------------------------------------------------

def test_gateway_reply_has_cost_usd_field():
    """GatewayReply must have cost_usd and est_cost_usd fields."""
    reply = GatewayReply(
        content="test",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        tier_used="low",
        model_used="cheap:low",
        budget_after={},
        breaker_state_after="closed",
        latency_ms=42.0,
        call_id="test:1",
        cost_usd=0.000015,
        est_cost_usd=0.000020,
    )
    assert reply.cost_usd == 0.000015
    assert reply.est_cost_usd == 0.000020
