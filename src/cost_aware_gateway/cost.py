"""M1 — Per-model dollar cost calculator.

Config-driven: prices come from the TierConfig on each provider.
The calculator is a pure function of (pricing, usage) — no hardcoded
per-model if-statements.
"""

from __future__ import annotations

from cost_aware_gateway.config import GatewayConfig, TierConfig


class CostCalculator:
    """Compute dollar cost from token usage + per-tier pricing.

    Prices are declared in config (input / output / cache_read per token).
    The calculator looks up the (provider, tier) pair and applies:

        cost = input_tokens × input_price
             + output_tokens × output_price
             + cached_tokens × cache_read_price

    Cached tokens are billed at the cache_read price (typically a fraction
    of the input price) INSTEAD of the input price — they are not double-
    charged.
    """

    def __init__(self, config: GatewayConfig) -> None:
        self._config = config

    def get_pricing(self, provider: str, tier: str) -> TierConfig:
        """Return the TierConfig for (provider, tier). Raises KeyError."""
        pcfg = self._config.providers[provider]
        return pcfg.tiers[tier]

    def cost_usd(
        self,
        provider: str,
        tier: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        """Compute the dollar cost of a call.

        ``cached_tokens`` are billed at ``cache_read_price_per_token``
        INSTEAD of ``input_price_per_token``. The remaining
        ``input_tokens - cached_tokens`` are billed at the full input price.
        """
        pricing = self.get_pricing(provider, tier)
        billable_input = max(0, input_tokens - cached_tokens)
        return (
            billable_input * pricing.input_price_per_token
            + output_tokens * pricing.output_price_per_token
            + cached_tokens * pricing.cache_read_price_per_token
        )
