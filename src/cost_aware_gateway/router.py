"""C3 — LiteLLM Router declarative multi-model routing.

Uses litellm.Router for declarative per-tier routing with fallbacks.
Calls flow through BudgetTracker (pre-call) and CircuitBreaker (post-call).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable

import litellm
from litellm import Router

from cost_aware_gateway.config import GatewayConfig
from cost_aware_gateway.models import CircuitState, GatewayReply
from cost_aware_gateway.budget import BudgetTracker, BudgetExceededError
from cost_aware_gateway.breaker import CircuitBreaker

logger = logging.getLogger(__name__)

# Suppress LiteLLM verbose logging
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


class TieredRouter:
    """Declarative tier-based LLM router backed by LiteLLM Router.

    Caller specifies a tier (low/medium/high), never a model id.
    The router resolves the tier to a model via LiteLLM Router config.

    Every call is wrapped by:
      - BudgetTracker.check_and_reserve()  (pre-call)
      - CircuitBreaker.allow() + record_success/failure (around call)
    """

    def __init__(
        self,
        config: GatewayConfig,
        budget: BudgetTracker,
        call_fn: Callable[..., dict] | None = None,
    ) -> None:
        self._config = config
        self._budget = budget
        self._call_fn = call_fn
        self._router = self._build_router()
        self._default_provider = config.default_provider
        # Map provider -> breaker name (one breaker per provider)
        self._provider_breakers: dict[str, CircuitBreaker] = {}
        for provider_name in config.providers:
            self._provider_breakers[provider_name] = CircuitBreaker(
                name=f"provider:{provider_name}",
                redis_url=config.redis.url,
                failure_threshold=config.breaker.failure_threshold,
                recovery_seconds=config.breaker.recovery_seconds,
            )

    def _build_router(self) -> Router:
        """Build a litellm.Router from config providers."""
        models: list[dict[str, Any]] = []
        for provider_name, pcfg in self._config.providers.items():
            for tier, tcfg in pcfg.tiers.items():
                models.append(
                    {
                        "model_name": f"{provider_name}:{tier}",
                        "litellm_params": {
                            "model": f"openai/{tcfg['model']}",
                            "api_key": pcfg.api_key,
                            "api_base": pcfg.base_url,
                            "timeout": self._config.litellm.timeout,
                            "num_retries": self._config.litellm.num_retries,
                        },
                    }
                )
        if not models:
            raise ValueError("No models configured in providers")
        return Router(model_list=models, default_litellm_params={"timeout": self._config.litellm.timeout})

    def complete(
        self,
        *,
        user_id: str,
        system: str,
        user_message: str,
        tier: str = "medium",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> GatewayReply:
        """Send a completion request through the gateway.

        Steps:
          1. Budget pre-check + reserve
          2. Provider breaker allow()
          3. Route via LiteLLM Router
          4. Budget reconcile
          5. Breaker record success/failure
        """
        call_id = f"{user_id}:{tier}:{time.time():.0f}"
        provider = self._default_provider
        model_name = f"{provider}:{tier}"

        # 1. Budget pre-check
        #    Estimate total tokens as prompt + completion. This is a rough
        #    heuristic (~1 token per 4 characters); the reconcile step corrects
        #    with actual usage from the API response.
        prompt_text = system + user_message
        prompt_est = max(0, (len(prompt_text) - prompt_text.count(" ")) // 4)
        est_tokens = prompt_est + max_tokens
        self._budget.check_and_reserve(user_id, est_tokens)

        # 2. Provider breaker
        breaker = self._provider_breakers.get(provider)
        if breaker is None or not breaker.allow():
            self._budget.record_actual(user_id, est_tokens, 0)
            raise RuntimeError(
                f"Circuit breaker open for provider={provider}; call rejected"
            )

        # 3. Route
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]

        t0 = time.perf_counter()
        try:
            if self._call_fn is not None:
                response = self._call_fn(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                response = self._router.completion(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            latency_ms = (time.perf_counter() - t0) * 1000.0

            content = response.choices[0].message.content or ""
            usage = {}
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                }

            # 4. Budget reconcile
            #    Use total_tokens when available; it includes prompt + completion.
            actual_tokens = usage.get("total_tokens", est_tokens)
            self._budget.record_actual(user_id, est_tokens, actual_tokens)

            # 5. Breaker success
            if breaker is not None:
                breaker.record_success()

            budget_after = self._budget.status(user_id) or {}
            breaker_state = breaker.state.value if breaker else "unknown"

            return GatewayReply(
                content=content,
                usage=usage,
                tier_used=tier,
                model_used=model_name,
                budget_after=budget_after,
                breaker_state_after=breaker_state,
                latency_ms=round(latency_ms, 2),
                call_id=call_id,
            )

        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if breaker is not None:
                breaker.record_failure()
            self._budget.record_actual(user_id, est_tokens, 0)
            raise RuntimeError(
                f"Gateway call failed (provider={provider}, tier={tier}): {exc}"
            ) from exc

    def __repr__(self) -> str:
        return f"TieredRouter(provider={self._default_provider!r}, tiers={list(self._config.providers[self._default_provider].tiers.keys())})"
