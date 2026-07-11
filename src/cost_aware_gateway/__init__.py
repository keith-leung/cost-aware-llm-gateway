"""cost_aware_gateway — Redis-backed cost-aware LLM gateway.

Components:
  C1  BudgetTracker   — per-user token budget with two-step reserve-then-reconcile
  C2  CircuitBreaker  — 3-state breaker with Redis backing + SETNX single-probe
  C3  TieredRouter    — LiteLLM Router declarative multi-model routing
  C5  Redactor        — AgentLeak runtime credential redaction
"""

from cost_aware_gateway.config import load_config
from cost_aware_gateway.models import (
    BudgetExceededError,
    CircuitState,
    GatewayReply,
    RedactionResult,
)
from cost_aware_gateway.budget import BudgetTracker
from cost_aware_gateway.breaker import CircuitBreaker
from cost_aware_gateway.router import TieredRouter
from cost_aware_gateway.redact import redact_tool_output

__all__ = [
    "load_config",
    "BudgetExceededError",
    "CircuitState",
    "GatewayReply",
    "RedactionResult",
    "BudgetTracker",
    "CircuitBreaker",
    "TieredRouter",
    "redact_tool_output",
]
