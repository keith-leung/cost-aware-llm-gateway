"""Data models for cost-aware-gateway."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class BudgetExceededError(Exception):
    """Raised when a user's dollar budget is exceeded."""

    def __init__(self, user_id: str, spent: float, limit: float) -> None:
        self.user_id = user_id
        self.spent = spent
        self.limit = limit
        super().__init__(
            f"Budget exceeded for user={user_id}: "
            f"spent=${spent:.6f}, limit=${limit:.6f}"
        )


@dataclass
class RedactionResult:
    """Result of running the AgentLeak redactor on a tool output."""

    cleaned_text: str
    redactions: list[dict[str, Any]] = field(default_factory=list)
    log_safe: bool = True
    llm_flagged: bool = False


@dataclass
class GatewayReply:
    """Per-call record from the gateway."""

    content: str
    usage: dict[str, int]
    tier_used: str
    model_used: str
    budget_after: dict[str, Any]
    breaker_state_after: str
    latency_ms: float
    call_id: str
    cost_usd: float = 0.0
    est_cost_usd: float = 0.0
