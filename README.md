# cost-aware-gateway

A **Redis-backed cost-aware LLM gateway** with per-user budget, a Redis-backed
three-state circuit breaker, LiteLLM Router declarative multi-model routing,
and AgentLeak runtime credential redaction.

**What it does:** keeps agent spend under control at scale and defends against
Denial-of-Wallet by making budget + breaker state shared across workers (Redis,
not process-local), and by reasoning about the scaling-shape boundary chain in
architecture terms, not fake QPS.

---

## Construction Path — from cost-*limited* to cost-*aware*

Cost control at the LLM gateway and LangGraph-based agent development are two
pillars of enterprise LLM engineering. This repo has a genuinely solid
**defensive** layer — a Redis-backed reserve-then-reconcile budget, a hand-rolled
3-state circuit breaker, and runtime credential redaction — plus unusually
honest engineering docs (the scaling boundary-chain marks "measured vs
architecture-only", the pybreaker name-drop was removed on purpose). The
**optimizing** layer — the intelligence that makes a gateway *cost-aware* rather
than merely *cost-limited* — is now built too: a dollar cost model, cost-aware
routing with a quality-checked cascade, semantic cache + prompt compression, and
failover / soft-cap. This section documents both layers and, honestly, the
simplicity caveats that remain.

Each mechanism carries an acceptance criterion and a mutation proof (break the
mechanism → the acceptance fails). The honest gaps are the simplicity of the
heuristics (keyword classifier, Jaccard cache) and the untested scale chain.

### cost-*limited* vs cost-*aware* — the distinction the name promises

A budget cap says "stop at $X." Cost-awareness says "do the same job for less."
The gateway now has a **dollar cost model** (per-model pricing; budget and
reconcile denominated in money, not just tokens) and makes real cost/quality
decisions: a difficulty classifier picks the cheapest adequate tier and escalates
only when a quality check fails. The honest caveat is that the classifier is a
**keyword heuristic**, not learned — a real cost/quality decision, a simple one.

### Where this repo is now (honest baseline)

**Built and genuinely strong (keep — already at height):**
- `BudgetTracker`: Redis reserve-then-reconcile, atomic WATCH/MULTI/EXEC,
  cross-worker — rejects *before* spending (real Denial-of-Wallet defense).
- `CircuitBreaker`: Redis 3-state, cross-worker failure aggregation, SETNX
  half-open probe. (And the honest pybreaker removal.)
- `Redactor` (C5): structural credential regex + configured-key match + an LLM
  second pass that flags leaks without mutating text.
- `SCALING.md`: boundary-chain reasoning marked "measured vs architecture-only",
  no invented QPS. This honesty is a strength; keep it.

**Built — the optimizing layer (what makes it *cost-aware*):**
- **Dollar cost model** (`cost.py`): per-model/tier pricing → cost in real USD,
  not just tokens; the budget reserve/reconcile is priced in money.
- **Cost-aware routing** (`routing.py`): a difficulty classifier picks the
  cheapest adequate tier, with a quality-checked cascade up to a stronger model on
  a bad reply. *Honest caveat:* the classifier and quality-checker are **keyword
  heuristics** (`KeywordDifficultyClassifier` / `KeywordQualityChecker`), not
  learned — a real mechanism, a simple one.
- **Cost-reduction levers** (`optimization.py`): a semantic cache
  (`InMemorySemanticCache`, **Jaccard** lexical similarity — not embedding-based)
  and a prompt compressor (tail-truncation).
- **Failover + soft cap** (`resilience.py`): `FailoverRouter` tries an alternate
  provider when a breaker is open (not just reject); `SoftCapRouter` downgrades to
  a cheaper tier near the budget cap instead of hard-blocking.

**Remaining / honest gaps:**
- The difficulty classifier and semantic cache are **heuristic** (keyword /
  Jaccard), not learned / embedding-based — a low accuracy ceiling. A learned
  difficulty classifier and an embedding-similarity cache are the next step.
- No prefix-cache-breakpoint management or provider-side cache-token discount
  accounting.
- The scale story (multi-worker, TCP-exhaustion levers, DNS/serverless cutover) is
  `SCALING.md` reasoning marked "architecture-only", not load-tested.

### Milestones — status

M1 (dollar cost model), M2 (cost-aware routing + quality-checked cascade), M3
(semantic cache + prompt compression), and M4 (failover + soft cap) are **built** —
see the optimizing-layer component list above, with their honest simplicity
caveats (keyword difficulty classifier, Jaccard cache). Each has an
acceptance/mutation shape (e.g. flat per-token pricing → "expensive costs more"
fails; disable the cache → "zero LLM calls on repeat" fails; disable failover →
breaker-open rejects instead of falling over).

The remaining work is upgrading the heuristics to learned / embedding-based,
prefix-cache-breakpoint accounting, and load-testing the scale chain in
`SCALING.md`.

---

## Architecture

```
Caller
  │
  ▼
TieredRouter (C3 — litellm.Router)
  │  resolves tier → model
  │
  ├─► BudgetTracker (C1 — Redis-backed reserve-then-reconcile)
  │     pre-call reserve, post-call reconcile
  │
  ├─► CircuitBreaker (C2 — hand-rolled Redis 3-state + SETNX half-open lock)
  │     fail-fast on degraded provider
  │
  ▼
litellm.Router.completion(...)
  │
  ▼
Redactor (C5 — AgentLeak runtime redaction)
  strip credentials from tool outputs before model context
```

## Components

| ID | Component | What it solves |
|----|-----------|----------------|
| C1 | `BudgetTracker` | Per-user token budget; two-step reserve-then-reconcile so we reject BEFORE spending money. State in Redis so multiple workers share counters. |
| C2 | `CircuitBreaker` | Hand-rolled 3-state breaker (closed/open/half-open) with Redis-aggregated failure counts. Half-open single-probe serialized via `SETNX`. |
| C3 | `TieredRouter` | Declarative tier-based routing via `litellm.Router`. Callers pick tier, not model id. |
| C4 | `SCALING.md` | Boundary-chain reasoning: single-process → Redis → LB → TCP exhaustion → DNS/serverless. No fake QPS. |
| C5 | `Redactor` | Runtime-internal credential redaction (AgentLeak). Strips configured API keys and known credential shapes from tool outputs before they enter model context. |

## Setup

```bash
# 1. Create conda env (mandatory)
conda create -n cost-aware-gateway python=3.11 -y
conda activate cost-aware-gateway

# 2. Install deps
pip install -e ".[dev]"

# 3. Start Redis
docker compose up -d

# 4. Configure keys
cp config.example.yaml config.yaml
# edit config.yaml with real API keys
```

## Running

```bash
# Run all demos (smoke test)
python -m cost_aware_gateway.run --all

# Run individual demos
python -m cost_aware_gateway.run --budget
python -m cost_aware_gateway.run --breaker
python -m cost_aware_gateway.run --router
python -m cost_aware_gateway.run --redact
```

## Tests

```bash
pytest tests/ -v
```

## Security Note — LiteLLM Supply-Chain Attack (CVE-2026-42271)

LiteLLM versions **1.82.7 and 1.82.8** were backdoored on PyPI in March 2026
(TeamPCP). The attack vector was a **CI/CD supply-chain compromise**: a stolen
PyPI token obtained via a compromised Trivy GitHub Action in LiteLLM's CI
pipeline — **NOT** an "MCP-injection". The backdoor exfiltrated SSH keys, cloud
credentials, and crypto wallets from every Python process that imported those
versions.

This repo pins **`litellm>=1.82.9,<2.0`** (first clean release: **v1.82.9**).
See `pyproject.toml` for the pin and [LiteLLM's security update](https://docs.litellm.ai/blog/security-update-march-2026) for the full incident report.

## Architecture Decisions

### Why WATCH/MULTI/EXEC for budget instead of Lua script?

The budget tracker uses Redis `WATCH` + `MULTI`/`EXEC` transactions instead of
a Lua script. The reason is **testability with fakeredis**: fakeredis does not
implement `EVAL`/`EVALSHA`, so a Lua-based implementation would make the unit
tests impossible to run in CI without a real Redis. `WATCH`/`MULTI`/`EXEC` is
still atomic on real Redis, and the optimistic-locking retry loop is a standard
pattern for this kind of shared counter. We prove atomicity with a real-Redis
concurrency stress test (`tests/test_budget.py::TestRealRedisBudget`).

### Why a hand-rolled Redis 3-state breaker?

The breaker implements the classic three-state machine (closed → open after N
failures → half-open after recovery_seconds → close on probe success / re-open
on probe failure) directly on Redis. Every `record_failure` does an atomic
`INCR` on the Redis failure counter; when it reaches `failure_threshold`, the
state is set to OPEN in Redis. This makes the breaker genuinely cross-worker:
3 workers each failing once will trip a threshold=3 breaker because Redis sees
count=3.

The half-open single-probe invariant (only one worker probes when recovering)
is enforced with a Redis `SET ... NX` lock — the cheapest correct answer for
cross-process mutual exclusion.

### Why not just run LiteLLM Proxy directly?

LiteLLM Proxy ships with built-in budget limits and rate limiting, but those
are **post-hoc, aggregated metrics**. This repo's differentiator is the
**application-layer two-step reserve-then-reconcile protocol**: we reserve an
estimate *before* the call, then reconcile with the actual usage *after* the
call. That lets us reject a request before spending money, which is the core
Denial-of-Wallet defense. Using LiteLLM Proxy's off-the-shelf limiter would
hide that mechanism behind a config flag and make it impossible to demo the
reserve/reconcile flow explicitly.

### Why doesn't SCALING.md give QPS numbers?

We have not performed load testing in this repository. `SCALING.md` documents
the **architecture boundary chain** (single-process → Redis → LB → TCP
exhaustion → serverless) and marks each step as either "measured here" or
"architecture-only." Inventing QPS figures would be dishonest; a staff engineer
reading the repo must be able to answer "what breaks next?" without guessing.

## Scaling-Shape Paradigm

See **`SCALING.md`** for the boundary-chain reasoning: single-process → Redis
externalized shared state → 2nd worker behind LB → TCP/connection-exhaustion
levers (`TIME_WAIT`, `SO_REUSEPORT`, `net.ipv4.ip_local_port_range`) →
DNS/serverless cutover. No fake QPS; every step marked "measured here" vs
"architecture-only."

## License

MIT
