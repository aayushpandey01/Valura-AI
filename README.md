# Valura AI — Microservice

> AI co-investor spine: safety guard → intent classifier → specialist agent router → SSE streaming.

**Video walkthrough:** [Link to be added after recording]

---

## Quick Start

```bash
# 1. Clone and enter
cd valura-ai

# 2. Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and set OPENAI_API_KEY

# 5. Run the server
uvicorn src.main:app --reload --port 8000

# 6. Run tests (no API key needed)
pytest tests/ -v
```

---

## Architecture

```
POST /query
    │
    ▼
[Safety Guard]  ← synchronous, no LLM, < 10ms
    │  blocked? → SSE blocked event → DONE
    │  pass ↓
    ▼
[Session Store] ← load conversation history (in-memory)
    │
    ▼
[Intent Classifier] ← single LLM call, structured JSON output
    │
    ▼
[Pipeline Metadata] → SSE metadata event (agent, intent, entities, safety_verdict)
    │
    ▼
[Agent Router]
    ├── portfolio_health  → FULLY IMPLEMENTED
    ├── market_research   → stub
    ├── investment_strategy → stub
    ├── financial_calculator → stub
    ├── predictive_analysis → stub
    └── support           → stub
    │
    ▼
[SSE Stream] → client
```

**Request flow for a portfolio health check:**
1. `POST /query` with `{"query": "How is my portfolio doing?", "user_context": {...}}`
2. Safety guard runs in ~0.1ms — passes
3. Classifier LLM call returns `{"agent": "portfolio_health", "intent": "portfolio_health_check", ...}`
4. Metadata event streamed to client
5. Portfolio Health agent computes concentration, performance, fetches benchmark, generates narrative
6. Structured JSON streamed as SSE data events
7. `[DONE]` sentinel closes the stream

---

## API

### `POST /query`

**Request:**
```json
{
  "query": "How is my portfolio doing?",
  "user_context": {
    "user_id": "user_001",
    "name": "Arjun Mehta",
    "kyc_status": "verified",
    "risk_profile": "aggressive",
    "investment_horizon_years": 5,
    "currency": "USD",
    "portfolio": {
      "total_value_usd": 250000,
      "cash_pct": 5.0,
      "holdings": [
        {"ticker": "NVDA", "shares": 200, "avg_cost": 450.00, "current_price": 875.00, "sector": "Technology"}
      ],
      "purchase_date": "2022-01-15"
    },
    "benchmark": "S&P 500"
  },
  "conversation_id": "optional-uuid"
}
```

**Response:** `text/event-stream`

```
data: {"type": "status", "message": "Analysing your portfolio..."}

data: {"type": "metadata", "agent": "portfolio_health", "intent": "portfolio_health_check", ...}

data: {"concentration_risk": {...}, "performance": {...}, "benchmark_comparison": {...}, "observations": [...], "summary": "...", "disclaimer": "..."}

data: [DONE]
```

### `GET /health`

Returns service status, model, active session count.

---

## Environment Variables

| Variable | Required | Default | Notes |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | Your OpenAI API key |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Use `gpt-4.1` for evaluation |
| `PIPELINE_TIMEOUT_SECONDS` | No | `30` | Total pipeline timeout |

---

## Library Choices

| Library | Why |
|---|---|
| **FastAPI** | Best-in-class async Python web framework; native Pydantic integration; OpenAPI docs free |
| **OpenAI SDK** | Official async client; handles retries, streaming, structured outputs |
| **Pydantic v2** | Schema validation across the entire pipeline; fast, Rust-backed |
| **yfinance** | Free, no-key-needed market data for benchmark comparison |
| **python-dotenv** | Zero-friction env var management |
| **pytest + pytest-asyncio** | Standard async testing; `asyncio_mode = auto` removes boilerplate |

**Not used:** sse-starlette (used native FastAPI `StreamingResponse` instead — simpler, no extra dep for this use case).

---

## Design Decisions

### 1. Safety guard: pure regex, no LLM

The guard runs before the classifier, so it must be synchronous and sub-10ms. Regex over a small set of patterns (< 50) runs in < 1ms on any modern CPU. The tradeoff: the guard may over-block edge cases (e.g. very unusual phrasing of educational queries). This is documented and intentional — it is better to occasionally ask a user to rephrase than to let a manipulation request through.

Educational signal detection mitigates false positives: if the query contains phrases like "what is", "explain", "why is it illegal", "historical examples", the guard reduces sensitivity for most categories.

### 2. Classifier: one LLM call with `response_format: json_object`

A single structured LLM call handles intent + entity extraction + safety verdict. This keeps latency low (one round trip) and cost predictable. The classifier never crashes the request — any failure returns a typed fallback that routes to `support`.

Follow-up resolution is handled by including the last 6 messages of conversation history in the classifier prompt. The classifier's system prompt instructs it to resolve pronouns ("it", "that") and replace the previous topic when the user says "what about X?".

### 3. Session memory: in-memory

**Why in-memory:** Zero latency, no infrastructure dependency, correct for a demo/single-instance deployment. The `SessionStore` interface is thin — swapping to Redis or Postgres requires only replacing the `get_store()` implementation, not touching the pipeline.

**Tradeoffs:** Sessions are lost on restart; cannot scale horizontally without sticky sessions or an external store. For production, Redis with a 1-hour TTL is the obvious next step.

### 4. Pipeline timeout: 30 seconds total

The classifier gets 8 seconds; the agent gets the remainder (~22 seconds). 30 seconds is a generous upper bound — the p95 target is 6 seconds end-to-end. The timeout exists to prevent runaway requests from holding connections open. All timeouts yield structured SSE error events, never stack traces.

### 5. Stub contract

All unimplemented agents return a typed stub response with: the classified intent, extracted entities, the agent that would handle the request, and a clear message. The router never crashes regardless of agent. Adding a new agent is a one-line addition to `src/agents/router.py`.

### 6. Empty portfolio (user_004)

The Portfolio Health agent detects an empty portfolio before running any calculations and returns a BUILD-oriented response with three practical observations for a new investor. This is done with a dedicated `_build_empty_portfolio_response()` function — no special-casing spread across the main agent logic.

---

## Performance Targets

| Target | Value | How Measured |
|---|---|---|
| Model (dev) | `gpt-4o-mini` | `OPENAI_MODEL` env var |
| Model (eval) | `gpt-4.1` | Set `OPENAI_MODEL=gpt-4.1` |
| p95 first-token latency | < 2s | Safety + classifier latency; metadata event is first token |
| p95 end-to-end | < 6s | Benchmark fetch (yfinance) is the main variable; cached in-process |
| Cost per query at gpt-4.1 | < $0.05 | Classifier: ~400 tokens in + ~100 out; Narrative: ~600 in + ~200 out. At gpt-4.1 pricing ($2/M in, $8/M out): ~$0.003 classifier + ~$0.003 narrative = **~$0.006 per query** |

**Measurement method:** Run `pytest tests/ -v -s` with a real API key and observe logged latencies. For load testing, use `locust` or `ab` against the `/query` endpoint with pre-built user context payloads.

---

## Testing

```bash
# All tests — no API key needed
pytest tests/ -v

# Specific suites
pytest tests/test_safety.py -v         # Safety guard
pytest tests/test_classifier.py -v     # Classifier (mocked LLM)
pytest tests/test_portfolio_health.py -v  # Portfolio health agent
pytest tests/test_pipeline.py -v       # Full pipeline integration
pytest tests/test_conversations.py -v  # Conversation follow-up
```

### Test coverage

| Suite | What it tests | Key thresholds |
|---|---|---|
| `test_safety.py` | Safety guard recall, educational pass-through, latency, category distinctness | Recall ≥ 95%, pass-through ≥ 90%, latency < 10ms |
| `test_classifier.py` | Routing accuracy, entity normalisation, fallback on error | Routing ≥ 85% |
| `test_portfolio_health.py` | Concentration, performance, empty portfolio, streaming, disclaimer | user_004 must not crash |
| `test_pipeline.py` | Full SSE pipeline, session memory, conversation_id | All 5 users complete |
| `test_conversations.py` | Follow-up resolution, topic switching, history passing | All 3 conversations complete |

---

## What I'd Do With Another Week

1. **Redis session store** — replace in-memory with Redis for horizontal scaling and persistence across restarts
2. **Embedding-based pre-classifier** — embed the query and compare against a labelled centroid set; skip the LLM call when cosine similarity to a centroid exceeds 0.92. Estimated 40-60% LLM call reduction for common query types
3. **Implement market_research agent** — yfinance + OpenAI to produce real stock analysis, which would make the platform actually useful for the MONITOR use case
4. **Per-tenant model selection** — read `risk_profile` or a `tier` field from user context; premium users get `gpt-4.1`, free tier gets `gpt-4o-mini`. Already wired into the router via the `model` parameter
5. **Structured logging + observability** — emit structured JSON logs with query latency breakdowns per pipeline stage; hook into Prometheus or Datadog

---

## Project Structure

```
valura-ai/
├── README.md
├── requirements.txt
├── pytest.ini
├── .env.example
├── fixtures/
│   ├── README.md
│   ├── user_profiles/        (user_001 – user_005)
│   ├── conversations/        (conv_001 – conv_003)
│   └── test_queries/
│       ├── intent_classification.json   (60 labeled queries)
│       └── safety_pairs.json            (45 safety queries)
├── src/
│   ├── main.py               (FastAPI app + /query endpoint)
│   ├── core/
│   │   ├── models.py         (all Pydantic schemas)
│   │   ├── safety.py         (synchronous safety guard)
│   │   ├── classifier.py     (intent classifier, one LLM call)
│   │   └── session.py        (in-memory session store)
│   └── agents/
│       ├── router.py         (agent dispatcher)
│       ├── portfolio_health.py  (fully implemented)
│       └── stubs.py          (structured stubs for all other agents)
└── tests/
    ├── conftest.py           (mock LLM fixtures)
    ├── test_safety.py
    ├── test_classifier.py
    ├── test_portfolio_health.py
    ├── test_pipeline.py
    └── test_conversations.py
```
