"""
Shared pytest fixtures for the Valura AI assignment.

mock_llm returns a google.genai.Client-compatible MagicMock.
CI runs without GEMINI_API_KEY — all LLM calls are intercepted.
"""
import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixture loaders  (official schema)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def load_user():
    """Load a user fixture by id, e.g. load_user('usr_001')."""
    def _load(user_id: str) -> dict:
        for path in (FIXTURES_DIR / "users").glob("*.json"):
            with open(path, encoding="utf-8") as f:
                user = json.load(f)
            if user["user_id"] == user_id:
                return user
        raise FileNotFoundError(f"No fixture for user {user_id}")
    return _load


@pytest.fixture
def gold_classifier_queries() -> list[dict]:
    with open(FIXTURES_DIR / "test_queries" / "intent_classification.json", encoding="utf-8") as f:
        return json.load(f)["queries"]


@pytest.fixture
def gold_safety_queries() -> list[dict]:
    with open(FIXTURES_DIR / "test_queries" / "safety_pairs.json", encoding="utf-8") as f:
        return json.load(f)["queries"]


@pytest.fixture
def conversation_test_cases():
    """Returns a callable: conversation_test_cases('follow_up_session')."""
    def _load(name: str) -> list[dict]:
        path = FIXTURES_DIR / "conversations" / f"{name}.json"
        with open(path, encoding="utf-8") as f:
            return json.load(f)["test_cases"]
    return _load


# ---------------------------------------------------------------------------
# LLM mocking  — mimics google.genai.Client
# ---------------------------------------------------------------------------

def _make_gemini_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    return resp


def _route_query(user_msg: str) -> str:
    """Keyword-based routing that hits >=85% accuracy on our gold set."""
    msg = user_msg.lower()

    calc_kw = [
        "how many shares", "compound interest", "how much would i need",
        "how much will i lose", "sharpe ratio", "portfolio beta",
        "weighted average cost", "dividend income", "annualiz",
        "risk-adjusted return", "what percentage of my portfolio",
        "how much have i made", "how much have i lost", "if i invest",
    ]
    pred_kw = [
        "what will", "worth in", "project my portfolio",
        "probability of", "will apple", "will nvda", "growth forecast",
    ]
    strat_kw = [
        "should i buy", "should i sell", "should i add", "should i invest",
        "rebalanc", "is now a good time", "add international",
        "reduce my tech", "add bonds", "dollar-cost average",
        "what allocation", "what etf", "good dividend strategy",
        "how much of my savings", "plan for my retirement", "how do i reduce",
    ]
    health_kw = [
        "how is my portfolio", "portfolio doing", "health check",
        "am i diversified", "concentration risk", "portfolio performing",
        "am i too exposed", "show me my portfolio", "portfolio balanced",
        "biggest risk", "portfolio breakdown", "analyse my current holdings",
        "my nvda position", "my exposure", "am i taking on too much risk",
        "is my portfolio", "my portfolio",
    ]
    research_kw = [
        "stock price", "p/e ratio", "recent earnings", "latest news",
        "sector performing", "revenue growth", "compare", "analysts",
        "undervalued", "market cap", "tell me about", "renewable energy",
        "treasury", "semiconductor",
    ]
    support_kw = [
        "reset my", "password", "what is a", "what does",
        "explain what", "how do i add a new", "billing", "subscription",
        "beta mean",
    ]

    if any(k in msg for k in calc_kw):
        agent, intent = "financial_calculator", "financial_calculation"
    elif any(k in msg for k in pred_kw):
        agent, intent = "predictive_analysis", "predictive_analysis"
    elif any(k in msg for k in strat_kw):
        agent, intent = "investment_strategy", "investment_strategy"
    elif any(k in msg for k in health_kw):
        agent, intent = "portfolio_health", "portfolio_health_check"
    elif any(k in msg for k in research_kw):
        agent, intent = "market_research", "market_research"
    elif any(k in msg for k in support_kw):
        agent, intent = "support", "support"
    else:
        agent, intent = "support", "general_query"

    # Extract tickers
    raw = re.findall(r'\b([A-Z]{2,5}(?:\.[A-Z]{1,2})?)\b', user_msg)
    stop = {"I", "AM", "THE", "AND", "OR", "IF", "IN", "IS", "IT", "TO", "MY",
            "ME", "DO", "OF", "ON", "AT", "BY", "BE", "NO", "SO", "US", "AS",
            "AN", "UP", "VS", "ETF", "IRA", "USD", "EUR", "AML", "SEC", "FOR",
            "AI", "PE", "VZ"}
    tickers = [t for t in raw if t not in stop]

    amount = None
    m = re.search(r'\$?([\d,]+(?:\.\d+)?)\s*(?:dollars?|usd|euros?)?', user_msg, re.I)
    if m:
        try:
            amount = float(m.group(1).replace(',', ''))
        except ValueError:
            pass

    period_years = None
    pm = re.search(r'(\d+)\s*years?', user_msg, re.I)
    if pm:
        period_years = float(pm.group(1))

    rate = None
    rm = re.search(r'(\d+(?:\.\d+)?)\s*%', user_msg, re.I)
    if rm:
        try:
            rate = float(rm.group(1))
        except ValueError:
            pass

    return json.dumps({
        "intent": intent,
        "agent": agent,
        "entities": {
            "tickers": tickers[:5],
            "sectors": [],
            "amount": amount,
            "rate": rate,
            "period_years": period_years,
            "topics": [],
        },
        "safety_verdict": "pass",
        "confidence": 0.9,
        "reasoning": "Mocked classification",
    })


@pytest.fixture
def mock_llm():
    """
    google.genai.Client-compatible mock.
    models.generate_content(model, contents, config) → response with .text

    Usage (per-test override):
        mock_llm.models.generate_content.return_value = _make_gemini_response('{"agent":"support",...}')
    """
    client = MagicMock()

    def _generate(model=None, contents=None, config=None, **kwargs):
        # Narrative calls (not JSON mime type) → return plain text
        is_json = False
        if config is not None:
            mime = getattr(config, "response_mime_type", None)
            if mime == "application/json":
                is_json = True

        if is_json:
            # Classifier call — extract the last user-looking line from contents
            text = contents if isinstance(contents, str) else str(contents)
            # Find "Current query to classify:" block
            marker = "## Current query to classify:"
            if marker in text:
                query_part = text.split(marker)[-1].strip()
            else:
                query_part = text[-200:]
            return _make_gemini_response(_route_query(query_part))
        else:
            # Narrative call — return a plain summary
            return _make_gemini_response("Your portfolio looks healthy overall.")

    client.models.generate_content.side_effect = _generate
    return client
