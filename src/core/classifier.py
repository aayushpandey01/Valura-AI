"""
Intent Classifier — one Gemini call per query.
Returns structured output: intent, entities, target agent, safety verdict.
"""
from __future__ import annotations
import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from pydantic import ValidationError

from .models import ClassifierOutput, ExtractedEntities

logger = logging.getLogger(__name__)

KNOWN_AGENTS = {
    "portfolio_health",
    "market_research",
    "investment_strategy",
    "financial_calculator",
    "predictive_analysis",
    "support",
}

_CLASSIFIER_SYSTEM = """You are the intent classifier for Valura, a wealth management AI platform.

Your job: analyse the user's query (considering conversation history) and return a single JSON object.

## Output schema (return ONLY valid JSON, no markdown, no explanation):
{
  "intent": "<short intent label, e.g. portfolio_health_check>",
  "agent": "<one of: portfolio_health | market_research | investment_strategy | financial_calculator | predictive_analysis | support>",
  "entities": {
    "tickers": ["AAPL", "NVDA"],
    "sectors": ["technology"],
    "amount": 10000,
    "rate": 7.5,
    "period_years": 10,
    "topics": []
  },
  "safety_verdict": "pass",
  "confidence": 0.95,
  "reasoning": "one sentence"
}

## Agent routing rules:
- portfolio_health: "how is my portfolio?", health check, diversification, concentration, performance vs benchmark, "am I at risk?"
- market_research: specific stock/sector research, price, P/E, news, analyst ratings, comparisons
- investment_strategy: "should I buy/sell?", rebalancing advice, allocation strategy, entry/exit guidance
- financial_calculator: computations — returns, compound interest, position sizing, Sharpe ratio, beta, income projections
- predictive_analysis: future projections, forecasts, "what will X be worth in Y years", probability estimates
- support: platform help, account issues, concept explanations, onboarding

## Follow-up resolution:
- If the user says "what about X?" after asking about Y, X replaces Y as the primary entity
- If the user says "it" or "that", resolve the reference from the previous turn
- If the user says "double that amount", compute the new amount

Always resolve pronouns and references using the conversation history provided.
"""

_FALLBACK_OUTPUT = ClassifierOutput(
    intent="unknown",
    agent="support",
    entities=ExtractedEntities(),
    safety_verdict="pass",
    confidence=0.0,
    reasoning="Classifier failed; routed to support as fallback.",
)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _build_prompt(query: str, history: List[Dict[str, str]]) -> str:
    """Combine history + current query into a single prompt string."""
    parts = [_CLASSIFIER_SYSTEM, ""]
    if history:
        parts.append("## Conversation history (most recent last):")
        for turn in history[-6:]:
            role = turn["role"].capitalize()
            parts.append(f"{role}: {turn['content']}")
        parts.append("")
    parts.append(f"## Current query to classify:\n{query}")
    return "\n".join(parts)


async def classify_intent(
    query: str,
    conversation_history: List[Dict[str, str]],
    client: genai.Client,
    model: str = "gemini-2.0-flash",
) -> ClassifierOutput:
    """
    Single Gemini call to classify intent and extract entities.
    Never raises — returns fallback on any error.
    """
    prompt = _build_prompt(query, conversation_history)

    try:
        loop = asyncio.get_event_loop()

        def _call():
            return client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                    max_output_tokens=400,
                ),
            )

        response = await loop.run_in_executor(None, _call)
        raw = response.text or ""
        data = _extract_json(raw)

        if data is None:
            logger.warning("Classifier: could not extract JSON: %s", raw[:200])
            return _FALLBACK_OUTPUT

        agent = data.get("agent", "support")
        if agent not in KNOWN_AGENTS:
            logger.warning("Classifier: unknown agent '%s', defaulting to support", agent)
            data["agent"] = "support"

        # Normalise tickers to uppercase
        if "entities" in data and "tickers" in (data["entities"] or {}):
            data["entities"]["tickers"] = [
                t.upper() for t in (data["entities"]["tickers"] or [])
            ]

        return ClassifierOutput.model_validate(data)

    except ValidationError as e:
        logger.warning("Classifier: schema validation failed: %s", e)
        return _FALLBACK_OUTPUT
    except Exception as e:
        logger.error("Classifier: unexpected error: %s", e)
        return _FALLBACK_OUTPUT
