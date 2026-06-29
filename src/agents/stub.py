"""
Stub agent — returned for all agents not yet fully implemented.

Returns a structured, non-error response identifying:
- the classified intent
- extracted entities
- which agent would handle this
- a clear "not implemented in this build" message

Never crashes.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

from src.core.classifier import ClassificationResult
from src.core.session import Session


async def run(
    query: str,
    classification: ClassificationResult,
    user_profile: dict,
    session: Session,
) -> AsyncIterator[str]:
    """Yields a structured stub response as SSE text."""

    agent_descriptions = {
        "market_research": "researches stocks, sectors, news, and market data",
        "investment_strategy": "provides allocation and investment strategy guidance",
        "financial_calculator": "runs compound interest, Sharpe ratio, and other financial calculations",
        "risk_assessment": "analyses portfolio volatility, drawdown, and exposure risk",
        "recommendations": "generates specific rebalance and investment recommendations",
        "predictive_analysis": "provides price targets, forecasts, and scenario analysis",
        "support": "handles account, KYC, deposits, withdrawals, and platform questions",
    }

    agent = classification.agent
    desc = agent_descriptions.get(agent, "handles this type of query")

    stub = {
        "status": "not_implemented",
        "classified_intent": classification.intent,
        "target_agent": agent,
        "agent_description": desc,
        "extracted_entities": classification.entities.model_dump(exclude_none=True),
        "message": (
            f"The **{agent.replace('_', ' ').title()}** agent ({desc}) "
            f"would handle this request in the full build. "
            f"This sprint covers portfolio_health end-to-end; "
            f"all other agents are queued for implementation."
        ),
        "confidence": classification.confidence,
    }

    # Friendly plain-text preamble
    preamble = (
        f"I understand you're asking about: *{classification.intent}*.\n\n"
        f"This would be handled by the **{agent.replace('_', ' ').title()}** agent, "
        f"which {desc}. "
        f"This agent is not yet available in the current build.\n\n"
    )
    if classification.entities.tickers:
        preamble += f"Tickers detected: {', '.join(classification.entities.tickers)}\n"
    if classification.entities.topics:
        preamble += f"Topics detected: {', '.join(classification.entities.topics)}\n"
    if classification.entities.amount is not None:
        preamble += f"Amount detected: ${classification.entities.amount:,.0f}\n"

    yield preamble
    yield f"\n```json\n{json.dumps(stub, indent=2)}\n```\n"
