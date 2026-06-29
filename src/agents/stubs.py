"""
Stub agents for all non-implemented agents.
Returns structured "not implemented" response — does NOT crash.
"""
from __future__ import annotations
import json
from typing import AsyncIterator
from ..core.models import AgentRequest


_STUB_DESCRIPTIONS = {
    "market_research": "Real-time market data, stock analysis, sector trends, and news aggregation.",
    "investment_strategy": "Personalised investment recommendations, rebalancing, and allocation advice.",
    "financial_calculator": "Portfolio metrics, compound interest, position sizing, and risk calculations.",
    "predictive_analysis": "AI-powered projections, scenario modelling, and probability forecasting.",
    "support": "Platform help, onboarding guidance, and concept explanations.",
}


async def run_stub_agent(request: AgentRequest) -> AsyncIterator[str]:
    """Yields a structured stub response for any unimplemented agent."""
    agent = request.classifier_output.agent
    entities = request.classifier_output.entities

    response = {
        "type": "not_implemented",
        "agent": agent,
        "intent": request.classifier_output.intent,
        "entities": entities.model_dump(),
        "description": _STUB_DESCRIPTIONS.get(agent, "This agent handles specialised queries."),
        "message": (
            f"The '{agent}' agent is not yet implemented in this build. "
            f"It would handle: {_STUB_DESCRIPTIONS.get(agent, 'specialised queries')}. "
            f"Your query has been correctly classified and routed."
        ),
        "query_received": request.query,
    }
    yield f"data: {json.dumps(response)}\n\n"
    yield "data: [DONE]\n\n"
