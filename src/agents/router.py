"""Agent router — maps classifier output to the correct agent runner."""
from __future__ import annotations
from typing import AsyncIterator
from google import genai

from ..core.models import AgentRequest
from .portfolio_health import run_portfolio_health
from .stubs import run_stub_agent


async def route(
    request: AgentRequest,
    client: genai.Client,
    model: str = "gemini-2.0-flash",
) -> AsyncIterator[str]:
    if request.classifier_output.agent == "portfolio_health":
        async for chunk in run_portfolio_health(request, client, model):
            yield chunk
    else:
        async for chunk in run_stub_agent(request):
            yield chunk
