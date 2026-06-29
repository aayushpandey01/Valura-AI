"""
Portfolio Health agent tests — skeleton wired to implementation.
"""
import json
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from src.core.models import AgentRequest, ClassifierOutput, ExtractedEntities, UserContext
from src.agents.portfolio_health import run_portfolio_health


def _make_request(user_dict: dict, query: str = "How is my portfolio doing?") -> AgentRequest:
    return AgentRequest(
        query=query,
        user_context=UserContext(**user_dict),
        classifier_output=ClassifierOutput(
            intent="portfolio_health_check",
            agent="portfolio_health",
            entities=ExtractedEntities(),
            safety_verdict="pass",
            confidence=0.95,
        ),
        conversation_history=[],
    )


def _make_gemini_client(narrative: str = "Your portfolio looks healthy.") -> MagicMock:
    """Returns a mock google.genai.Client."""
    client = MagicMock()
    resp = MagicMock()
    resp.text = narrative
    client.models.generate_content.return_value = resp
    return client


async def _collect_structured(user_dict: dict) -> dict:
    req = _make_request(user_dict)
    client = _make_gemini_client()
    with patch("src.agents.portfolio_health._fetch_benchmark_return", AsyncMock(return_value=14.0)):
        async for chunk in run_portfolio_health(req, client):
            if chunk.startswith("data:") and "[DONE]" not in chunk:
                raw = chunk.replace("data:", "").strip()
                try:
                    parsed = json.loads(raw)
                    if "concentration_risk" in parsed:
                        return parsed
                except json.JSONDecodeError:
                    pass
    return {}


@pytest.mark.asyncio
async def test_portfolio_health_does_not_crash_on_empty_portfolio(load_user, mock_llm):
    """usr_004 has no positions. Agent must not crash and must include disclaimer."""
    user = load_user("usr_004")
    req = _make_request(user)
    client = _make_gemini_client()

    chunks = []
    with patch("src.agents.portfolio_health._fetch_benchmark_return", AsyncMock(return_value=14.0)):
        async for chunk in run_portfolio_health(req, client):
            chunks.append(chunk)

    full = "".join(chunks)
    assert "[DONE]" in full

    output = {}
    for chunk in chunks:
        if chunk.startswith("data:") and "[DONE]" not in chunk:
            raw = chunk.replace("data:", "").strip()
            try:
                parsed = json.loads(raw)
                if "disclaimer" in parsed:
                    output = parsed
            except json.JSONDecodeError:
                pass

    assert output, "No structured output found"
    assert "disclaimer" in output
    assert output["disclaimer"]


@pytest.mark.asyncio
async def test_portfolio_health_flags_concentration(load_user, mock_llm):
    """usr_003 has ~60%+ in NVDA. Agent must surface high concentration."""
    user = load_user("usr_003")
    output = await _collect_structured(user)
    assert output, "No structured output"
    assert output["concentration_risk"]["flag"] in {"high", "warning"}


@pytest.mark.asyncio
async def test_portfolio_health_includes_disclaimer(load_user, mock_llm):
    user = load_user("usr_001")
    output = await _collect_structured(user)
    assert output
    assert output.get("disclaimer")
    assert "not investment advice" in output["disclaimer"].lower()


@pytest.mark.asyncio
async def test_empty_portfolio_has_build_guidance(load_user, mock_llm):
    user = load_user("usr_004")
    req = _make_request(user)
    client = _make_gemini_client()

    data_chunks = []
    with patch("src.agents.portfolio_health._fetch_benchmark_return", AsyncMock(return_value=14.0)):
        async for chunk in run_portfolio_health(req, client):
            if chunk.startswith("data:") and "[DONE]" not in chunk:
                raw = chunk.replace("data:", "").strip()
                try:
                    parsed = json.loads(raw)
                    if "observations" in parsed:
                        data_chunks.append(parsed)
                except json.JSONDecodeError:
                    pass

    assert data_chunks
    obs_text = " ".join(o["text"] for o in data_chunks[0].get("observations", []))
    assert any(w in obs_text.lower() for w in ["empty", "starting", "invest", "portfolio", "build"])


@pytest.mark.asyncio
async def test_all_users_complete_without_crash(load_user, mock_llm):
    for uid in ["usr_001", "usr_002", "usr_003", "usr_004", "usr_005"]:
        user = load_user(uid)
        req = _make_request(user)
        client = _make_gemini_client()
        chunks = []
        with patch("src.agents.portfolio_health._fetch_benchmark_return", AsyncMock(return_value=14.0)):
            async for chunk in run_portfolio_health(req, client):
                chunks.append(chunk)
        assert "[DONE]" in "".join(chunks), f"Stream did not terminate for {uid}"
