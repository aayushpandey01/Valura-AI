"""
Classifier routing accuracy on the labeled gold set.
Success threshold: >= 85% routing accuracy (ASSIGNMENT.md).
Uses mock_llm (Gemini-compatible MagicMock) — no GEMINI_API_KEY needed.
"""
from typing import Any
import pytest
from src.core.classifier import classify_intent


def _normalize_ticker(t: str) -> str:
    return t.upper().split(".")[0]


def matches_entities(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    for field, exp_value in expected.items():
        act_value = actual.get(field)
        if act_value is None:
            return False
        if field == "tickers":
            exp_set = {_normalize_ticker(t) for t in exp_value}
            act_set = {_normalize_ticker(t) for t in act_value}
            if not exp_set.issubset(act_set):
                return False
        elif field in ("topics", "sectors"):
            exp_set = {s.lower() for s in exp_value}
            act_set = {s.lower() for s in act_value}
            if not exp_set.issubset(act_set):
                return False
        elif field in ("amount", "rate"):
            if abs(act_value - exp_value) > abs(exp_value) * 0.05:
                return False
        elif field == "period_years":
            if int(act_value) != int(exp_value):
                return False
        else:
            if str(act_value).lower() != str(exp_value).lower():
                return False
    return True


@pytest.mark.asyncio
async def test_classifier_routing_accuracy(gold_classifier_queries, mock_llm):
    """Threshold: >= 85% routing accuracy."""
    correct = 0
    failures = []
    for case in gold_classifier_queries:
        result = await classify_intent(case["query"], [], mock_llm)
        if result.agent == case["expected_agent"]:
            correct += 1
        else:
            failures.append({"id": case["id"], "expected": case["expected_agent"], "got": result.agent, "query": case["query"][:60]})

    accuracy = correct / len(gold_classifier_queries)
    print(f"\nRouting accuracy: {accuracy:.2%} ({correct}/{len(gold_classifier_queries)})")
    if failures:
        for f in failures[:10]:
            print(f"  {f['id']}: expected={f['expected']}, got={f['got']} | {f['query']}")

    assert accuracy >= 0.85, f"Routing accuracy {accuracy:.2%} below 85%"


@pytest.mark.asyncio
async def test_classifier_entity_extraction(gold_classifier_queries, mock_llm):
    """Soft signal — informational, no hard threshold."""
    matched = 0
    total_with_entities = 0
    for case in gold_classifier_queries:
        if not case["expected_entities"]:
            continue
        total_with_entities += 1
        result = await classify_intent(case["query"], [], mock_llm)
        if matches_entities(result.entities.model_dump(), case["expected_entities"]):
            matched += 1
    rate = matched / total_with_entities if total_with_entities else 0.0
    print(f"\nEntity match rate: {rate:.2%} ({matched}/{total_with_entities})")


@pytest.mark.asyncio
async def test_classifier_returns_valid_schema(mock_llm):
    from src.core.models import ClassifierOutput
    result = await classify_intent("How is my portfolio doing?", [], mock_llm)
    assert isinstance(result, ClassifierOutput)
    assert result.agent in {"portfolio_health", "market_research", "investment_strategy",
                            "financial_calculator", "predictive_analysis", "support"}


@pytest.mark.asyncio
async def test_classifier_fallback_on_bad_response():
    """Classifier returns fallback (not raises) when LLM returns garbage."""
    from unittest.mock import MagicMock
    bad_client = MagicMock()
    bad_resp = MagicMock()
    bad_resp.text = "this is not json!!!"
    bad_client.models.generate_content.return_value = bad_resp

    result = await classify_intent("hello", [], bad_client)
    assert result.agent == "support"
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_classifier_fallback_on_exception():
    """Classifier returns fallback (not raises) when LLM client raises."""
    from unittest.mock import MagicMock
    err_client = MagicMock()
    err_client.models.generate_content.side_effect = RuntimeError("Network error")

    result = await classify_intent("hello", [], err_client)
    assert result.confidence == 0.0
