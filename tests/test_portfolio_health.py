"""
Unit tests for portfolio health computation logic (no LLM needed).
Complements the skeleton integration tests.
"""
import pytest
from src.core.models import Holding
from src.agents.portfolio_health import (
    _compute_concentration,
    _compute_performance,
    _build_empty_portfolio_response,
)


class TestConcentrationRisk:
    def test_high_concentration_single_stock(self):
        holdings = [
            Holding(ticker="NVDA", shares=100, avg_cost=100, current_price=875, sector="Tech"),
            Holding(ticker="AAPL", shares=10, avg_cost=100, current_price=175, sector="Tech"),
        ]
        total = sum(h.shares * h.current_price for h in holdings)
        result = _compute_concentration(holdings, total)
        assert result.flag == "high"
        assert result.top_position_pct > 80

    def test_diversified_low_concentration(self):
        holdings = [
            Holding(ticker=f"T{i:02d}", shares=100, avg_cost=100, current_price=100, sector="Tech")
            for i in range(10)
        ]
        total = sum(h.shares * h.current_price for h in holdings)
        result = _compute_concentration(holdings, total)
        assert result.flag == "low"

    def test_empty_no_crash(self):
        result = _compute_concentration([], 0)
        assert result.flag == "low"
        assert result.top_position_pct == 0.0


class TestPerformance:
    def test_positive_return_calculated_correctly(self):
        holdings = [
            Holding(ticker="NVDA", shares=100, avg_cost=220, current_price=875, sector="Tech"),
        ]
        perf, _ = _compute_performance(holdings, "2022-01-01")
        assert perf.total_return_pct > 0
        assert perf.annualized_return_pct is not None

    def test_negative_return(self):
        holdings = [
            Holding(ticker="BABA", shares=100, avg_cost=200, current_price=72, sector="Tech"),
        ]
        perf, _ = _compute_performance(holdings, "2022-01-01")
        assert perf.total_return_pct < 0

    def test_empty_no_crash(self):
        perf, val = _compute_performance([], None)
        assert perf.total_return_pct == 0.0
        assert val == 0.0


class TestEmptyPortfolioResponse:
    def test_not_none(self):
        assert _build_empty_portfolio_response() is not None

    def test_has_observations(self):
        result = _build_empty_portfolio_response()
        assert len(result.observations) > 0

    def test_has_disclaimer(self):
        result = _build_empty_portfolio_response()
        assert len(result.disclaimer) > 20

    def test_zero_concentration(self):
        result = _build_empty_portfolio_response()
        assert result.concentration_risk.top_position_pct == 0.0
