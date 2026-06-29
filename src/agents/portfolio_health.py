"""
Portfolio Health Check Agent — fully implemented.
Covers MONITOR and PROTECT pillars of the Valura mission.
Uses Gemini for narrative generation.
"""
from __future__ import annotations
import asyncio
import logging
import math
from datetime import datetime, date
from typing import AsyncIterator, Dict, List, Optional, Tuple

import yfinance as yf
from google import genai
from google.genai import types

from ..core.models import (
    AgentRequest,
    BenchmarkComparison,
    ConcentrationRisk,
    Holding,
    Observation,
    Performance,
    PortfolioHealthOutput,
    UserContext,
)

logger = logging.getLogger(__name__)

_BENCHMARK_TICKERS: Dict[str, str] = {
    "S&P 500": "SPY",
    "MSCI World": "URTH",
    "FTSE 100": "ISF.L",
    "Nikkei 225": "^N225",
    "DAX": "^GDAXI",
}
_DEFAULT_BENCHMARK_RETURN = 14.0


def _annualised_return(total_pct: float, years: float) -> Optional[float]:
    if years <= 0:
        return None
    try:
        return ((1 + total_pct / 100) ** (1 / years) - 1) * 100
    except (ValueError, ZeroDivisionError):
        return None


def _years_since(purchase_date: Optional[str]) -> float:
    if not purchase_date:
        return 1.0
    try:
        start = datetime.strptime(purchase_date, "%Y-%m-%d").date()
        return max((date.today() - start).days / 365.25, 0.01)
    except ValueError:
        return 1.0


def _compute_concentration(holdings: List[Holding], total_invested: float) -> ConcentrationRisk:
    if not holdings or total_invested <= 0:
        return ConcentrationRisk(top_position_pct=0, top_3_positions_pct=0, flag="low")
    values = sorted([h.shares * h.current_price for h in holdings], reverse=True)
    top1 = (values[0] / total_invested) * 100
    top3 = (sum(values[:3]) / total_invested) * 100
    flag = "high" if top1 >= 50 or top3 >= 80 else "medium" if top1 >= 30 or top3 >= 60 else "low"
    return ConcentrationRisk(
        top_position_pct=round(top1, 1),
        top_3_positions_pct=round(top3, 1),
        flag=flag,
    )


def _compute_performance(
    holdings: List[Holding], purchase_date: Optional[str]
) -> Tuple[Performance, float]:
    if not holdings:
        return Performance(total_return_pct=0.0, annualized_return_pct=None), 0.0
    cost_basis = sum(h.shares * h.avg_cost for h in holdings)
    current_value = sum(h.shares * h.current_price for h in holdings)
    if cost_basis <= 0:
        return Performance(total_return_pct=0.0, annualized_return_pct=None), current_value
    total_pct = ((current_value - cost_basis) / cost_basis) * 100
    years = _years_since(purchase_date)
    ann = _annualised_return(total_pct, years)
    return (
        Performance(
            total_return_pct=round(total_pct, 1),
            annualized_return_pct=round(ann, 1) if ann is not None else None,
        ),
        current_value,
    )


async def _fetch_benchmark_return(benchmark: str, purchase_date: Optional[str]) -> float:
    ticker_sym = _BENCHMARK_TICKERS.get(benchmark, "SPY")
    years = _years_since(purchase_date)
    try:
        loop = asyncio.get_event_loop()
        ticker = await loop.run_in_executor(None, lambda: yf.Ticker(ticker_sym))
        hist = await loop.run_in_executor(
            None,
            lambda: ticker.history(period=f"{max(1, int(math.ceil(years)))}y"),
        )
        if hist.empty or len(hist) < 2:
            return _DEFAULT_BENCHMARK_RETURN
        total = ((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]) * 100
        return round(float(total), 1)
    except Exception as e:
        logger.warning("Benchmark fetch failed for %s: %s", benchmark, e)
        return _DEFAULT_BENCHMARK_RETURN


def _build_observations(
    holdings: List[Holding],
    concentration: ConcentrationRisk,
    performance: Performance,
    benchmark_comparison: Optional[BenchmarkComparison],
    user_context: UserContext,
) -> List[Observation]:
    obs: List[Observation] = []

    if concentration.flag == "high":
        top = max(holdings, key=lambda h: h.shares * h.current_price)
        obs.append(Observation(
            severity="warning",
            text=f"{concentration.top_position_pct:.0f}% of your portfolio is in {top.ticker} — "
                 f"highly concentrated. A large move in {top.ticker} will significantly impact your total portfolio.",
        ))
    elif concentration.flag == "medium":
        obs.append(Observation(
            severity="info",
            text=f"Your top position is {concentration.top_position_pct:.0f}% of your portfolio. "
                 "Consider whether this aligns with your risk tolerance.",
        ))

    # Sector concentration
    total_val = sum(h.shares * h.current_price for h in holdings)
    sector_values: Dict[str, float] = {}
    for h in holdings:
        sector_values[h.sector] = sector_values.get(h.sector, 0) + h.shares * h.current_price
    for sector, val in sorted(sector_values.items(), key=lambda x: -x[1]):
        if total_val > 0 and (val / total_val) * 100 > 60:
            obs.append(Observation(
                severity="warning",
                text=f"{(val/total_val)*100:.0f}% of your portfolio is in {sector}. "
                     "High sector concentration amplifies risk.",
            ))

    # Performance
    if performance.total_return_pct > 0:
        obs.append(Observation(
            severity="info",
            text=f"Portfolio is up {performance.total_return_pct:.1f}% overall"
                 + (f" ({performance.annualized_return_pct:.1f}% annualised)." if performance.annualized_return_pct else "."),
        ))
    else:
        obs.append(Observation(
            severity="warning",
            text=f"Portfolio is down {abs(performance.total_return_pct):.1f}% overall.",
        ))

    if benchmark_comparison:
        alpha = benchmark_comparison.alpha_pct
        if alpha > 0:
            obs.append(Observation(
                severity="info",
                text=f"Outperforming {benchmark_comparison.benchmark} by {alpha:.1f} percentage points.",
            ))
        elif alpha < -5:
            obs.append(Observation(
                severity="warning",
                text=f"Underperforming {benchmark_comparison.benchmark} by {abs(alpha):.1f} points. "
                     "Consider whether active stock-picking is adding value vs a low-cost index fund.",
            ))

    if user_context.risk_profile == "conservative" and concentration.flag == "high":
        obs.append(Observation(
            severity="warning",
            text="Your conservative risk profile conflicts with your high portfolio concentration.",
        ))

    for h in holdings:
        ret = (h.current_price - h.avg_cost) / h.avg_cost * 100
        if ret < -25:
            obs.append(Observation(
                severity="warning",
                text=f"{h.ticker} is down {abs(ret):.0f}% from your average cost. "
                     "Review whether your original thesis still holds.",
            ))

    return obs[:6]


def _build_empty_portfolio_response() -> PortfolioHealthOutput:
    return PortfolioHealthOutput(
        concentration_risk=ConcentrationRisk(top_position_pct=0.0, top_3_positions_pct=0.0, flag="low"),
        performance=Performance(total_return_pct=0.0, annualized_return_pct=None),
        benchmark_comparison=None,
        observations=[
            Observation(severity="info", text="Your portfolio is currently empty — you're starting from zero, which is a great position to be deliberate."),
            Observation(severity="info", text="Before investing: (1) ensure you have 3–6 months of emergency savings, (2) clarify your time horizon and risk tolerance."),
            Observation(severity="info", text="A common starting point: a low-cost S&P 500 index fund (e.g. VOO, VTI) gives instant diversification across hundreds of companies."),
        ],
        summary=(
            "You have no holdings yet. Start by defining your goals (retirement, growth, income), "
            "choose an asset allocation that matches your risk profile, and consider low-cost index "
            "funds as a foundation. Diversification from day one is one of the most powerful tools available."
        ),
    )


_NARRATIVE_SYSTEM = """You are a friendly wealth management assistant at Valura.
Given a structured portfolio health analysis as JSON, write a clear plain-language narrative summary 
(3-5 sentences) for a novice investor.
Rules: no unexplained jargon, surface the 1-2 most important things, use actual numbers, 
be empathetic not alarming, end with one concrete next step.
Do NOT include a disclaimer — that is added separately."""


async def _generate_narrative(output: PortfolioHealthOutput, client: genai.Client, model: str) -> str:
    try:
        loop = asyncio.get_event_loop()
        prompt = f"{_NARRATIVE_SYSTEM}\n\nPortfolio data:\n{output.model_dump_json(indent=2)}\n\nWrite the narrative."

        def _call():
            return client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=300,
                ),
            )

        response = await loop.run_in_executor(None, _call)
        return response.text or output.summary
    except Exception as e:
        logger.warning("Narrative generation failed: %s", e)
        return output.summary


async def run_portfolio_health(
    request: AgentRequest,
    client: genai.Client,
    model: str = "gemini-2.0-flash",
) -> AsyncIterator[str]:
    """Main entry point. Yields SSE data chunks."""
    import json as _json
    user = request.user_context
    holdings = user.portfolio.holdings

    yield f"data: {_json.dumps({'type': 'status', 'message': 'Analysing your portfolio...'})}\n\n"

    if not holdings or user.portfolio.total_value_usd == 0:
        result = _build_empty_portfolio_response()
        yield f"data: {result.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        return

    current_value = sum(h.shares * h.current_price for h in holdings)
    concentration = _compute_concentration(holdings, current_value)
    performance, _ = _compute_performance(holdings, user.portfolio.purchase_date)

    yield f"data: {_json.dumps({'type': 'status', 'message': 'Fetching benchmark data...'})}\n\n"
    benchmark_return = await _fetch_benchmark_return(user.benchmark, user.portfolio.purchase_date)

    benchmark_comparison = BenchmarkComparison(
        benchmark=user.benchmark,
        portfolio_return_pct=performance.total_return_pct,
        benchmark_return_pct=benchmark_return,
        alpha_pct=round(performance.total_return_pct - benchmark_return, 1),
    )

    observations = _build_observations(holdings, concentration, performance, benchmark_comparison, user)

    result = PortfolioHealthOutput(
        concentration_risk=concentration,
        performance=performance,
        benchmark_comparison=benchmark_comparison,
        observations=observations,
        summary="",
    )

    yield f"data: {_json.dumps({'type': 'status', 'message': 'Generating insights...'})}\n\n"
    narrative = await _generate_narrative(result, client, model)
    result.summary = narrative

    yield f"data: {result.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"
