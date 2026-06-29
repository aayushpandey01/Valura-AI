"""
Market data fetcher — wraps yfinance.

All live price/fundamentals fetching goes through this module.
Never hardcode prices; always fetch.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from functools import lru_cache

import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class QuoteData:
    ticker: str
    current_price: float
    currency: str
    name: str
    sector: str
    market_cap: float | None
    dividend_yield: float | None
    pe_ratio: float | None
    beta: float | None


@dataclass
class BenchmarkData:
    ticker: str
    name: str
    period_return_pct: float  # YTD or 1Y return


async def fetch_quote(ticker: str) -> QuoteData | None:
    """Fetch current quote data for a single ticker. Non-blocking via thread executor."""
    try:
        data = await asyncio.get_event_loop().run_in_executor(None, _fetch_quote_sync, ticker)
        return data
    except Exception as exc:
        logger.warning("Failed to fetch quote for %s: %s", ticker, exc)
        return None


def _fetch_quote_sync(ticker: str) -> QuoteData | None:
    try:
        t = yf.Ticker(ticker)
        info = t.info
        price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
            or 0.0
        )
        return QuoteData(
            ticker=ticker,
            current_price=float(price),
            currency=info.get("currency", "USD"),
            name=info.get("longName") or info.get("shortName") or ticker,
            sector=info.get("sector") or "Unknown",
            market_cap=info.get("marketCap"),
            dividend_yield=info.get("dividendYield"),
            pe_ratio=info.get("trailingPE"),
            beta=info.get("beta"),
        )
    except Exception as exc:
        logger.warning("Sync fetch failed for %s: %s", ticker, exc)
        return None


async def fetch_quotes_bulk(tickers: list[str]) -> dict[str, QuoteData]:
    """Fetch multiple quotes concurrently."""
    tasks = [fetch_quote(t) for t in tickers]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out: dict[str, QuoteData] = {}
    for ticker, result in zip(tickers, results):
        if isinstance(result, QuoteData):
            out[ticker] = result
    return out


async def fetch_benchmark_return(benchmark_ticker: str = "SPY", period: str = "1y") -> BenchmarkData:
    """Fetch benchmark return for the given period."""
    names = {"SPY": "S&P 500", "QQQ": "NASDAQ-100", "EFA": "MSCI EAFE", "EEM": "MSCI EM", "AGG": "US Bonds"}
    try:
        data = await asyncio.get_event_loop().run_in_executor(
            None, _fetch_benchmark_sync, benchmark_ticker, period
        )
        return data
    except Exception as exc:
        logger.warning("Benchmark fetch failed: %s", exc)
        return BenchmarkData(
            ticker=benchmark_ticker,
            name=names.get(benchmark_ticker, benchmark_ticker),
            period_return_pct=0.0,
        )


def _fetch_benchmark_sync(ticker: str, period: str) -> BenchmarkData:
    names = {"SPY": "S&P 500", "QQQ": "NASDAQ-100", "EFA": "MSCI EAFE", "EEM": "MSCI EM", "AGG": "US Bonds"}
    t = yf.Ticker(ticker)
    hist = t.history(period=period)
    if hist.empty:
        return BenchmarkData(ticker=ticker, name=names.get(ticker, ticker), period_return_pct=0.0)
    first = hist["Close"].iloc[0]
    last = hist["Close"].iloc[-1]
    ret = ((last - first) / first) * 100
    return BenchmarkData(ticker=ticker, name=names.get(ticker, ticker), period_return_pct=round(ret, 2))


def pick_benchmark(risk_profile: str, base_currency: str) -> str:
    """Choose a contextually relevant benchmark based on user profile."""
    if base_currency not in ("USD", "AED", "GBP", "EUR", "CHF"):
        return "EFA"  # international proxy
    if risk_profile == "conservative":
        return "AGG"
    if risk_profile == "aggressive":
        return "QQQ"
    return "SPY"  # moderate default
