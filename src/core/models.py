"""
Shared Pydantic models used across the pipeline.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── User / Portfolio ────────────────────────────────────────────────────────

class Holding(BaseModel):
    ticker: str
    shares: float
    avg_cost: float
    current_price: float
    sector: str = "Unknown"


class Portfolio(BaseModel):
    total_value_usd: float = 0.0
    cash_pct: float = 100.0
    holdings: List[Holding] = Field(default_factory=list)
    purchase_date: Optional[str] = None


class UserContext(BaseModel):
    user_id: str
    name: str = "User"
    kyc_status: str = "unknown"
    risk_profile: str = "moderate"
    investment_horizon_years: int = 10
    currency: str = "USD"
    portfolio: Portfolio = Field(default_factory=Portfolio)
    benchmark: str = "S&P 500"


# ── Classifier output ────────────────────────────────────────────────────────

class ExtractedEntities(BaseModel):
    tickers: List[str] = Field(default_factory=list)
    sectors: List[str] = Field(default_factory=list)
    amount: Optional[float] = None
    rate: Optional[float] = None
    period_years: Optional[float] = None
    topics: List[str] = Field(default_factory=list)


class ClassifierOutput(BaseModel):
    intent: str
    agent: str
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)
    safety_verdict: str = "pass"  # informational only
    confidence: float = 1.0
    reasoning: str = ""


# ── Agent request / response ─────────────────────────────────────────────────

class AgentRequest(BaseModel):
    query: str
    user_context: UserContext
    classifier_output: ClassifierOutput
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)


class ConcentrationRisk(BaseModel):
    top_position_pct: float
    top_3_positions_pct: float
    flag: str  # "low" | "medium" | "high"


class Performance(BaseModel):
    total_return_pct: float
    annualized_return_pct: Optional[float] = None


class BenchmarkComparison(BaseModel):
    benchmark: str
    portfolio_return_pct: float
    benchmark_return_pct: float
    alpha_pct: float


class Observation(BaseModel):
    severity: str  # "info" | "warning" | "critical"
    text: str


class PortfolioHealthOutput(BaseModel):
    concentration_risk: ConcentrationRisk
    performance: Performance
    benchmark_comparison: Optional[BenchmarkComparison] = None
    observations: List[Observation] = Field(default_factory=list)
    summary: str = ""
    disclaimer: str = (
        "This is not investment advice. Past performance is not indicative of future results. "
        "All investments involve risk. Please consult a qualified financial adviser before making "
        "any investment decisions."
    )


# ── HTTP layer ───────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    user_context: UserContext
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None


class PipelineMetadata(BaseModel):
    agent: str
    intent: str
    entities: ExtractedEntities
    safety_verdict: str
    classifier_confidence: float
    safety_latency_ms: float
    conversation_id: Optional[str] = None
