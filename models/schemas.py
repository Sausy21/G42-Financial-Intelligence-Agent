"""
Pydantic models for structured agent output.

Every response from the Financial Intelligence Agent conforms to these schemas,
ensuring downstream systems can always parse the output programmatically.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyDirection(str, Enum):
    SPIKE = "spike"
    DROP = "drop"


class EscalationAction(str, Enum):
    NONE = "none"
    FLAG_FOR_REVIEW = "flag_for_review"
    ESCALATE_TO_CFO = "escalate_to_cfo"
    BLOCK = "block"


# ── Citation ───────────────────────────────────────────────────────────

class Citation(BaseModel):
    """A grounded reference to a source document section."""
    source_document: str = Field(description="Filename or document identifier")
    section: str = Field(description="Section heading or table name")
    page: Optional[int] = Field(default=None, description="Page number if applicable")
    text_excerpt: str = Field(description="Exact excerpt from source (≤100 words)")
    confidence: float = Field(ge=0.0, le=1.0, description="Retrieval confidence score")


# ── Financial KPIs ─────────────────────────────────────────────────────

class FinancialKPI(BaseModel):
    """A single extracted financial metric."""
    name: str = Field(description="KPI name, e.g. 'Revenue', 'EBITDA', 'EPS'")
    value: float = Field(description="Numeric value")
    unit: str = Field(default="USD", description="Currency or unit")
    period: str = Field(description="Fiscal period, e.g. 'FY2024', 'Q3 2024'")
    source: Citation = Field(description="Where this number was extracted from")


class FinancialRatio(BaseModel):
    """A computed financial ratio."""
    name: str = Field(description="Ratio name, e.g. 'P/E Ratio', 'Debt-to-Equity'")
    value: float = Field(description="Computed ratio value")
    formula: str = Field(description="Formula used, e.g. 'Net Income / Shares Outstanding'")
    interpretation: str = Field(description="What this ratio means in context")
    benchmark: Optional[float] = Field(default=None, description="Industry benchmark if available")


# ── Anomaly ────────────────────────────────────────────────────────────

class Anomaly(BaseModel):
    """A detected year-over-year anomaly."""
    metric: str = Field(description="The KPI that exhibited the anomaly")
    current_value: float
    previous_value: float
    change_pct: float = Field(description="Percentage change")
    z_score: float = Field(description="Z-score of the change vs historical distribution")
    direction: AnomalyDirection
    severity: RiskLevel
    explanation: Optional[str] = Field(default=None, description="Agent's interpretation")
    citation: Optional[Citation] = Field(default=None)


# ── Forecast ───────────────────────────────────────────────────────────

class ForecastPoint(BaseModel):
    """A single forecast data point."""
    period: str
    predicted_value: float
    lower_bound: float
    upper_bound: float


class ForecastResult(BaseModel):
    """Time-series forecast output."""
    metric: str
    model_used: str = Field(description="e.g. 'Prophet', 'ARIMA', 'Linear Trend'")
    horizon: int = Field(description="Number of periods forecasted")
    points: list[ForecastPoint]
    mape: Optional[float] = Field(default=None, description="Mean Absolute Percentage Error")


# ── Transaction Screening ─────────────────────────────────────────────

class TransactionScreen(BaseModel):
    """Result of a transaction risk screening."""
    transaction_id: str
    amount: float
    entity: str
    risk_level: RiskLevel
    escalation: EscalationAction
    reasons: list[str] = Field(description="Reasons for the risk classification")
    governance_rule: str = Field(description="Which governance rule triggered this")


# ── Agent Response ─────────────────────────────────────────────────────

class AgentResponse(BaseModel):
    """
    The top-level structured response from the Financial Intelligence Agent.
    Every agent output conforms to this schema.
    """
    answer: str = Field(description="Natural language answer to the user's query")
    citations: list[Citation] = Field(default_factory=list, description="Source references")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence in the answer")

    # Optional structured data attached to the response
    kpis: list[FinancialKPI] = Field(default_factory=list)
    ratios: list[FinancialRatio] = Field(default_factory=list)
    anomalies: list[Anomaly] = Field(default_factory=list)
    forecasts: list[ForecastResult] = Field(default_factory=list)
    transaction_screens: list[TransactionScreen] = Field(default_factory=list)

    # Audit metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_used: str = Field(default="llama3.1:8b")
    tools_invoked: list[str] = Field(default_factory=list, description="Tools called during this response")
    governance_flags: list[str] = Field(default_factory=list, description="Any governance rules triggered")


# ── Document Chunk ─────────────────────────────────────────────────────

class DocumentChunk(BaseModel):
    """A chunk of text from an ingested document, ready for embedding."""
    chunk_id: str
    document_name: str
    section_heading: str
    page_number: Optional[int] = None
    text: str
    tables: list[dict] = Field(default_factory=list, description="Extracted tables as list of dicts")
    char_count: int = Field(default=0)
    token_estimate: int = Field(default=0)


# ── Agent State (for LangGraph) ───────────────────────────────────────

class AgentState(BaseModel):
    """State object passed between LangGraph nodes."""
    messages: list[dict] = Field(default_factory=list)
    documents_loaded: list[str] = Field(default_factory=list)
    extracted_kpis: list[FinancialKPI] = Field(default_factory=list)
    detected_anomalies: list[Anomaly] = Field(default_factory=list)
    retrieval_context: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    governance_flags: list[str] = Field(default_factory=list)
    current_query: str = ""
    iteration_count: int = 0
    max_iterations: int = 5
