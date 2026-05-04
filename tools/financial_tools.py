"""
Phase 3: Financial Tools

Tool functions wired into the LangGraph agent:
1. Ratio calculator (P/E, EBITDA margin, D/E, ROE, current ratio)
2. YoY anomaly detector (z-score based, threshold configurable)
3. Trend forecaster (Prophet or statsmodels fallback)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from models.schemas import (
    FinancialRatio,
    Anomaly,
    AnomalyDirection,
    RiskLevel,
    ForecastResult,
    ForecastPoint,
)

logger = logging.getLogger(__name__)

Z_THRESHOLD = float(os.getenv("ANOMALY_Z_THRESHOLD", "2.5"))


# ── Ratio Calculator ──────────────────────────────────────────────────

class RatioCalculator:
    """
    Computes standard financial ratios from extracted KPI data.
    
    Each ratio includes the formula used and an interpretation,
    enabling the agent to explain its analysis.
    """

    RATIO_DEFINITIONS = {
        "pe_ratio": {
            "name": "P/E Ratio",
            "formula": "Stock Price / Earnings Per Share",
            "numerator": "stock_price",
            "denominator": "eps",
            "interpretation_high": "Stock may be overvalued or investors expect high growth",
            "interpretation_low": "Stock may be undervalued or company faces challenges",
            "benchmark": 25.0,  # S&P 500 average
        },
        "ebitda_margin": {
            "name": "EBITDA Margin",
            "formula": "EBITDA / Revenue × 100",
            "numerator": "ebitda",
            "denominator": "revenue",
            "multiplier": 100,
            "interpretation_high": "Strong operational profitability",
            "interpretation_low": "Thin margins — operational cost pressure",
            "benchmark": 20.0,
        },
        "debt_to_equity": {
            "name": "Debt-to-Equity Ratio",
            "formula": "Total Debt / Total Equity",
            "numerator": "total_debt",
            "denominator": "total_equity",
            "interpretation_high": "High leverage — greater financial risk",
            "interpretation_low": "Conservative capital structure",
            "benchmark": 1.5,
        },
        "roe": {
            "name": "Return on Equity",
            "formula": "Net Income / Shareholders' Equity × 100",
            "numerator": "net_income",
            "denominator": "shareholders_equity",
            "multiplier": 100,
            "interpretation_high": "Efficient use of equity capital",
            "interpretation_low": "Equity not generating strong returns",
            "benchmark": 15.0,
        },
        "current_ratio": {
            "name": "Current Ratio",
            "formula": "Current Assets / Current Liabilities",
            "numerator": "current_assets",
            "denominator": "current_liabilities",
            "interpretation_high": "Strong short-term liquidity",
            "interpretation_low": "Potential liquidity concerns",
            "benchmark": 2.0,
        },
        "gross_margin": {
            "name": "Gross Margin",
            "formula": "(Revenue - COGS) / Revenue × 100",
            "numerator": "gross_profit",
            "denominator": "revenue",
            "multiplier": 100,
            "interpretation_high": "Strong pricing power and cost control",
            "interpretation_low": "Thin product margins",
            "benchmark": 40.0,
        },
        "net_margin": {
            "name": "Net Profit Margin",
            "formula": "Net Income / Revenue × 100",
            "numerator": "net_income",
            "denominator": "revenue",
            "multiplier": 100,
            "interpretation_high": "High bottom-line profitability",
            "interpretation_low": "Expenses consuming most revenue",
            "benchmark": 10.0,
        },
    }

    def calculate(self, ratio_name: str, data: dict[str, float]) -> Optional[FinancialRatio]:
        """
        Calculate a specific financial ratio.

        Args:
            ratio_name: Key from RATIO_DEFINITIONS (e.g., 'pe_ratio')
            data: Dict of financial values (e.g., {'revenue': 1000000, 'ebitda': 250000})

        Returns:
            FinancialRatio or None if inputs are missing/invalid.
        """
        defn = self.RATIO_DEFINITIONS.get(ratio_name)
        if not defn:
            logger.warning(f"Unknown ratio: {ratio_name}")
            return None

        numerator = data.get(defn["numerator"])
        denominator = data.get(defn["denominator"])

        if numerator is None or denominator is None or denominator == 0:
            # stock_price is never in an annual report filing — log at debug not warning
            if defn["numerator"] == "stock_price" or defn["denominator"] == "stock_price":
                logger.debug(f"Skipping {ratio_name}: stock_price not available in annual reports")
            else:
                logger.warning(
                    f"Cannot compute {ratio_name}: "
                    f"missing {defn['numerator']}={numerator} or {defn['denominator']}={denominator}"
                )
            return None

        value = numerator / denominator
        multiplier = defn.get("multiplier", 1)
        value *= multiplier

        benchmark = defn.get("benchmark")
        if value > (benchmark or float("inf")):
            interpretation = defn["interpretation_high"]
        else:
            interpretation = defn["interpretation_low"]

        return FinancialRatio(
            name=defn["name"],
            value=round(value, 2),
            formula=defn["formula"],
            interpretation=interpretation,
            benchmark=benchmark,
        )

    def calculate_all(self, data: dict[str, float]) -> list[FinancialRatio]:
        """Calculate all possible ratios from the available data."""
        results = []
        for name in self.RATIO_DEFINITIONS:
            ratio = self.calculate(name, data)
            if ratio:
                results.append(ratio)
        return results


# ── Anomaly Detector ──────────────────────────────────────────────────

class AnomalyDetector:
    """
    Detects year-over-year anomalies using z-score analysis.

    Only flags genuine statistical outliers (z > threshold, default 2.5).
    Avoids noise by requiring sufficient historical data.
    """

    def __init__(self, z_threshold: float = Z_THRESHOLD, min_history: int = 3):
        self.z_threshold = z_threshold
        self.min_history = min_history

    def detect(
        self,
        metric_name: str,
        values: list[float],
        periods: list[str],
    ) -> list[Anomaly]:
        """
        Detect anomalies in a time series of financial values.

        Args:
            metric_name: Name of the metric (e.g., 'Revenue')
            values: Historical values in chronological order
            periods: Corresponding period labels (e.g., ['FY2022', 'FY2023', 'FY2024'])

        Returns:
            List of detected Anomaly objects.
        """
        if len(values) < self.min_history:
            logger.info(
                f"Skipping anomaly detection for {metric_name}: "
                f"only {len(values)} data points (need {self.min_history})"
            )
            return []

        anomalies = []

        # Compute YoY changes
        changes = []
        for i in range(1, len(values)):
            if values[i - 1] != 0:
                pct_change = (values[i] - values[i - 1]) / abs(values[i - 1]) * 100
            else:
                pct_change = 0.0
            changes.append(pct_change)

        if len(changes) < 2:
            return []

        # Compute z-scores of the changes
        mean_change = np.mean(changes)
        std_change = np.std(changes)

        if std_change == 0:
            return []  # No variation

        for i, change in enumerate(changes):
            z = abs(change - mean_change) / std_change

            if z >= self.z_threshold:
                direction = AnomalyDirection.SPIKE if change > 0 else AnomalyDirection.DROP
                severity = self._classify_severity(z)

                anomalies.append(Anomaly(
                    metric=metric_name,
                    current_value=values[i + 1],
                    previous_value=values[i],
                    change_pct=round(change, 2),
                    z_score=round(z, 2),
                    direction=direction,
                    severity=severity,
                    explanation=(
                        f"{metric_name} {'increased' if change > 0 else 'decreased'} "
                        f"by {abs(change):.1f}% from {periods[i]} to {periods[i + 1]}, "
                        f"which is {z:.1f} standard deviations from the historical mean change "
                        f"of {mean_change:.1f}%. This qualifies as a statistically significant "
                        f"{'spike' if change > 0 else 'drop'}."
                    ),
                ))

        return anomalies

    def detect_from_dataframe(
        self,
        df: pd.DataFrame,
        metric_col: str,
        period_col: str,
    ) -> list[Anomaly]:
        """Convenience method for DataFrames."""
        df_sorted = df.sort_values(period_col)
        values = df_sorted[metric_col].tolist()
        periods = df_sorted[period_col].astype(str).tolist()
        return self.detect(metric_col, values, periods)

    @staticmethod
    def _classify_severity(z_score: float) -> RiskLevel:
        if z_score >= 4.0:
            return RiskLevel.CRITICAL
        elif z_score >= 3.0:
            return RiskLevel.HIGH
        elif z_score >= 2.5:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW


# ── Trend Forecaster ──────────────────────────────────────────────────

class TrendForecaster:
    """
    Financial time-series forecaster.
    
    Attempts Prophet first, falls back to statsmodels linear trend
    if Prophet is unavailable.
    """

    def forecast(
        self,
        metric_name: str,
        values: list[float],
        periods: list[str],
        horizon: int = 4,
    ) -> ForecastResult:
        """
        Forecast future values for a metric.

        Args:
            metric_name: Name of the metric
            values: Historical values
            periods: Historical period labels
            horizon: Number of periods to forecast

        Returns:
            ForecastResult with predicted values and bounds.
        """
        try:
            return self._forecast_prophet(metric_name, values, periods, horizon)
        except (ImportError, Exception) as e:
            logger.info(f"Prophet unavailable ({e}), using linear trend")
            return self._forecast_linear(metric_name, values, periods, horizon)

    def _forecast_prophet(
        self,
        metric_name: str,
        values: list[float],
        periods: list[str],
        horizon: int,
    ) -> ForecastResult:
        """Forecast using Facebook Prophet."""
        from prophet import Prophet

        # Create Prophet DataFrame
        df = pd.DataFrame({
            "ds": pd.to_datetime(periods),
            "y": values,
        })

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.90,
        )
        model.fit(df)

        # Create future dates
        future = model.make_future_dataframe(periods=horizon, freq="QE")
        forecast = model.predict(future)

        # Extract forecast points
        points = []
        for _, row in forecast.tail(horizon).iterrows():
            points.append(ForecastPoint(
                period=row["ds"].strftime("%Y-Q%q") if hasattr(row["ds"], "quarter") else str(row["ds"].date()),
                predicted_value=round(row["yhat"], 2),
                lower_bound=round(row["yhat_lower"], 2),
                upper_bound=round(row["yhat_upper"], 2),
            ))

        # Calculate MAPE on historical fit
        historical = forecast.head(len(values))
        mape = np.mean(np.abs((values - historical["yhat"].values) / np.array(values))) * 100

        return ForecastResult(
            metric=metric_name,
            model_used="Prophet",
            horizon=horizon,
            points=points,
            mape=round(mape, 2),
        )

    def _forecast_linear(
        self,
        metric_name: str,
        values: list[float],
        periods: list[str],
        horizon: int,
    ) -> ForecastResult:
        """Fallback: simple linear trend with confidence intervals."""
        x = np.arange(len(values))
        y = np.array(values)

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Residual standard error
        y_pred = slope * x + intercept
        residual_std = np.sqrt(np.mean((y - y_pred) ** 2))

        # Forecast
        points = []
        for i in range(horizon):
            future_x = len(values) + i
            predicted = slope * future_x + intercept
            # 90% confidence interval
            margin = 1.645 * residual_std * np.sqrt(1 + 1 / len(values))
            points.append(ForecastPoint(
                period=f"Period +{i + 1}",
                predicted_value=round(predicted, 2),
                lower_bound=round(predicted - margin, 2),
                upper_bound=round(predicted + margin, 2),
            ))

        # MAPE
        mape = np.mean(np.abs((y - y_pred) / y)) * 100

        return ForecastResult(
            metric=metric_name,
            model_used="Linear Trend (statsmodels fallback)",
            horizon=horizon,
            points=points,
            mape=round(mape, 2),
        )


# ── Tool Registry (for LangGraph) ────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "calculate_financial_ratio",
        "description": (
            "Calculate a financial ratio from provided data. "
            "Available ratios: pe_ratio, ebitda_margin, debt_to_equity, roe, "
            "current_ratio, gross_margin, net_margin. "
            "Input data should include the required fields for the ratio."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ratio_name": {
                    "type": "string",
                    "enum": list(RatioCalculator.RATIO_DEFINITIONS.keys()),
                    "description": "Which ratio to calculate",
                },
                "data": {
                    "type": "object",
                    "description": "Financial data points needed for the ratio",
                },
            },
            "required": ["ratio_name", "data"],
        },
    },
    {
        "name": "detect_anomalies",
        "description": (
            "Detect year-over-year anomalies in a financial metric's time series. "
            "Flags only genuine statistical outliers (z-score > 2.5). "
            "Requires at least 3 historical data points."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "metric_name": {
                    "type": "string",
                    "description": "Name of the metric (e.g., 'Revenue', 'EBITDA')",
                },
                "values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Historical values in chronological order",
                },
                "periods": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Corresponding period labels",
                },
            },
            "required": ["metric_name", "values", "periods"],
        },
    },
    {
        "name": "forecast_metric",
        "description": (
            "Forecast future values for a financial metric using time-series analysis. "
            "Uses Prophet if available, otherwise falls back to linear trend."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "metric_name": {"type": "string"},
                "values": {
                    "type": "array",
                    "items": {"type": "number"},
                },
                "periods": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "horizon": {
                    "type": "integer",
                    "default": 4,
                    "description": "Number of periods to forecast",
                },
            },
            "required": ["metric_name", "values", "periods"],
        },
    },
]
