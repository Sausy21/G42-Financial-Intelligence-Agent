"""
Generate synthetic financial data for demo and testing.

Creates a sample CSV with 5 years of quarterly financial data,
mimicking a tech company's growth trajectory.

Usage:
    python data/generate_sample.py
"""

import csv
import random
from pathlib import Path


def generate_sample_financials(output_path: str = "data/sample_financials.csv"):
    """Generate 5 years of quarterly financial data."""
    random.seed(42)

    headers = [
        "period", "revenue", "cogs", "gross_profit", "operating_expenses",
        "operating_income", "net_income", "ebitda", "eps",
        "total_assets", "total_liabilities", "shareholders_equity",
        "total_debt", "current_assets", "current_liabilities",
        "operating_cash_flow", "capex", "free_cash_flow",
    ]

    rows = []
    base_revenue = 5_000_000_000  # $5B starting revenue

    for year in range(2020, 2025):
        for q in range(1, 5):
            period = f"Q{q} {year}"

            # Revenue grows ~5% per quarter with noise, big spike in 2024
            growth = 1.05 + random.uniform(-0.02, 0.03)
            if year == 2024:
                growth *= 1.10  # 10% extra growth — anomaly year
            base_revenue *= growth
            revenue = round(base_revenue)

            cogs = round(revenue * random.uniform(0.35, 0.42))
            gross_profit = revenue - cogs
            opex = round(revenue * random.uniform(0.25, 0.32))
            operating_income = gross_profit - opex
            net_income = round(operating_income * random.uniform(0.75, 0.85))
            ebitda = round(operating_income + revenue * random.uniform(0.05, 0.08))
            eps = round(net_income / 2_500_000_000 * 100) / 100  # ~2.5B shares

            total_assets = round(revenue * random.uniform(2.5, 3.5))
            total_liabilities = round(total_assets * random.uniform(0.35, 0.50))
            shareholders_equity = total_assets - total_liabilities
            total_debt = round(total_liabilities * random.uniform(0.5, 0.7))
            current_assets = round(total_assets * random.uniform(0.25, 0.40))
            current_liabilities = round(total_liabilities * random.uniform(0.3, 0.5))

            ocf = round(net_income * random.uniform(1.2, 1.5))
            capex = round(revenue * random.uniform(0.08, 0.15))
            fcf = ocf - capex

            rows.append([
                period, revenue, cogs, gross_profit, opex,
                operating_income, net_income, ebitda, eps,
                total_assets, total_liabilities, shareholders_equity,
                total_debt, current_assets, current_liabilities,
                ocf, capex, fcf,
            ])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Generated {len(rows)} rows → {output}")
    print(f"Revenue range: ${rows[0][1]:,.0f} → ${rows[-1][1]:,.0f}")

    return output


if __name__ == "__main__":
    generate_sample_financials()
