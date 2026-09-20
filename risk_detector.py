"""
risk_detector.py  –  Deterministic risk detection engine.
All thresholds, scores, and evidence values come from Python.
The LLM only receives these pre-computed facts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────
# Risk Data Model
# ─────────────────────────────────────────────

SEVERITY_LEVELS = ("Critical", "High", "Medium", "Low")

SEVERITY_COLORS = {
    "Critical": "#ef4444",
    "High":     "#f97316",
    "Medium":   "#eab308",
    "Low":      "#22c55e",
}


@dataclass
class RiskFinding:
    risk_id: str
    title: str
    category: str           # e.g. "Revenue", "Margin", "Customer Concentration"
    severity: str           # Critical / High / Medium / Low
    severity_score: float   # 0-100
    summary: str            # One-liner shown in the risk card
    evidence: dict[str, Any]    # Pre-computed facts handed to the LLM
    chart_data: pd.DataFrame | None = field(default=None, repr=False)
    chart_type: str = "bar"     # bar | line | pie
    chart_title: str = ""


# ─────────────────────────────────────────────
# Individual Risk Detectors
# ─────────────────────────────────────────────

def _detect_revenue_decline(df: pd.DataFrame) -> RiskFinding | None:
    """Detect sustained month-over-month revenue decline."""
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M").astype(str)
    monthly = df.groupby("month")["revenue"].sum().reset_index().sort_values("month")

    if len(monthly) < 3:
        return None

    # Look at last 3 months vs previous 3 months
    n = len(monthly)
    recent = monthly.iloc[max(n - 3, 0):]["revenue"].values
    prior  = monthly.iloc[max(n - 6, 0):max(n - 3, 0)]["revenue"].values

    if len(prior) == 0:
        return None

    recent_avg = float(np.mean(recent))
    prior_avg  = float(np.mean(prior))

    if prior_avg == 0:
        return None

    change_pct = (recent_avg - prior_avg) / prior_avg * 100

    if change_pct >= -5:
        return None  # Not a meaningful decline

    # Consecutive decline check
    revenues = monthly["revenue"].values
    consecutive_declines = 0
    for i in range(len(revenues) - 1, 0, -1):
        if revenues[i] < revenues[i - 1]:
            consecutive_declines += 1
        else:
            break

    if change_pct < -30:
        severity, score = "Critical", 90
    elif change_pct < -15:
        severity, score = "High", 72
    else:
        severity, score = "Medium", 50

    evidence = {
        "recent_3mo_avg_revenue": round(recent_avg, 2),
        "prior_3mo_avg_revenue": round(prior_avg, 2),
        "revenue_change_pct": round(change_pct, 1),
        "consecutive_declining_months": consecutive_declines,
        "recent_month_revenues": {
            str(monthly.iloc[i]["month"]): round(float(monthly.iloc[i]["revenue"]), 2)
            for i in range(max(n - 6, 0), n)
        },
    }

    return RiskFinding(
        risk_id="revenue_decline",
        title="Revenue Decline Detected",
        category="Revenue",
        severity=severity,
        severity_score=score,
        summary=f"Revenue dropped {abs(change_pct):.1f}% over the last 3 months vs prior period.",
        evidence=evidence,
        chart_data=monthly,
        chart_type="line",
        chart_title="Monthly Revenue Trend",
    )


def _detect_margin_compression(df: pd.DataFrame) -> RiskFinding | None:
    """Detect shrinking profit margins."""
    if "cost" not in df.columns:
        return None

    df = df.copy()
    df["month"] = df["date"].dt.to_period("M").astype(str)
    monthly = (
        df.groupby("month")
        .agg(revenue=("revenue", "sum"), cost=("cost", "sum"))
        .reset_index()
        .sort_values("month")
    )
    monthly["margin_pct"] = (monthly["revenue"] - monthly["cost"]) / monthly["revenue"] * 100

    if len(monthly) < 3:
        return None

    current_margin = float(monthly["margin_pct"].iloc[-1])
    avg_margin     = float(monthly["margin_pct"].mean())
    min_margin     = float(monthly["margin_pct"].min())

    margin_trend = float(monthly["margin_pct"].iloc[-1] - monthly["margin_pct"].iloc[0])

    if current_margin > 30 and margin_trend > -5:
        return None  # Healthy margins, no risk

    if current_margin < 10:
        severity, score = "Critical", 88
    elif current_margin < 20 or margin_trend < -10:
        severity, score = "High", 70
    elif margin_trend < -5:
        severity, score = "Medium", 48
    else:
        return None

    # Which products have lowest margin?
    if "product" in df.columns:
        prod_margin = (
            df.groupby("product")
            .agg(revenue=("revenue", "sum"), cost=("cost", "sum"))
            .reset_index()
        )
        prod_margin["margin_pct"] = (
            (prod_margin["revenue"] - prod_margin["cost"]) / prod_margin["revenue"] * 100
        )
        worst_products = prod_margin.nsmallest(3, "margin_pct")[
            ["product", "margin_pct"]
        ].to_dict("records")
    else:
        worst_products = []

    evidence = {
        "current_margin_pct": round(current_margin, 1),
        "average_margin_pct": round(avg_margin, 1),
        "minimum_margin_pct": round(min_margin, 1),
        "margin_trend_change_pct": round(margin_trend, 1),
        "lowest_margin_products": [
            {"product": r["product"], "margin_pct": round(r["margin_pct"], 1)}
            for r in worst_products
        ],
    }

    return RiskFinding(
        risk_id="margin_compression",
        title="Margin Compression Risk",
        category="Profitability",
        severity=severity,
        severity_score=score,
        summary=f"Current gross margin is {current_margin:.1f}%, down {abs(margin_trend):.1f}pp since period start.",
        evidence=evidence,
        chart_data=monthly[["month", "margin_pct"]],
        chart_type="line",
        chart_title="Monthly Gross Margin %",
    )


def _detect_customer_concentration(df: pd.DataFrame) -> RiskFinding | None:
    """Detect over-reliance on a small number of customers."""
    if "customer" not in df.columns:
        return None

    cust = (
        df.groupby("customer")["revenue"]
        .sum()
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    total = float(cust["revenue"].sum())
    if total == 0:
        return None

    cust["pct"] = cust["revenue"] / total * 100
    top1_pct  = float(cust.iloc[0]["pct"])
    top3_pct  = float(cust.head(3)["pct"].sum())
    n_customers = len(cust)

    if top1_pct < 30 and top3_pct < 60:
        return None  # Well diversified

    if top1_pct >= 50:
        severity, score = "Critical", 85
    elif top1_pct >= 35 or top3_pct >= 70:
        severity, score = "High", 68
    else:
        severity, score = "Medium", 45

    top_customers = cust.head(5)[["customer", "revenue", "pct"]].to_dict("records")

    evidence = {
        "total_customers": n_customers,
        "top_customer_revenue_pct": round(top1_pct, 1),
        "top_3_customers_revenue_pct": round(top3_pct, 1),
        "top_customer_name": str(cust.iloc[0]["customer"]),
        "top_customer_revenue": round(float(cust.iloc[0]["revenue"]), 2),
        "top_customers": [
            {
                "customer": r["customer"],
                "revenue": round(float(r["revenue"]), 2),
                "pct_of_total": round(float(r["pct"]), 1),
            }
            for r in top_customers
        ],
    }

    return RiskFinding(
        risk_id="customer_concentration",
        title="High Customer Concentration",
        category="Customer Risk",
        severity=severity,
        severity_score=score,
        summary=f"Top customer '{cust.iloc[0]['customer']}' accounts for {top1_pct:.1f}% of revenue.",
        evidence=evidence,
        chart_data=cust.head(8)[["customer", "revenue"]],
        chart_type="bar",
        chart_title="Revenue by Customer",
    )


def _detect_product_decline(df: pd.DataFrame) -> RiskFinding | None:
    """Detect products with declining revenue trends."""
    if "product" not in df.columns:
        return None

    df = df.copy()
    df["month"] = df["date"].dt.to_period("M").astype(str)
    months = sorted(df["month"].unique())
    if len(months) < 4:
        return None

    recent_months = months[-3:]
    prior_months  = months[-6:-3] if len(months) >= 6 else months[:-3]
    if not prior_months:
        return None

    recent_rev = df[df["month"].isin(recent_months)].groupby("product")["revenue"].sum()
    prior_rev  = df[df["month"].isin(prior_months)].groupby("product")["revenue"].sum()

    all_products = recent_rev.index.union(prior_rev.index)
    comparison = pd.DataFrame({"recent": recent_rev, "prior": prior_rev}).reindex(all_products).fillna(0)
    comparison["change_pct"] = np.where(
        comparison["prior"] > 0,
        (comparison["recent"] - comparison["prior"]) / comparison["prior"] * 100,
        0.0,
    )

    declining = comparison[comparison["change_pct"] < -15].sort_values("change_pct")
    if len(declining) == 0:
        return None

    worst_product = declining.index[0]
    worst_change  = float(declining.iloc[0]["change_pct"])

    if worst_change < -40:
        severity, score = "High", 75
    else:
        severity, score = "Medium", 50

    evidence = {
        "declining_products_count": int(len(declining)),
        "worst_declining_product": str(worst_product),
        "worst_decline_pct": round(worst_change, 1),
        "declining_products": [
            {
                "product": str(idx),
                "recent_revenue": round(float(row["recent"]), 2),
                "prior_revenue": round(float(row["prior"]), 2),
                "change_pct": round(float(row["change_pct"]), 1),
            }
            for idx, row in declining.iterrows()
        ],
    }

    return RiskFinding(
        risk_id="product_decline",
        title="Product Revenue Decline",
        category="Product",
        severity=severity,
        severity_score=score,
        summary=f"{len(declining)} product(s) showing >15% revenue decline. Worst: '{worst_product}' ({worst_change:.1f}%).",
        evidence=evidence,
        chart_data=declining.reset_index().rename(columns={"index": "product"}),
        chart_type="bar",
        chart_title="Product Revenue Change %",
    )


def _detect_regional_imbalance(df: pd.DataFrame) -> RiskFinding | None:
    """Detect over-dependence on a single region."""
    if "region" not in df.columns:
        return None

    reg = (
        df.groupby("region")["revenue"]
        .sum()
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    total = float(reg["revenue"].sum())
    if total == 0:
        return None

    reg["pct"] = reg["revenue"] / total * 100
    top_pct = float(reg.iloc[0]["pct"])

    if top_pct < 40:
        return None

    if top_pct >= 60:
        severity, score = "High", 65
    else:
        severity, score = "Medium", 42

    evidence = {
        "total_regions": int(len(reg)),
        "top_region": str(reg.iloc[0]["region"]),
        "top_region_revenue_pct": round(top_pct, 1),
        "region_breakdown": [
            {
                "region": r["region"],
                "revenue": round(float(r["revenue"]), 2),
                "pct": round(float(r["pct"]), 1),
            }
            for _, r in reg.iterrows()
        ],
    }

    return RiskFinding(
        risk_id="regional_imbalance",
        title="Regional Revenue Imbalance",
        category="Geographic",
        severity=severity,
        severity_score=score,
        summary=f"Region '{reg.iloc[0]['region']}' accounts for {top_pct:.1f}% of total revenue.",
        evidence=evidence,
        chart_data=reg[["region", "revenue"]],
        chart_type="bar",
        chart_title="Revenue by Region",
    )


def _detect_revenue_volatility(df: pd.DataFrame) -> RiskFinding | None:
    """Detect high month-to-month revenue volatility (CV)."""
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M").astype(str)
    monthly = df.groupby("month")["revenue"].sum()

    if len(monthly) < 4:
        return None

    mean_rev = float(monthly.mean())
    std_rev  = float(monthly.std())
    cv       = (std_rev / mean_rev * 100) if mean_rev else 0

    if cv < 20:
        return None

    if cv >= 50:
        severity, score = "High", 70
    elif cv >= 35:
        severity, score = "Medium", 52
    else:
        severity, score = "Low", 30

    evidence = {
        "monthly_revenue_mean": round(mean_rev, 2),
        "monthly_revenue_std": round(std_rev, 2),
        "coefficient_of_variation_pct": round(cv, 1),
        "min_monthly_revenue": round(float(monthly.min()), 2),
        "max_monthly_revenue": round(float(monthly.max()), 2),
    }

    monthly_df = monthly.reset_index()
    monthly_df.columns = ["month", "revenue"]

    return RiskFinding(
        risk_id="revenue_volatility",
        title="High Revenue Volatility",
        category="Revenue",
        severity=severity,
        severity_score=score,
        summary=f"Monthly revenue coefficient of variation is {cv:.1f}% — indicating inconsistent cash flow.",
        evidence=evidence,
        chart_data=monthly_df,
        chart_type="line",
        chart_title="Monthly Revenue (Volatility View)",
    )


def _detect_low_margin_products(df: pd.DataFrame) -> RiskFinding | None:
    """Detect products with margins significantly below average."""
    if "product" not in df.columns or "cost" not in df.columns:
        return None

    prod = (
        df.groupby("product")
        .agg(revenue=("revenue", "sum"), cost=("cost", "sum"))
        .reset_index()
    )
    prod["margin_pct"] = (prod["revenue"] - prod["cost"]) / prod["revenue"].replace(0, np.nan) * 100
    prod = prod.dropna(subset=["margin_pct"])

    if len(prod) < 2:
        return None

    avg_margin = float(prod["margin_pct"].mean())
    threshold  = avg_margin - 10  # 10pp below average

    low_margin = prod[prod["margin_pct"] < threshold].sort_values("margin_pct")
    if len(low_margin) == 0:
        return None

    worst      = low_margin.iloc[0]
    worst_name = str(worst["product"])
    worst_marg = float(worst["margin_pct"])

    if worst_marg < 10:
        severity, score = "High", 72
    else:
        severity, score = "Medium", 48

    evidence = {
        "average_product_margin_pct": round(avg_margin, 1),
        "below_average_threshold_pct": round(threshold, 1),
        "low_margin_products_count": int(len(low_margin)),
        "worst_margin_product": worst_name,
        "worst_margin_pct": round(worst_marg, 1),
        "low_margin_products": [
            {
                "product": str(r["product"]),
                "margin_pct": round(float(r["margin_pct"]), 1),
                "revenue": round(float(r["revenue"]), 2),
            }
            for _, r in low_margin.iterrows()
        ],
    }

    return RiskFinding(
        risk_id="low_margin_products",
        title="Low-Margin Products Detected",
        category="Profitability",
        severity=severity,
        severity_score=score,
        summary=f"{len(low_margin)} product(s) have margins >10pp below average ({avg_margin:.1f}%). Worst: '{worst_name}' at {worst_marg:.1f}%.",
        evidence=evidence,
        chart_data=prod[["product", "margin_pct"]].sort_values("margin_pct"),
        chart_type="bar",
        chart_title="Gross Margin % by Product",
    )


# ─────────────────────────────────────────────
# Main Detection Entry Point
# ─────────────────────────────────────────────

def detect_all_risks(df: pd.DataFrame) -> list[RiskFinding]:
    """Run all detectors and return findings sorted by severity score."""
    detectors = [
        _detect_revenue_decline,
        _detect_margin_compression,
        _detect_customer_concentration,
        _detect_product_decline,
        _detect_regional_imbalance,
        _detect_revenue_volatility,
        _detect_low_margin_products,
    ]

    findings: list[RiskFinding] = []
    for detector in detectors:
        try:
            result = detector(df)
            if result is not None:
                findings.append(result)
        except Exception:
            pass  # Never crash the app on a detector failure

    findings.sort(key=lambda r: r.severity_score, reverse=True)
    return findings
