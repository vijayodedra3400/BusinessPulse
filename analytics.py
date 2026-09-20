# -*- coding: utf-8 -*-
"""
analytics.py  --  Pure Python/Pandas/NumPy business analytics engine.
===========================================================================
ALL numbers shown to users originate here.  No LLM touches this file.

Public API
----------
load_and_validate(df)           -> (df_clean, errors, warnings)
calculate_kpis(df)              -> dict
calculate_period_metrics(df)    -> dict
calculate_product_metrics(df)   -> dict
calculate_region_metrics(df)    -> dict
calculate_customer_metrics(df)  -> dict
calculate_profit_metrics(df)    -> dict

Legacy helpers (kept for app.py compatibility)
----------------------------------------------
compute_kpis(df)                -> KPISummary
monthly_revenue_trend(df)       -> DataFrame
revenue_by_product(df)          -> DataFrame | None
revenue_by_region(df)           -> DataFrame | None
revenue_by_category(df)         -> DataFrame | None
margin_by_product(df)           -> DataFrame | None
customer_revenue_concentration(df) -> DataFrame | None
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: frozenset = frozenset({"date", "revenue"})
OPTIONAL_COLUMNS: frozenset = frozenset(
    {"product", "category", "region", "customer", "quantity", "cost"}
)
ALL_EXPECTED: frozenset = REQUIRED_COLUMNS | OPTIONAL_COLUMNS

_MIN_ROWS_FOR_TREND = 3
_INACTIVE_CUSTOMER_MONTHS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Division that never raises -- returns fallback on zero/NaN denominator."""
    if denominator == 0 or not np.isfinite(float(denominator)):
        return fallback
    result = numerator / denominator
    return float(result) if np.isfinite(result) else fallback


def _pct_change(new_val: float, old_val: float):
    """Return percentage change; None when old is 0 or NaN."""
    if old_val == 0 or not np.isfinite(old_val):
        return None
    return (new_val - old_val) / old_val * 100


def _col(df: pd.DataFrame, name: str) -> bool:
    """True if column exists AND has at least one non-null value."""
    return name in df.columns and df[name].notna().any()


# ---------------------------------------------------------------------------
# 1. Load & Validate
# ---------------------------------------------------------------------------

def load_and_validate(df: pd.DataFrame) -> tuple:
    """
    Normalise, validate and clean an uploaded sales DataFrame.

    Returns
    -------
    (cleaned_df, errors, warnings)
        errors   -- fatal problems; downstream analysis should abort.
        warnings -- non-fatal data quality notes.
    """
    errors: list = []
    warnings_out: list = []

    if df is None or len(df) == 0:
        errors.append("The uploaded file is empty.")
        return (df if df is not None else pd.DataFrame()), errors, warnings_out

    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Required columns
    missing_required = REQUIRED_COLUMNS - set(df.columns)
    if missing_required:
        errors.append(
            f"Missing required columns: {sorted(missing_required)}. "
            f"Found: {sorted(df.columns.tolist())}"
        )
        return df, errors, warnings_out

    # Optional columns warning
    missing_optional = OPTIONAL_COLUMNS - set(df.columns)
    if missing_optional:
        warnings_out.append(
            f"Optional columns not present (analysis will be partial): "
            f"{sorted(missing_optional)}"
        )

    # Parse date
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception as exc:
        errors.append(f"Cannot parse 'date' column: {exc}")
        return df, errors, warnings_out

    bad_dates = df["date"].isna().sum()
    if bad_dates > 0:
        warnings_out.append(f"'date': {bad_dates} rows have unparseable dates and were dropped.")
        df = df.dropna(subset=["date"])

    # Cast numeric columns
    for col in ["revenue", "cost", "quantity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            n_bad = int(df[col].isna().sum())
            if n_bad > 0:
                warnings_out.append(f"'{col}': {n_bad} non-numeric value(s) set to NaN.")

    # Drop rows where revenue is NaN
    if df["revenue"].isna().any():
        n_drop = int(df["revenue"].isna().sum())
        warnings_out.append(f"{n_drop} row(s) with missing revenue dropped.")
        df = df.dropna(subset=["revenue"])

    # Data quality checks
    if (df["revenue"] < 0).any():
        n_neg = int((df["revenue"] < 0).sum())
        warnings_out.append(f"{n_neg} row(s) with negative revenue detected.")
    if _col(df, "cost") and (df["cost"] < 0).any():
        n_neg = int((df["cost"] < 0).sum())
        warnings_out.append(f"{n_neg} row(s) with negative cost detected.")
    if len(df) < 10:
        warnings_out.append(f"Dataset has only {len(df)} rows. Analysis may be unreliable.")

    df = df.sort_values("date").reset_index(drop=True)
    return df, errors, warnings_out


# ---------------------------------------------------------------------------
# 2. Core KPIs
# ---------------------------------------------------------------------------

def calculate_kpis(df: pd.DataFrame) -> dict:
    """
    Compute core business KPI metrics from a clean DataFrame.

    Returns a flat dict with scalar metrics covering revenue, cost, profit,
    orders, AOV, date range, and segment presence flags.
    """
    if df is None or len(df) == 0:
        return _empty_kpis()

    has_cost     = _col(df, "cost")
    has_quantity = _col(df, "quantity")
    has_product  = _col(df, "product")
    has_region   = _col(df, "region")
    has_customer = _col(df, "customer")

    total_revenue = float(df["revenue"].sum())
    total_cost    = float(df["cost"].sum()) if has_cost else 0.0
    total_profit  = total_revenue - total_cost
    profit_margin = _safe_div(total_profit, total_revenue) * 100

    total_orders = len(df)
    total_qty    = float(df["quantity"].sum()) if has_quantity else 0.0
    aov          = _safe_div(total_revenue, total_orders)

    date_start = df["date"].min()
    date_end   = df["date"].max()
    date_range = int((date_end - date_start).days)

    unique_products  = int(df["product"].nunique())  if has_product  else 0
    unique_regions   = int(df["region"].nunique())   if has_region   else 0
    unique_customers = int(df["customer"].nunique()) if has_customer else 0

    top_product  = str(df.groupby("product")["revenue"].sum().idxmax())  if has_product  else None
    top_region   = str(df.groupby("region")["revenue"].sum().idxmax())   if has_region   else None
    top_customer = str(df.groupby("customer")["revenue"].sum().idxmax()) if has_customer else None

    return {
        "total_revenue":     total_revenue,
        "total_cost":        total_cost,
        "total_profit":      total_profit,
        "profit_margin_pct": profit_margin,
        "total_orders":      total_orders,
        "total_quantity":    total_qty,
        "avg_order_value":   aov,
        "date_range_days":   date_range,
        "date_start":        date_start,
        "date_end":          date_end,
        "has_cost":          has_cost,
        "has_quantity":      has_quantity,
        "has_product":       has_product,
        "has_region":        has_region,
        "has_customer":      has_customer,
        "unique_products":   unique_products,
        "unique_regions":    unique_regions,
        "unique_customers":  unique_customers,
        "top_product":       top_product,
        "top_region":        top_region,
        "top_customer":      top_customer,
    }


def _empty_kpis() -> dict:
    return {
        "total_revenue": 0.0, "total_cost": 0.0, "total_profit": 0.0,
        "profit_margin_pct": 0.0, "total_orders": 0, "total_quantity": 0.0,
        "avg_order_value": 0.0, "date_range_days": 0,
        "date_start": None, "date_end": None,
        "has_cost": False, "has_quantity": False,
        "has_product": False, "has_region": False, "has_customer": False,
        "unique_products": 0, "unique_regions": 0, "unique_customers": 0,
        "top_product": None, "top_region": None, "top_customer": None,
    }


# ---------------------------------------------------------------------------
# 3. Period / Time Analytics
# ---------------------------------------------------------------------------

def calculate_period_metrics(df: pd.DataFrame) -> dict:
    """
    Time-series analytics: revenue trend, period-over-period growth,
    AOV trend, profit trend, order-count trend.

    Automatically uses daily granularity for <=60-day span, monthly otherwise.

    Returns dict with keys:
        granularity, trend_df, current_period_rev, previous_period_rev,
        revenue_growth_pct, revenue_mom_pct, profit_trend_df,
        order_trend_df, aov_trend_df
    """
    if df is None or len(df) < _MIN_ROWS_FOR_TREND:
        return _empty_period_metrics()

    df = df.copy()
    n_days = int((df["date"].max() - df["date"].min()).days)
    granularity = "daily" if n_days <= 60 else "monthly"

    if granularity == "daily":
        df["period"] = df["date"].dt.date.astype(str)
    else:
        df["period"] = df["date"].dt.to_period("M").astype(str)

    has_cost = _col(df, "cost")

    agg_spec = {
        "revenue": ("revenue", "sum"),
        "orders":  ("revenue", "count"),
    }
    if has_cost:
        agg_spec["cost"] = ("cost", "sum")

    trend = df.groupby("period").agg(**agg_spec).reset_index().sort_values("period")
    trend["aov"] = trend["revenue"] / trend["orders"].replace(0, np.nan)
    if has_cost:
        trend["profit"] = trend["revenue"] - trend["cost"]

    current_rev  = float(trend["revenue"].iloc[-1]) if len(trend) >= 1 else 0.0
    previous_rev = float(trend["revenue"].iloc[-2]) if len(trend) >= 2 else 0.0
    growth_pct   = _pct_change(current_rev, previous_rev)

    # MoM always on monthly grain
    df_m = df.copy()
    df_m["_m"] = df_m["date"].dt.to_period("M").astype(str)
    monthly_rev = df_m.groupby("_m")["revenue"].sum().reset_index().sort_values("_m")
    mom_pct = None
    if len(monthly_rev) >= 2:
        last = float(monthly_rev["revenue"].iloc[-1])
        prev = float(monthly_rev["revenue"].iloc[-2])
        mom_pct = _pct_change(last, prev)

    profit_trend = trend[["period", "profit"]].copy() if has_cost else None
    order_trend  = trend[["period", "orders"]].copy()
    aov_trend    = trend[["period", "aov"]].copy()

    return {
        "granularity":         granularity,
        "trend_df":            trend,
        "current_period_rev":  current_rev,
        "previous_period_rev": previous_rev,
        "revenue_growth_pct":  growth_pct,
        "revenue_mom_pct":     mom_pct,
        "profit_trend_df":     profit_trend,
        "order_trend_df":      order_trend,
        "aov_trend_df":        aov_trend,
    }


def _empty_period_metrics() -> dict:
    return {
        "granularity": "monthly",
        "trend_df": pd.DataFrame(),
        "current_period_rev": 0.0,
        "previous_period_rev": 0.0,
        "revenue_growth_pct": None,
        "revenue_mom_pct": None,
        "profit_trend_df": None,
        "order_trend_df": pd.DataFrame(),
        "aov_trend_df": pd.DataFrame(),
    }


# ---------------------------------------------------------------------------
# 4. Product Analytics
# ---------------------------------------------------------------------------

def calculate_product_metrics(df: pd.DataFrame) -> dict:
    """
    Product-level analytics.

    Returns None values for all keys when the 'product' column is absent.

    Keys: available, revenue_by_product, product_growth_df,
          profitability_df, top_product, n_products
    """
    if not _col(df, "product"):
        return {
            "available": False,
            "revenue_by_product": None,
            "product_growth_df": None,
            "profitability_df": None,
            "top_product": None,
            "n_products": 0,
        }

    df = df.copy()
    has_cost = _col(df, "cost")

    # Revenue by product + contribution %
    rev_prod = (
        df.groupby("product")["revenue"]
        .sum()
        .reset_index()
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
    total_rev = float(rev_prod["revenue"].sum())
    rev_prod["contribution_pct"] = rev_prod["revenue"].apply(
        lambda r: _safe_div(r, total_rev) * 100
    )

    # Product growth (first half vs second half of date range)
    df["_month"] = df["date"].dt.to_period("M").astype(str)
    months = sorted(df["_month"].unique())
    growth_df = None

    if len(months) >= 4:
        mid           = len(months) // 2
        prior_months  = months[:mid]
        recent_months = months[mid:]
        prior_rev_p   = df[df["_month"].isin(prior_months)].groupby("product")["revenue"].sum()
        recent_rev_p  = df[df["_month"].isin(recent_months)].groupby("product")["revenue"].sum()
        all_prods     = prior_rev_p.index.union(recent_rev_p.index)
        growth_df = pd.DataFrame({
            "prior_rev":  prior_rev_p.reindex(all_prods, fill_value=0),
            "recent_rev": recent_rev_p.reindex(all_prods, fill_value=0),
        }).reset_index().rename(columns={"index": "product"})
        growth_df["growth_pct"] = growth_df.apply(
            lambda row: _pct_change(row["recent_rev"], row["prior_rev"]),
            axis=1,
        )
        growth_df = growth_df.sort_values("growth_pct")

    # Profitability by product
    profit_df = None
    if has_cost:
        profit_df = (
            df.groupby("product")
            .agg(revenue=("revenue", "sum"), cost=("cost", "sum"))
            .reset_index()
        )
        profit_df["profit"]     = profit_df["revenue"] - profit_df["cost"]
        profit_df["margin_pct"] = profit_df.apply(
            lambda row: _safe_div(row["profit"], row["revenue"]) * 100, axis=1
        )
        profit_df = profit_df.sort_values("margin_pct").reset_index(drop=True)

    top_product = str(rev_prod.iloc[0]["product"]) if len(rev_prod) > 0 else None

    return {
        "available":          True,
        "revenue_by_product": rev_prod,
        "product_growth_df":  growth_df,
        "profitability_df":   profit_df,
        "top_product":        top_product,
        "n_products":         int(df["product"].nunique()),
    }


# ---------------------------------------------------------------------------
# 5. Region Analytics
# ---------------------------------------------------------------------------

def calculate_region_metrics(df: pd.DataFrame) -> dict:
    """
    Regional revenue analytics.

    Keys: available, revenue_by_region, regional_growth_df, top_region, n_regions
    """
    if not _col(df, "region"):
        return {
            "available": False,
            "revenue_by_region": None,
            "regional_growth_df": None,
            "top_region": None,
            "n_regions": 0,
        }

    df = df.copy()

    rev_reg = (
        df.groupby("region")["revenue"]
        .sum()
        .reset_index()
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
    total_rev = float(rev_reg["revenue"].sum())
    rev_reg["contribution_pct"] = rev_reg["revenue"].apply(
        lambda r: _safe_div(r, total_rev) * 100
    )

    df["_month"] = df["date"].dt.to_period("M").astype(str)
    months = sorted(df["_month"].unique())
    growth_df = None

    if len(months) >= 4:
        mid           = len(months) // 2
        prior_months  = months[:mid]
        recent_months = months[mid:]
        prior_rev_r   = df[df["_month"].isin(prior_months)].groupby("region")["revenue"].sum()
        recent_rev_r  = df[df["_month"].isin(recent_months)].groupby("region")["revenue"].sum()
        all_regions   = prior_rev_r.index.union(recent_rev_r.index)
        growth_df = pd.DataFrame({
            "prior_rev":  prior_rev_r.reindex(all_regions, fill_value=0),
            "recent_rev": recent_rev_r.reindex(all_regions, fill_value=0),
        }).reset_index().rename(columns={"index": "region"})
        growth_df["growth_pct"] = growth_df.apply(
            lambda row: _pct_change(row["recent_rev"], row["prior_rev"]),
            axis=1,
        )
        growth_df = growth_df.sort_values("growth_pct", ascending=False).reset_index(drop=True)

    top_region = str(rev_reg.iloc[0]["region"]) if len(rev_reg) > 0 else None

    return {
        "available":          True,
        "revenue_by_region":  rev_reg,
        "regional_growth_df": growth_df,
        "top_region":         top_region,
        "n_regions":          int(df["region"].nunique()),
    }


# ---------------------------------------------------------------------------
# 6. Customer Analytics
# ---------------------------------------------------------------------------

def calculate_customer_metrics(df: pd.DataFrame) -> dict:
    """
    Customer-level analytics: concentration and churn signals.

    Keys: available, customer_revenue_df, inactive_customers,
          top_customer, n_customers, top_customer_pct, top3_customer_pct,
          has_inactive
    """
    if not _col(df, "customer"):
        return {
            "available": False,
            "customer_revenue_df": None,
            "inactive_customers": [],
            "top_customer": None,
            "n_customers": 0,
            "top_customer_pct": 0.0,
            "top3_customer_pct": 0.0,
            "has_inactive": False,
        }

    df = df.copy()

    cust = (
        df.groupby("customer")
        .agg(revenue=("revenue", "sum"), orders=("revenue", "count"))
        .reset_index()
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
    cust["avg_order"] = cust["revenue"] / cust["orders"].replace(0, np.nan)
    total_rev = float(cust["revenue"].sum())
    cust["pct_of_total"]  = cust["revenue"].apply(lambda r: _safe_div(r, total_rev) * 100)
    cust["cumulative_pct"] = cust["pct_of_total"].cumsum()

    top_customer      = str(cust.iloc[0]["customer"]) if len(cust) > 0 else None
    top_customer_pct  = float(cust.iloc[0]["pct_of_total"]) if len(cust) > 0 else 0.0
    top3_customer_pct = float(cust.head(3)["pct_of_total"].sum())

    # Inactive customers (no orders in last N months, only when enough history)
    date_end   = df["date"].max()
    date_start = df["date"].min()
    history_months = (date_end - date_start).days / 30
    has_inactive = False
    inactive_customers: list = []

    if history_months >= (_INACTIVE_CUSTOMER_MONTHS + 1):
        has_inactive = True
        cutoff = date_end - pd.DateOffset(months=_INACTIVE_CUSTOMER_MONTHS)
        recent_customers = set(df[df["date"] >= cutoff]["customer"].unique())
        all_customers    = set(df["customer"].unique())
        inactive_customers = sorted(all_customers - recent_customers)

    return {
        "available":            True,
        "customer_revenue_df":  cust,
        "inactive_customers":   inactive_customers,
        "top_customer":         top_customer,
        "n_customers":          int(df["customer"].nunique()),
        "top_customer_pct":     top_customer_pct,
        "top3_customer_pct":    top3_customer_pct,
        "has_inactive":         has_inactive,
    }


# ---------------------------------------------------------------------------
# 7. Profit Analytics
# ---------------------------------------------------------------------------

def calculate_profit_metrics(df: pd.DataFrame) -> dict:
    """
    Profitability analytics at aggregate and product level.
    Returns available=False when the 'cost' column is absent.

    Keys: available, overall_margin_pct, margin_trend_df, product_profit_df,
          low_margin_products, margin_compressed, margin_change_pp,
          avg_margin_pct, min_margin_pct, max_margin_pct
    """
    if not _col(df, "cost"):
        return {
            "available": False,
            "overall_margin_pct": None,
            "margin_trend_df": None,
            "product_profit_df": None,
            "low_margin_products": [],
            "margin_compressed": False,
            "margin_change_pp": 0.0,
            "avg_margin_pct": None,
            "min_margin_pct": None,
            "max_margin_pct": None,
        }

    df = df.copy()

    total_rev    = float(df["revenue"].sum())
    total_cost   = float(df["cost"].sum())
    overall_margin = _safe_div(total_rev - total_cost, total_rev) * 100

    # Monthly margin trend
    df["_month"] = df["date"].dt.to_period("M").astype(str)
    monthly = (
        df.groupby("_month")
        .agg(revenue=("revenue", "sum"), cost=("cost", "sum"))
        .reset_index()
        .sort_values("_month")
        .rename(columns={"_month": "period"})
    )
    monthly["margin_pct"] = monthly.apply(
        lambda row: _safe_div(row["revenue"] - row["cost"], row["revenue"]) * 100,
        axis=1,
    )

    avg_margin = float(monthly["margin_pct"].mean())
    min_margin = float(monthly["margin_pct"].min())
    max_margin = float(monthly["margin_pct"].max())

    margin_change_pp = 0.0
    margin_compressed = False
    if len(monthly) >= 2:
        margin_change_pp  = float(monthly["margin_pct"].iloc[-1] - monthly["margin_pct"].iloc[0])
        margin_compressed = margin_change_pp < -5.0

    # Product profitability
    product_profit_df = None
    low_margin_products: list = []

    if _col(df, "product"):
        prod = (
            df.groupby("product")
            .agg(revenue=("revenue", "sum"), cost=("cost", "sum"))
            .reset_index()
        )
        prod["profit"]     = prod["revenue"] - prod["cost"]
        prod["margin_pct"] = prod.apply(
            lambda row: _safe_div(row["profit"], row["revenue"]) * 100, axis=1
        )
        prod = prod.sort_values("margin_pct").reset_index(drop=True)
        product_profit_df = prod

        avg_prod_margin = float(prod["margin_pct"].mean())
        threshold       = avg_prod_margin - 10.0
        low_margin_products = prod[prod["margin_pct"] < threshold]["product"].tolist()

    return {
        "available":           True,
        "overall_margin_pct":  overall_margin,
        "margin_trend_df":     monthly[["period", "revenue", "cost", "margin_pct"]],
        "product_profit_df":   product_profit_df,
        "low_margin_products": low_margin_products,
        "margin_compressed":   margin_compressed,
        "margin_change_pp":    margin_change_pp,
        "avg_margin_pct":      avg_margin,
        "min_margin_pct":      min_margin,
        "max_margin_pct":      max_margin,
    }


# ---------------------------------------------------------------------------
# Legacy API  (backward-compatible with app.py)
# ---------------------------------------------------------------------------

@dataclass
class KPISummary:
    """Dataclass consumed by app.py's render_kpis() and related helpers."""
    total_revenue:         float
    total_cost:            float
    gross_profit:          float
    gross_margin_pct:      float
    total_transactions:    int
    avg_transaction_value: float
    date_range_days:       int
    has_cost:              bool
    has_quantity:          bool
    revenue_mom_pct:       object   # float | None
    top_product:           object   # str | None
    top_region:            object   # str | None
    top_customer:          object   # str | None
    unique_products:       int
    unique_customers:      int
    monthly_revenue:       pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)


def compute_kpis(df: pd.DataFrame) -> KPISummary:
    """
    Legacy entry point used by app.py.
    Delegates to calculate_kpis / calculate_period_metrics.
    """
    kpis    = calculate_kpis(df)
    periods = calculate_period_metrics(df)

    # Build monthly_revenue DataFrame for the chart in app.py
    df2 = df.copy()
    df2["_month"] = df2["date"].dt.to_period("M")
    agg: dict = {"revenue": "sum"}
    if kpis["has_cost"]:
        agg["cost"] = "sum"
    monthly = (
        df2.groupby("_month")
        .agg(**{k: (k, v) for k, v in agg.items()})
        .reset_index()
        .rename(columns={"_month": "month"})
    )
    monthly["month_str"] = monthly["month"].astype(str)
    if kpis["has_cost"]:
        monthly["profit"] = monthly["revenue"] - monthly["cost"]
    monthly = monthly.sort_values("month").reset_index(drop=True)

    return KPISummary(
        total_revenue         = kpis["total_revenue"],
        total_cost            = kpis["total_cost"],
        gross_profit          = kpis["total_profit"],
        gross_margin_pct      = kpis["profit_margin_pct"],
        total_transactions    = kpis["total_orders"],
        avg_transaction_value = kpis["avg_order_value"],
        date_range_days       = kpis["date_range_days"],
        has_cost              = kpis["has_cost"],
        has_quantity          = kpis["has_quantity"],
        revenue_mom_pct       = periods["revenue_mom_pct"],
        top_product           = kpis["top_product"],
        top_region            = kpis["top_region"],
        top_customer          = kpis["top_customer"],
        unique_products       = kpis["unique_products"],
        unique_customers      = kpis["unique_customers"],
        monthly_revenue       = monthly,
    )


def monthly_revenue_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly revenue (and optionally profit) trend DataFrame for charting."""
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M").astype(str)
    agg: dict = {"revenue": "sum"}
    if _col(df, "cost"):
        agg["cost"] = "sum"
    out = df.groupby("month").agg(**{k: (k, v) for k, v in agg.items()}).reset_index()
    if "cost" in out.columns:
        out["profit"] = out["revenue"] - out["cost"]
    return out.sort_values("month").reset_index(drop=True)


def revenue_by_product(df: pd.DataFrame):
    """Revenue by product sorted descending.  None if column absent."""
    if not _col(df, "product"):
        return None
    return (
        df.groupby("product")["revenue"]
        .sum()
        .reset_index()
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )


def revenue_by_region(df: pd.DataFrame):
    """Revenue by region sorted descending.  None if column absent."""
    if not _col(df, "region"):
        return None
    return (
        df.groupby("region")["revenue"]
        .sum()
        .reset_index()
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )


def revenue_by_category(df: pd.DataFrame):
    """Revenue by category sorted descending.  None if column absent."""
    if "category" not in df.columns or not df["category"].notna().any():
        return None
    return (
        df.groupby("category")["revenue"]
        .sum()
        .reset_index()
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )


def margin_by_product(df: pd.DataFrame):
    """Margin % by product (ascending - worst first).  None if columns absent."""
    if not _col(df, "product") or not _col(df, "cost"):
        return None
    g = (
        df.groupby("product")
        .agg(revenue=("revenue", "sum"), cost=("cost", "sum"))
        .reset_index()
    )
    g["margin_pct"] = g.apply(
        lambda row: _safe_div(row["revenue"] - row["cost"], row["revenue"]) * 100, axis=1
    )
    return g.sort_values("margin_pct").reset_index(drop=True)


def customer_revenue_concentration(df: pd.DataFrame):
    """Customer revenue with cumulative concentration %.  None if column absent."""
    if not _col(df, "customer"):
        return None
    cust = (
        df.groupby("customer")["revenue"]
        .sum()
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    total = float(cust["revenue"].sum())
    cust["pct_of_total"]   = cust["revenue"].apply(lambda r: _safe_div(r, total) * 100)
    cust["cumulative_pct"] = cust["pct_of_total"].cumsum()
    return cust.reset_index(drop=True)
