"""
app.py  --  BusinessPulse AI  |  Intelligent Business Risk Detector
====================================================================
Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analytics as an
import ai_engine as ai
import risk_detector as rd

# -----------------------------------------------------------------------------
# Page Config & Global CSS
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="BusinessPulse AI - Risk Detector",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* -- Base -- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }

    /* -- Header Band -- */
    .bp-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 60%, #1e1b4b 100%);
        border-bottom: 1px solid #334155;
        padding: 1.75rem 2rem 1.25rem;
        margin: -3rem -3rem 1.5rem -3rem;
        border-radius: 0 0 16px 16px;
    }
    .bp-logo-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 6px;
    }
    .bp-title-group {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .bp-logo-icon {
        font-size: 2.2rem;
        line-height: 1;
    }
    .bp-product-name {
        font-size: 1.85rem;
        font-weight: 800;
        background: linear-gradient(90deg, #818cf8, #c084fc, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .bp-tagline {
        font-size: 0.92rem;
        color: #94a3b8;
        font-weight: 400;
    }
    .bp-ai-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(129, 140, 248, 0.12);
        border: 1px solid rgba(129, 140, 248, 0.35);
        color: #a5b4fc;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        padding: 4px 12px;
        border-radius: 20px;
    }

    /* -- KPI Cards -- */
    .kpi-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.15rem 1.25rem;
        text-align: left;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        border-color: #475569;
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 0.78rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.2;
    }
    .kpi-delta {
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 0.4rem;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .kpi-delta.up { color: #34d399; }
    .kpi-delta.down { color: #f87171; }
    .kpi-delta.flat { color: #94a3b8; }

    /* -- Risk Cards -- */
    .risk-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-left: 5px solid var(--severity-color, #818cf8);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .risk-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    .risk-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .severity-badge {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        padding: 3px 9px;
        border-radius: 12px;
        background: var(--badge-bg, rgba(239,68,68,0.15));
        color: var(--badge-color, #ef4444);
        border: 1px solid var(--badge-border, rgba(239,68,68,0.35));
    }
    .risk-category {
        font-size: 0.75rem;
        color: #94a3b8;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .risk-summary {
        font-size: 0.9rem;
        color: #cbd5e1;
        line-height: 1.45;
    }

    /* -- AI Explanation Box -- */
    .ai-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(129, 140, 248, 0.3);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-top: 1rem;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.05);
    }
    .ai-box-header {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.95rem;
        font-weight: 700;
        color: #c084fc;
        margin-bottom: 0.75rem;
    }
    .ai-section-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #a5b4fc;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.85rem;
        margin-bottom: 0.35rem;
    }
    .ai-text {
        font-size: 0.88rem;
        color: #e2e8f0;
        line-height: 1.55;
    }
    .ai-bullet {
        font-size: 0.86rem;
        color: #cbd5e1;
        margin-bottom: 0.25rem;
        padding-left: 0.5rem;
    }

    /* -- Evidence Box -- */
    .evidence-box {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.85rem;
        font-family: monospace;
        font-size: 0.82rem;
        color: #38bdf8;
        line-height: 1.6;
    }

    /* -- Section Title -- */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 1rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-divider {
        border: 0;
        height: 1px;
        background: #334155;
        margin: 1.5rem 0;
    }

    /* -- Upload Zone -- */
    .upload-zone {
        background: #1e293b;
        border: 2px dashed #475569;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        margin-top: 1rem;
    }

    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Plotly Dark Theme
# -----------------------------------------------------------------------------

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94a3b8"),
    xaxis=dict(gridcolor="#1e293b", linecolor="#334155", tickfont=dict(color="#94a3b8")),
    yaxis=dict(gridcolor="#1e293b", linecolor="#334155", tickfont=dict(color="#94a3b8")),
    margin=dict(l=20, r=20, t=40, b=20),
)

PLOTLY_COLORS = [
    "#818cf8", "#c084fc", "#34d399", "#fb923c",
    "#38bdf8", "#f472b6", "#a3e635", "#facc15",
]

def _fmt_curr(val: float) -> str:
    if abs(val) >= 1_000_000:
        return f"${val/1_000_000:.1f}M"
    if abs(val) >= 1_000:
        return f"${val/1_000:.1f}K"
    return f"${val:,.0f}"

def _severity_style(severity: str) -> dict[str, str]:
    colors = {
        "Critical": ("#ef4444", "rgba(239,68,68,0.15)", "rgba(239,68,68,0.35)"),
        "High":     ("#f97316", "rgba(249,115,22,0.15)", "rgba(249,115,22,0.35)"),
        "Medium":   ("#eab308", "rgba(234,179,8,0.15)", "rgba(234,179,8,0.35)"),
        "Low":      ("#22c55e", "rgba(34,197,94,0.15)", "rgba(34,197,94,0.35)"),
    }
    c, bg, border = colors.get(severity, colors["Low"])
    return {"color": c, "bg": bg, "border": border}

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------

def render_header():
    ps = ai.provider_status()
    if ps["status"] == "live":
        badge_label = f"✦ AI Active ({ps['model']})"
    elif ps["status"] == "mock":
        badge_label = "✦ AI Mock Mode"
    else:
        badge_label = "✦ AI Offline"

    st.markdown(
        f"""
        <div class="bp-header">
            <div class="bp-logo-row">
                <div class="bp-title-group">
                    <span class="bp-logo-icon">📊</span>
                    <div>
                        <span class="bp-product-name">BusinessPulse AI</span>
                        <div class="bp-tagline">Intelligent Business Risk Detector — Python Truth + Gemini AI Intelligence</div>
                    </div>
                </div>
                <span class="bp-ai-badge">{badge_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# KPI Overview Row
# -----------------------------------------------------------------------------

def render_kpis(kpi: an.KPISummary):
    st.markdown('<div class="section-title">📈 Business Performance Overview</div>', unsafe_allow_html=True)
    cols = st.columns(4)

    # 1. Total Revenue
    with cols[0]:
        delta_html = ""
        if kpi.revenue_mom_pct is not None:
            cls = "up" if kpi.revenue_mom_pct >= 0 else "down"
            sign = "▲" if kpi.revenue_mom_pct >= 0 else "▼"
            delta_html = f'<div class="kpi-delta {cls}">{sign} {abs(kpi.revenue_mom_pct):.1f}% vs prior period</div>'
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Revenue</div>
                <div class="kpi-value">{_fmt_curr(kpi.total_revenue)}</div>
                {delta_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Gross Profit / Transactions
    with cols[1]:
        if kpi.has_cost:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Gross Profit</div>
                    <div class="kpi-value">{_fmt_curr(kpi.gross_profit)}</div>
                    <div class="kpi-delta flat">Gross Profit Margin: {kpi.gross_margin_pct:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Total Orders</div>
                    <div class="kpi-value">{kpi.total_transactions:,}</div>
                    <div class="kpi-delta flat">{kpi.date_range_days} days span</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 3. Average Order Value
    with cols[2]:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Avg Transaction Value</div>
                <div class="kpi-value">{_fmt_curr(kpi.avg_transaction_value)}</div>
                <div class="kpi-delta flat">{kpi.total_transactions:,} total orders</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4. Gross Margin Status / Top Product
    with cols[3]:
        if kpi.has_cost:
            cls = "up" if kpi.gross_margin_pct >= 30 else ("flat" if kpi.gross_margin_pct >= 15 else "down")
            sign = "✓" if kpi.gross_margin_pct >= 30 else ("~" if kpi.gross_margin_pct >= 15 else "⚠")
            status = "Healthy" if kpi.gross_margin_pct >= 30 else ("Moderate" if kpi.gross_margin_pct >= 15 else "Low Margin")
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Gross Margin %</div>
                    <div class="kpi-value">{kpi.gross_margin_pct:.1f}%</div>
                    <div class="kpi-delta {cls}">{sign} {status}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            top = kpi.top_product or "N/A"
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Top Product</div>
                    <div class="kpi-value" style="font-size:1.1rem">{top}</div>
                    <div class="kpi-delta flat">{kpi.unique_products} products</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# -----------------------------------------------------------------------------
# Overview Charts
# -----------------------------------------------------------------------------

def render_overview_charts(df: pd.DataFrame):
    st.markdown('<div class="section-title">📊 Analytics Breakdown</div>', unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["Monthly Revenue Trend", "Product Breakdown", "Regional Breakdown", "Category Breakdown"])

    with t1:
        monthly = an.monthly_revenue_trend(df)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["month"], y=monthly["revenue"],
            name="Revenue", line=dict(color="#818cf8", width=3),
            fill="tozeroy", fillcolor="rgba(129,140,248,0.1)",
            mode="lines+markers", marker=dict(size=6, color="#818cf8"),
        ))
        if "profit" in monthly.columns:
            fig.add_trace(go.Scatter(
                x=monthly["month"], y=monthly["profit"],
                name="Gross Profit", line=dict(color="#34d399", width=2.5, dash="dash"),
                mode="lines+markers", marker=dict(size=5, color="#34d399"),
            ))
        fig.update_layout(title="Monthly Revenue & Profit Trajectory", **PLOTLY_LAYOUT, height=330,
                          legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        prod_df = an.revenue_by_product(df)
        if prod_df is not None:
            fig = px.bar(prod_df, x="product", y="revenue",
                         color="revenue", color_continuous_scale="Purples",
                         labels={"revenue": "Revenue ($)", "product": "Product"})
            fig.update_layout(title="Revenue Contribution by Product", **PLOTLY_LAYOUT, height=330,
                               coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No 'product' column available in dataset.")

    with t3:
        reg_df = an.revenue_by_region(df)
        if reg_df is not None:
            fig = px.pie(reg_df, names="region", values="revenue",
                         color_discrete_sequence=PLOTLY_COLORS, hole=0.45)
            fig.update_layout(title="Revenue Distribution by Region", **PLOTLY_LAYOUT, height=330,
                               legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")))
            fig.update_traces(textposition="outside", textinfo="percent+label", textfont=dict(color="#94a3b8"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No 'region' column available in dataset.")

    with t4:
        cat_df = an.revenue_by_category(df)
        if cat_df is not None:
            fig = px.bar(cat_df, x="revenue", y="category", orientation="h",
                         color="revenue", color_continuous_scale="Viridis",
                         labels={"revenue": "Revenue ($)", "category": "Category"})
            fig.update_layout(title="Revenue by Category", **PLOTLY_LAYOUT, height=330, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No 'category' column available in dataset.")

# -----------------------------------------------------------------------------
# Risk Intelligence & AI Explanations
# -----------------------------------------------------------------------------

def render_risk_section(findings: list[rd.RiskFinding], kpi: an.KPISummary):
    st.markdown('<div class="section-title">⚠️ Risk Intelligence Engine</div>', unsafe_allow_html=True)

    counts = {s: sum(1 for f in findings if f.severity == s) for s in rd.SEVERITY_LEVELS}
    total = len(findings)

    if total == 0:
        status_label = "✅ Healthy - Zero Risks Detected"
        status_color = "#34d399"
    elif counts["Critical"] > 0:
        status_label = "🔴 Critical Risks Detected"
        status_color = "#ef4444"
    elif counts["High"] > 0:
        status_label = "🟠 High Risks Detected"
        status_color = "#f97316"
    elif counts["Medium"] > 0:
        status_label = "🟡 Medium Risks Detected"
        status_color = "#eab308"
    else:
        status_label = "🟢 Low Risk Level"
        status_color = "#22c55e"

    summary_cols = st.columns([3, 1, 1, 1, 1])
    with summary_cols[0]:
        st.markdown(
            f"""
            <div style="padding:0.5rem 0;">
                <span style="font-size:1.1rem;font-weight:700;color:{status_color};">{status_label}</span>
                <div style="color:#94a3b8;font-size:0.85rem;margin-top:2px;">{total} total risk issue{'s' if total!=1 else ''} flagged by Python rule engine</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for i, sev in enumerate(rd.SEVERITY_LEVELS):
        css = _severity_style(sev)
        with summary_cols[i + 1]:
            st.markdown(
                f"""
                <div style="text-align:center;padding:0.5rem;background:#1e293b;border-radius:8px;border:1px solid #334155;">
                    <div style="font-size:1.3rem;font-weight:800;color:{css['color']};">{counts[sev]}</div>
                    <div style="font-size:0.68rem;color:#94a3b8;font-weight:700;text-transform:uppercase;">{sev}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)

    # Executive Briefing Accordion / Button
    with st.expander("⚡ AI Executive Briefing (Click to Expand)", expanded=False):
        if st.button("Generate Executive Briefing with Gemini AI", key="btn_exec_brief"):
            with st.spinner("Analyzing all metrics & risks with Gemini AI..."):
                brief = ai.get_executive_brief(kpi, findings)
                st.session_state["exec_brief_res"] = brief

        if "exec_brief_res" in st.session_state and st.session_state["exec_brief_res"]:
            b = st.session_state["exec_brief_res"]
            st.markdown(
                f"""
                <div class="ai-box">
                    <div class="ai-box-header">🤖 Executive Briefing ({b.get('_source', 'ai')})</div>
                    <div class="ai-text"><strong>Business Status:</strong> {b.get('business_pulse', '')}</div>
                    <div class="ai-section-title">Why It Matters</div>
                    <div class="ai-text">{b.get('why_it_matters', '')}</div>
                    <div class="ai-section-title">Top Recommended Actions</div>
                    {''.join([f'<div class="ai-bullet">• {act}</div>' for act in b.get('top_3_actions', [])])}
                    <div class="ai-section-title">Metrics to Watch Next</div>
                    {''.join([f'<div class="ai-bullet">👁 {w}</div>' for w in b.get('watch_next', [])])}
                </div>
                """,
                unsafe_allow_html=True,
            )

    if not findings:
        st.success("🎉 No business risks detected in the current view!")
        return

    # Render each risk finding card
    for idx, finding in enumerate(findings):
        css = _severity_style(finding.severity)
        st.markdown(
            f"""
            <div class="risk-card" style="--severity-color:{css['color']}">
                <div class="risk-card-header">
                    <span class="risk-title">{finding.title}</span>
                    <span class="severity-badge" style="--badge-bg:{css['bg']};--badge-color:{css['color']};--badge-border:{css['border']}">
                        {finding.severity}
                    </span>
                </div>
                <div class="risk-category">📁 {finding.category}</div>
                <div class="risk-summary">{finding.summary}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander(f"🔍 Investigate Risk & AI Recommendations ({finding.title})", expanded=False):
            ev_col, chart_col = st.columns([1, 1])

            with ev_col:
                st.markdown("**📋 Deterministic Evidence (Python Verified)**")
                lines = []
                for k, v in finding.evidence.items():
                    if isinstance(v, (int, float, str)):
                        lbl = k.replace("_", " ").title()
                        val_str = f"{v:,.2f}" if isinstance(v, float) else (f"{v:,}" if isinstance(v, int) else str(v))
                        lines.append(f"{lbl}: {val_str}")
                if lines:
                    st.markdown(f'<div class="evidence-box">' + "<br>".join(lines) + "</div>", unsafe_allow_html=True)

                for k, v in finding.evidence.items():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                        st.markdown(f"**{k.replace('_', ' ').title()}:**")
                        ev_df = pd.DataFrame(v)
                        st.dataframe(ev_df, use_container_width=True, hide_index=True)

            with chart_col:
                if finding.chart_data is not None and not finding.chart_data.empty:
                    st.markdown(f"**📊 {finding.chart_title}**")
                    cdata = finding.chart_data.copy()
                    cols_list = list(cdata.columns)
                    if finding.chart_type == "line" and len(cols_list) >= 2:
                        fig = px.line(cdata, x=cols_list[0], y=cols_list[1], markers=True, color_discrete_sequence=[css["color"]])
                        fig.update_layout(**PLOTLY_LAYOUT, height=250)
                        st.plotly_chart(fig, use_container_width=True)
                    elif finding.chart_type == "bar" and len(cols_list) >= 2:
                        fig = px.bar(cdata, x=cols_list[0], y=cols_list[1], color_discrete_sequence=[css["color"]])
                        fig.update_layout(**PLOTLY_LAYOUT, height=250)
                        st.plotly_chart(fig, use_container_width=True)

            # AI Explanation Button & Content
            st.markdown("---")
            ai_state_key = f"ai_exp_{finding.risk_id}_{idx}"

            if st.button(f"Generate Gemini Explanation for {finding.title}", key=f"btn_ai_{idx}"):
                with st.spinner("Generating AI explanation..."):
                    res = ai.get_risk_explanation(
                        risk_title=finding.title,
                        risk_category=finding.category,
                        severity=finding.severity,
                        evidence=finding.evidence,
                        risk_id=finding.risk_id,
                    )
                    st.session_state[ai_state_key] = res

            if ai_state_key in st.session_state and st.session_state[ai_state_key]:
                exp = st.session_state[ai_state_key]
                st.markdown(
                    f"""
                    <div class="ai-box">
                        <div class="ai-box-header">🤖 AI Root Cause & Recommendation ({exp.get('_source', 'ai')})</div>
                        <div class="ai-text"><strong>What Happened:</strong> {exp.get('what_happened', '')}</div>
                        <div class="ai-section-title">Likely Contributing Factors</div>
                        {''.join([f'<div class="ai-bullet">• {f}</div>' for f in exp.get('likely_contributors', [])])}
                        <div class="ai-section-title">Why It Matters</div>
                        <div class="ai-text">{exp.get('why_it_matters', '')}</div>
                        <div class="ai-section-title">Recommended Actions</div>
                        {''.join([f'<div class="ai-bullet">✅ {a}</div>' for a in exp.get('recommended_actions', [])])}
                        <div class="ai-section-title">Metrics to Monitor</div>
                        {''.join([f'<div class="ai-bullet">📈 {m}</div>' for m in exp.get('monitor_next', [])])}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# -----------------------------------------------------------------------------
# Deep Dive Section (Customers & Profitability)
# -----------------------------------------------------------------------------

def render_deep_dive(df: pd.DataFrame):
    st.markdown('<div class="section-title">👥 Customer & Margin Concentration Deep-Dive</div>', unsafe_allow_html=True)
    d1, d2 = st.tabs(["Top Customers (Pareto)", "Product Margin Analysis"])

    with d1:
        cust_res = an.calculate_customer_metrics(df)
        cust_df  = cust_res.get("customer_revenue_df") if isinstance(cust_res, dict) else cust_res

        if cust_df is not None and not cust_df.empty:
            top10 = cust_df.head(10)
            fig = px.bar(top10, x="customer", y="revenue",
                         title="Top 10 Customers by Revenue",
                         labels={"revenue": "Revenue ($)", "customer": "Customer"},
                         color="revenue", color_continuous_scale="Tealgrn")
            fig.update_layout(**PLOTLY_LAYOUT, height=330, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Top 1 Customer Share", f"{cust_res.get('top_customer_pct', 0):.1f}%")
            with col_b:
                st.metric("Top 3 Customers Share", f"{cust_res.get('top3_customer_pct', 0):.1f}%")

            if isinstance(cust_res, dict) and cust_res.get("has_inactive") and cust_res.get("inactive_customers"):
                inact = cust_res["inactive_customers"]
                st.warning(f"⚠️ **Inactive Customers Signal**: {len(inact)} customer(s) with no orders in past 3 months: {', '.join(inact[:5])}")
        else:
            st.info("No 'customer' column available in dataset.")

    with d2:
        margin_df = an.margin_by_product(df)
        if margin_df is not None and not margin_df.empty:
            fig = px.bar(margin_df, x="product", y="margin_pct",
                         title="Gross Margin % by Product",
                         labels={"margin_pct": "Gross Margin %", "product": "Product"},
                         color="margin_pct", color_continuous_scale="RdYlGn")
            fig.update_layout(**PLOTLY_LAYOUT, height=330, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(margin_df, use_container_width=True, hide_index=True)
        else:
            st.info("No 'cost' or 'product' column available for margin analysis.")

# -----------------------------------------------------------------------------
# Data Preview & Validation
# -----------------------------------------------------------------------------

def render_data_preview(df: pd.DataFrame, errors: list, warnings: list):
    with st.expander("📋 Dataset Validation & Raw Rows", expanded=False):
        if errors:
            for e in errors:
                st.error(f"❌ {e}")
        if warnings:
            for w in warnings:
                st.warning(f"⚠️ {w}")
        if not errors:
            st.success(f"✅ {len(df):,} valid transactions loaded into memory.")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Detected Columns:**")
                for c in df.columns:
                    st.markdown(f"  • `{c}`")
            with c2:
                st.markdown("**Date Bounds:**")
                st.markdown(f"  {df['date'].min().date()}  ➡  {df['date'].max().date()}")
            st.dataframe(df.head(15), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------

def main():
    render_header()

    # Session State Initialization
    if "df_raw" not in st.session_state:
        st.session_state.df_raw = None

    # Sidebar Data Controls & Filters
    with st.sidebar:
        st.markdown("### ⚙️ Engine Controls")
        ps = ai.provider_status()
        if ps["status"] == "live":
            st.success(f"Gemini API Live ({ps['model']})")
        elif ps["status"] == "mock":
            st.info("AI Mock Mode Active")
        else:
            st.warning("AI Offline (Missing Key)")

        st.markdown("---")
        st.markdown("### 📂 Data Ingestion")

        uploaded = st.file_uploader(
            "Upload Sales CSV",
            type=["csv"],
            help="CSV containing columns: date, revenue, cost, product, region, customer, category",
        )

        load_sample = st.button("📊 Load Demo Dataset", use_container_width=True)

        if uploaded is not None:
            st.session_state.df_raw = pd.read_csv(uploaded)
        elif load_sample:
            try:
                st.session_state.df_raw = pd.read_csv("sample_data.csv")
                st.success("Loaded sample_data.csv")
            except Exception as e:
                st.error(f"Failed to load sample_data.csv: {e}")

    # Process Data & Filters
    if st.session_state.df_raw is not None:
        df_clean, errors, warnings = an.load_and_validate(st.session_state.df_raw)

        if errors:
            st.error(f"Dataset Validation Failed: {errors}")
            st.stop()

        # Dynamic Filters in Sidebar
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 🎯 Dynamic Filters")

            # Date filter
            min_date = df_clean["date"].min().date()
            max_date = df_clean["date"].max().date()
            date_range = st.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

            # Product Filter
            selected_products = None
            if "product" in df_clean.columns:
                all_prods = sorted(df_clean["product"].dropna().unique().tolist())
                selected_products = st.multiselect("Products", all_prods, default=all_prods)

            # Region Filter
            selected_regions = None
            if "region" in df_clean.columns:
                all_regs = sorted(df_clean["region"].dropna().unique().tolist())
                selected_regions = st.multiselect("Regions", all_regs, default=all_regs)

            # Category Filter
            selected_categories = None
            if "category" in df_clean.columns:
                all_cats = sorted(df_clean["category"].dropna().unique().tolist())
                selected_categories = st.multiselect("Categories", all_cats, default=all_cats)

        # Apply Filters
        df_filtered = df_clean.copy()
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            df_filtered = df_filtered[(df_filtered["date"] >= start_d) & (df_filtered["date"] <= end_d)]

        if selected_products is not None and "product" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["product"].isin(selected_products)]

        if selected_regions is not None and "region" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["region"].isin(selected_regions)]

        if selected_categories is not None and "category" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["category"].isin(selected_categories)]

        if df_filtered.empty:
            st.warning("No data matching the selected filters.")
            st.stop()

        # Compute Metrics & Risks on Filtered Data
        kpi = an.compute_kpis(df_filtered)
        findings = rd.detect_all_risks(df_filtered)

        # Dashboard Layout
        render_data_preview(df_filtered, errors, warnings)
        st.markdown("<br>", unsafe_allow_html=True)

        render_kpis(kpi)
        st.markdown("<br>", unsafe_allow_html=True)

        render_overview_charts(df_filtered)
        st.markdown("<br>", unsafe_allow_html=True)

        render_risk_section(findings, kpi)
        st.markdown("<br>", unsafe_allow_html=True)

        render_deep_dive(df_filtered)

        # Footer
        st.markdown("<br><br><hr class='section-divider'/>", unsafe_allow_html=True)
        st.markdown(
            '<div style="text-align:center;color:#64748b;font-size:0.8rem;">'
            'BusinessPulse AI Hackathon Prototype · Deterministic Analytics (Pandas/NumPy) + Generative AI (Gemini)'
            '</div>',
            unsafe_allow_html=True,
        )

    else:
        # Welcome Landing Screen
        st.markdown(
            """
            <div class="upload-zone">
                <div style="font-size:3.5rem;margin-bottom:0.75rem;">📊</div>
                <div style="font-size:1.3rem;font-weight:700;color:#f8fafc;margin-bottom:0.5rem;">
                    Upload Your Sales Data to Begin Intelligence Analysis
                </div>
                <div style="color:#94a3b8;font-size:0.92rem;margin-bottom:1.5rem;">
                    Drag and drop a CSV in the sidebar or click "Load Demo Dataset" for instant risk detection.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        feat_cols = st.columns(3)
        features = [
            ("⚡ 100% Deterministic Truth", "Metrics & risk triggers are computed entirely by Python rule engines. No LLM math hallucinations."),
            ("🤖 Gemini AI Explanations", "Gemini interprets detected risks in plain business language and recommends concrete mitigation steps."),
            ("📊 Interactive Visuals", "Plotly trend lines, regional maps, Pareto concentration charts, and dynamic date/segment filtering."),
        ]
        for col, (title, desc) in zip(feat_cols, features):
            with col:
                st.markdown(
                    f"""
                    <div class="kpi-card" style="padding:1.5rem;height:100%;">
                        <div style="font-size:1.05rem;font-weight:700;color:#818cf8;margin-bottom:0.5rem;">{title}</div>
                        <div style="font-size:0.85rem;color:#94a3b8;line-height:1.5;">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

if __name__ == "__main__":
    main()
