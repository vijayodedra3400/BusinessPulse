# -*- coding: utf-8 -*-
"""
ai_engine.py  --  Gemini AI integration for BusinessPulse AI.
==============================================================
ARCHITECTURE PRINCIPLE:
  Python = Numerical Truth  |  Gemini = Explanation + Recommendations

STRICT CONTRACT:
  - Gemini NEVER calculates business metrics.
  - All numbers come from analytics.py / risk_detector.py.
  - Evidence is injected verbatim; Gemini only explains and recommends.
  - Application never crashes when Gemini is unavailable.

Environment Variables:
  GEMINI_API_KEY   -- Google AI Studio API key (required for live AI)
  GEMINI_MODEL     -- Model name (default: gemini-2.5-flash)
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
_MODEL   = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()

_PLACEHOLDER_KEYS = {"", "your_gemini_api_key_here", "YOUR_KEY_HERE", "GEMINI_API_KEY="}

_client          = None
_init_error      = None
_USER_KEY_ACTIVE = False


def _init():
    global _client, _init_error, _API_KEY, _MODEL, _USER_KEY_ACTIVE
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass

    if not _USER_KEY_ACTIVE:
        _API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
    _MODEL   = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()

    if not _API_KEY or _API_KEY in _PLACEHOLDER_KEYS:
        _init_error = "GEMINI_API_KEY not configured. Enter your key in the sidebar."
        _client = None
        return
    try:
        import google.genai as genai
        _client = genai.Client(api_key=_API_KEY)
        _init_error = None
        logger.info("Gemini client initialised. model=%s", _MODEL)
    except ImportError:
        _init_error = "google-genai package not installed. Run: pip install google-genai"
        _client = None
    except Exception as exc:
        _init_error = f"Gemini init failed: {exc}"
        _client = None
        logger.warning("Gemini init error: %s", exc)


_init()


def set_user_api_key(user_key: str) -> dict:
    """
    Configure or update a user-provided Gemini API key at runtime.
    Returns provider_status().
    """
    global _client, _init_error, _API_KEY, _USER_KEY_ACTIVE
    clean_key = (user_key or "").strip()

    if not clean_key or clean_key in _PLACEHOLDER_KEYS:
        _USER_KEY_ACTIVE = False
        _init()
        return provider_status()

    try:
        import google.genai as genai
        _client = genai.Client(api_key=clean_key)
        _API_KEY = clean_key
        _USER_KEY_ACTIVE = True
        _init_error = None
        logger.info("User-provided Gemini API key set successfully.")
    except Exception as exc:
        _init_error = f"Invalid API Key: {exc}"
        _client = None
        _USER_KEY_ACTIVE = False

    return provider_status()


def ai_available() -> bool:
    return _client is not None


def provider_status() -> dict:
    if _client:
        tag = " (User Key)" if _USER_KEY_ACTIVE else f" ({_MODEL})"
        return {"status": "live", "label": f"Gemini Active{tag}", "model": _MODEL, "is_user_key": _USER_KEY_ACTIVE}
    return {"status": "unavailable", "label": "Gemini Offline",
            "reason": _init_error or "Unknown", "model": _MODEL, "is_user_key": False}


_SYSTEM = """You are BusinessPulse AI, a business analytics explanation assistant.

You receive verified numerical findings from a deterministic Python analytics engine.
You are NOT the numerical calculation engine.

RULES:
- Never invent business metrics or percentages.
- Never modify supplied numbers.
- Never claim something is confirmed if evidence only suggests it.
- Distinguish: verified evidence vs likely contributors vs hypotheses.
- Use ONLY the supplied evidence.
- Be concise and business-oriented.

YOUR TASK:
1. Explain what happened (reference the exact numbers given).
2. Explain likely contributing factors (label as Possible or Likely).
3. Explain why it matters to the business.
4. Recommend 3 practical next actions.
5. Suggest 2-3 metrics to monitor going forward.

RESPONSE FORMAT - return valid JSON only, no markdown wrapper:
{
  "what_happened": "...",
  "likely_contributors": ["...", "..."],
  "why_it_matters": "...",
  "recommended_actions": ["...", "...", "..."],
  "monitor_next": ["...", "..."]
}"""

_EXECUTIVE_SYSTEM = """You are BusinessPulse AI generating an executive brief.

Rules:
- Use only the verified evidence provided.
- Never invent numbers.
- Be concise.
- Write for a senior executive.

Return valid JSON only, no markdown wrapper:
{
  "business_pulse": "2-3 sentence overall summary",
  "top_risks": ["risk 1", "risk 2", "risk 3"],
  "why_it_matters": "1-2 sentences on business impact",
  "top_3_actions": ["action 1", "action 2", "action 3"],
  "watch_next": ["metric 1", "metric 2", "metric 3"]
}"""


def _build_risk_prompt(risk_title, risk_category, severity, evidence):
    compact = {}
    for k, v in evidence.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            compact[k] = v[:5]
        else:
            compact[k] = v
    payload = {
        "risk": {"title": risk_title, "category": risk_category, "severity": severity},
        "verified_metrics": compact,
        "instruction": "Explain this risk using only the evidence above. Return JSON as specified."
    }
    return json.dumps(payload, indent=2, default=str)


def _kpi_val(kpis, key: str, default=0):
    if isinstance(kpis, dict):
        return kpis.get(key, default)
    return getattr(kpis, key, default)


def _build_executive_prompt(kpis, findings):
    safe_kpis = {
        "total_revenue":     round(float(_kpi_val(kpis, "total_revenue", 0) or 0), 2),
        "total_profit":      round(float(_kpi_val(kpis, "gross_profit", _kpi_val(kpis, "total_profit", 0)) or 0), 2),
        "profit_margin_pct": round(float(_kpi_val(kpis, "gross_margin_pct", _kpi_val(kpis, "profit_margin_pct", 0)) or 0), 1),
        "revenue_mom_pct":   _kpi_val(kpis, "revenue_mom_pct", None),
        "total_orders":      int(_kpi_val(kpis, "total_transactions", _kpi_val(kpis, "total_orders", 0)) or 0),
        "avg_order_value":   round(float(_kpi_val(kpis, "avg_transaction_value", _kpi_val(kpis, "avg_order_value", 0)) or 0), 2),
    }
    risk_list = [
        {"title": f.title, "severity": f.severity, "category": f.category, "summary": f.summary}
        for f in findings[:5]
    ]
    payload = {
        "verified_kpis": safe_kpis,
        "detected_risks": risk_list,
        "instruction": "Generate an executive brief using only the data provided. Return JSON."
    }
    return json.dumps(payload, indent=2, default=str)


class GeminiError(Exception):
    pass


def _call_gemini(prompt, system, max_tokens=1024):
    if not _client:
        raise GeminiError(_init_error or "Client not initialised.")
    
    import time
    import google.genai as genai
    import google.genai.types as gtypes

    last_exc = None
    for attempt in range(3):
        try:
            response = _client.models.generate_content(
                model=_MODEL,
                contents=prompt,
                config=gtypes.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    max_output_tokens=max_tokens,
                    temperature=0.3,
                )
            )
            if not response or not response.text:
                raise GeminiError("Empty response from Gemini.")
            return response.text.strip()
        except GeminiError:
            raise
        except Exception as exc:
            err = str(exc).lower()
            last_exc = exc
            if any(k in err for k in ["503", "unavailable", "high demand", "10053", "connection", "aborted", "reset", "socket"]):
                if attempt < 2:
                    time.sleep(1.2 * (attempt + 1))
                    continue
            if "api_key" in err or "unauthorized" in err or "authentication" in err or "403" in err:
                raise GeminiError("Invalid GEMINI_API_KEY. Check your .env file.")
            if "quota" in err or "rate" in err or "429" in err:
                raise GeminiError("Gemini quota exceeded. Please wait and try again.")
            if "not found" in err or "404" in err or "not exist" in err:
                raise GeminiError(
                    f"Model '{_MODEL}' not available. Set GEMINI_MODEL in .env to a valid model."
                )
            if "timeout" in err or "deadline" in err:
                raise GeminiError("Gemini request timed out. Check your connection.")
            raise GeminiError(f"Gemini error: {exc}")

    raise GeminiError(f"Gemini error: {last_exc}")


def _parse_json(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()
            if text.startswith("json"):
                text = text[4:].strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return None


_MATTER = {
    "Revenue": "Sustained revenue decline reduces operating cash flow and may limit investment capacity.",
    "Profitability": "Shrinking margins reduce profit per sale, undermining long-term financial health.",
    "Customer Risk": "Over-reliance on a few customers creates fragility -- losing one can cause major immediate impact.",
    "Product": "Declining product lines reduce revenue diversity and may signal competitive or demand issues.",
    "Geographic": "Regional concentration exposes the business to localised downturns.",
}

_ACTIONS = {
    "Revenue": ["Identify the exact month decline began and correlate with known events.",
                "Segment the decline by product and region to find the primary driver.",
                "Review the sales pipeline for lost or stalled deals."],
    "Profitability": ["Audit costs per product line for unexpected increases.",
                      "Review supplier pricing and renegotiate where possible.",
                      "Consider price adjustments on consistently low-margin products."],
    "Customer Risk": ["Actively prospect 3-5 new customers in the next 30 days.",
                      "Strengthen contracts with your top customers to reduce churn risk.",
                      "Set a diversification target: no single customer above 20%."],
    "Product": ["Investigate the root cause: quality, competition, or demand shift.",
                "Survey customers who reduced or stopped purchases.",
                "Reallocate sales resources toward growing product lines."],
    "Geographic": ["Identify 1-2 new target regions for expansion.",
                   "Investigate why underperforming regions are behind.",
                   "Develop region-specific sales strategies."],
}

_MONITOR = {
    "Revenue": ["Monthly revenue trend", "Order volume", "Average order value"],
    "Profitability": ["Gross margin %", "Cost per unit by product", "Revenue mix"],
    "Customer Risk": ["New customer acquisition rate", "Top customer order frequency", "Customer count"],
    "Product": ["Product A revenue trend", "Product A order count", "Comparable product performance"],
    "Geographic": ["Revenue by region", "Order count by region", "Average order value by region"],
}

_DEF_ACTIONS = ["Review the full evidence below.", "Assign an owner for this risk.",
                "Schedule a 30-day review checkpoint."]
_DEF_MONITOR = ["Revenue trend", "Gross margin %", "Order volume"]


def _fallback_risk(risk_title, risk_category, severity, evidence, error_note=""):
    ev_bullets = []
    for k, v in evidence.items():
        if isinstance(v, (int, float, str)):
            label = k.replace("_", " ").title()
            ev_bullets.append(f"{label}: {v:,.2f}" if isinstance(v, float) else f"{label}: {v}")
    what = (f"{risk_title} has been detected (Severity: {severity}). "
            "All figures are Python-computed from your uploaded data.")
    if ev_bullets:
        what += " Key evidence: " + "; ".join(ev_bullets[:3]) + "."
    result = {
        "what_happened": what,
        "likely_contributors": [
            "Possible: Underlying market or operational changes during the period.",
            "Possible: Segment concentration that experienced disruption.",
        ],
        "why_it_matters": _MATTER.get(risk_category, "This risk warrants prompt investigation."),
        "recommended_actions": _ACTIONS.get(risk_category, _DEF_ACTIONS)[:3],
        "monitor_next": _MONITOR.get(risk_category, _DEF_MONITOR)[:3],
        "_source": "fallback",
    }
    if error_note:
        result["_error"] = error_note
    return result


def _fallback_executive(kpis, findings):
    n = len(findings)
    top3 = [f"{f.severity}: {f.title}" for f in findings[:3]]
    rev = float(_kpi_val(kpis, "total_revenue", 0) or 0)
    margin = float(_kpi_val(kpis, "gross_margin_pct", _kpi_val(kpis, "profit_margin_pct", 0)) or 0)
    rev_str = f"${rev/1e6:.1f}M" if rev >= 1e6 else f"${rev/1e3:.0f}K" if rev >= 1e3 else f"${rev:,.0f}"
    return {
        "business_pulse": (
            f"BusinessPulse analysis detected {n} risk(s) across your sales data. "
            f"Total revenue is {rev_str} with a gross margin of {margin:.1f}%. "
            "Immediate attention is recommended on the highest-severity findings."
        ),
        "top_risks": top3 or ["No significant risks detected."],
        "why_it_matters": (
            "Unaddressed business risks compound over time. "
            "Early identification significantly improves outcomes."
        ),
        "top_3_actions": [
            "Investigate the highest-severity risk immediately.",
            "Assign a clear owner for each risk finding.",
            "Schedule a weekly monitoring cadence for key metrics.",
        ],
        "watch_next": ["Revenue trend", "Gross margin %", "Order volume"],
        "_source": "fallback",
    }


def get_risk_explanation(risk_title: str, risk_category: str,
                         severity: str, evidence: dict,
                         risk_id: str = "") -> dict:
    """
    Return a structured dict explaining a detected risk.
    Keys: what_happened, likely_contributors, why_it_matters,
          recommended_actions, monitor_next, _source
    Never raises. Falls back gracefully.
    """
    if not _client:
        return _fallback_risk(risk_title, risk_category, severity, evidence,
                              error_note=_init_error or "Gemini not configured.")
    try:
        prompt = _build_risk_prompt(risk_title, risk_category, severity, evidence)
        raw    = _call_gemini(prompt, _SYSTEM, max_tokens=2048)
        parsed = _parse_json(raw)
        if parsed and "what_happened" in parsed:
            parsed["_source"] = "gemini"
            for key in ("likely_contributors", "why_it_matters", "recommended_actions", "monitor_next"):
                if key not in parsed:
                    parsed[key] = []
            return parsed
        return _fallback_risk(risk_title, risk_category, severity, evidence, error_note="JSON parse warning")
    except GeminiError as exc:
        logger.warning("Gemini failed: %s", exc)
        return _fallback_risk(risk_title, risk_category, severity, evidence, error_note=str(exc))
    except Exception as exc:
        logger.exception("Unexpected Gemini error")
        return _fallback_risk(risk_title, risk_category, severity, evidence,
                              error_note=f"Unexpected error: {exc}")


def get_executive_brief(kpis: dict, findings: list) -> dict:
    """
    Generate an executive brief. Never raises.
    """
    if not findings:
        return {
            "business_pulse": "No significant risks detected. Business metrics are within normal ranges.",
            "top_risks": [],
            "why_it_matters": "Continue monitoring to maintain healthy performance.",
            "top_3_actions": ["Maintain current strategy.", "Review data monthly.", "Track KPIs weekly."],
            "watch_next": ["Revenue trend", "Gross margin", "Order volume"],
            "_source": "fallback",
        }
    if not _client:
        return _fallback_executive(kpis, findings)
    try:
        prompt = _build_executive_prompt(kpis, findings)
        raw    = _call_gemini(prompt, _EXECUTIVE_SYSTEM, max_tokens=2048)
        parsed = _parse_json(raw)
        if parsed and "business_pulse" in parsed:
            parsed["_source"] = "gemini"
            for key in ("top_risks", "why_it_matters", "top_3_actions", "watch_next"):
                if key not in parsed:
                    parsed[key] = []
            return parsed
        return _fallback_executive(kpis, findings)
    except Exception as exc:
        logger.warning("Executive brief failed: %s", exc)
        return _fallback_executive(kpis, findings)
