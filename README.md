# BusinessPulse AI 📊
## Intelligent Business Risk Detector

> **Hackathon Prototype** | Architectural Paradigm: **Deterministic Python Truth + Generative LLM Explanation Layer**

BusinessPulse AI is an intelligent business risk detection platform designed for modern SMBs, finance teams, and executive leaders. It automatically ingests sales datasets, runs a suite of deterministic rule engines to identify operational and financial risks with 100% numerical precision, and leverages **Google Gemini AI** to produce natural-language explanations, root-cause insights, and recommended action plans.

---

## 🌟 Key Features

* **⚡ 100% Deterministic Numerical Truth**: All KPIs, trends, percentages, variance calculations, and risk thresholds are computed exclusively by Python standard libraries (`pandas`, `numpy`). The LLM is **never** asked to perform math, eliminating hallucinations.
* **⚠️ 7 Algorithmic Risk Detectors**:
  1. **Overall Revenue Decline**: Detects significant H2 vs. H1 drops or MoM contraction.
  2. **Product Line Contraction**: Identifies specific declining products dragging down revenue.
  3. **Low-Margin / Unprofitable Lines**: Flags items selling near or below cost margin thresholds.
  4. **Margin Compression**: Triggers when gross margin drops over time.
  5. **Customer Concentration Risk**: Flags over-reliance on top customers (Pareto analysis).
  6. **Regional Imbalance**: Identifies underperforming geographic sales regions.
  7. **Revenue Volatility Anomaly**: Detects irregular spikes or drops in daily/monthly transactions.
* **🤖 Gemini AI Explanation Engine**: Converts compact JSON risk evidence into executive-ready business explanations, root cause hypotheses, and step-by-step mitigation plans.
* **🎯 Dynamic Segment Filtering**: Instantly recalculates all metrics and risk findings across date bounds, products, categories, and geographic regions.
* **⚡ Fallback & Offline Resilience**: Operates seamlessly in offline mode (`AI_MOCK_MODE=1` or missing API keys) with pre-formatted deterministic summaries so demo environments never fail.
* **📊 Modern Dark-Mode Dashboard**: Built with Streamlit and Plotly for visual presentation.

---

## 🏗️ System Architecture

```
+-------------------------------------------------------------------+
|                        User Sales Data (CSV)                       |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|               1. Analytics Engine (analytics.py)                  |
|    - Normalizes & validates schema                                |
|    - Computes KPIs: Revenue, Profit, Margin %, AOV, MoM %         |
|    - Computes segment metrics (Product, Region, Customer, Date)  |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|              2. Risk Detector (risk_detector.py)                  |
|    - Executes 7 deterministic rule algorithms                     |
|    - Generates RiskFinding objects with verified evidence        |
|    - Classifies severity: CRITICAL | HIGH | MEDIUM | LOW          |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|               3. AI Engine (ai_engine.py - Gemini API)            |
|    - Receives compact structured JSON payload (NO raw CSVs)       |
|    - System Prompt enforces strict adherence to Python truth      |
|    - Returns JSON: what_happened, contributors, actions           |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                 4. Streamlit Dashboard (app.py)                   |
|    - Interactive KPI cards, Plotly charts, dynamic filters       |
|    - One-click AI Executive Briefings & Risk Explanations         |
+-------------------------------------------------------------------+
```

---

## 📁 Repository Structure

```
businesspulse/
├── app.py                # Main Streamlit dashboard application UI
├── analytics.py          # Deterministic pandas/numpy business metric calculation engine
├── risk_detector.py      # Rule-based business risk detection algorithms
├── ai_engine.py          # Google Gemini API integration & fallback explanation layer
├── sample_data.csv       # Synthetic 900+ row sales dataset with embedded risk patterns
├── requirements.txt      # Project dependencies
├── .env.example          # Environment variable template
├── .gitignore            # Ignored files (secrets, caches)
└── README.md             # Platform documentation & submission guide
```

---

## ⚡ Quick Start

### 1. Prerequisites
* Python 3.9 or higher installed.

### 2. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/your-username/businesspulse.git
cd businesspulse
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env` and configure your API credentials:
```bash
cp .env.example .env
```
Edit `.env`:
```env
# Google Gemini API Key (from https://aistudio.google.com/)
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Set Gemini model (Default: gemini-2.5-flash)
GEMINI_MODEL=gemini-2.5-flash
```
> **Note**: If `GEMINI_API_KEY` is not provided, the application operates in **Mock/Offline Mode**, using built-in deterministic fallbacks.

### 4. Run the Application
Launch the Streamlit dashboard:
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 🧪 Testing with Sample Data

Click **"Load Demo Dataset"** in the app sidebar to load `sample_data.csv`. The demo dataset includes 900+ realistic transaction records with embedded patterns:
- **Product A**: Sharp 56% decline in H2.
- **West Region**: 63% revenue contraction.
- **Product D**: Low gross margin (0.8%).
- **Margin Compression**: Dropping overall profit margins over time.

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for details.
