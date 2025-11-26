# Version1.py
"""
Live Nifty Robo-Advisor (Version1) - Streamlit
Features:
 - Profile-based recommended allocation (age, income, risk)
 - Editable allocation table (tweak recommendation)
 - Macro tilt (MOSPI/Quandl fallback -> World Bank)
 - SIP projection + Monte Carlo (cache toggle)
 - Efficient frontier (random portfolios)
 - Rebalance worksheet (editable holdings -> buy/sell)
Requirements: streamlit, pandas, numpy, plotly, yfinance, requests
"""

import os
from math import ceil
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import requests
import plotly.io as pio

pio.templates["moon"] = pio.templates["plotly_dark"]

pio.templates["moon"].layout.update(
    font=dict(color="#F5D5E0"),
    colorway=["#F5D5E0", "#6667AB", "#7B337E", "#420D4B", "#210635"],
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)

pio.templates.default = "moon"



# -------------------------
# Page config & header
# -------------------------
st.set_page_config(page_title="Live Nifty Robo-Advisor (v1)", layout="wide", initial_sidebar_state="expanded")
# -------------------------
# -------------------------
# Moon Theme – Polished UI (fonts + tables)
# -------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

:root {
    --moon-light: #F5D5E0;
    --moon-blue:  #6667AB;
    --moon-purple:#7B337E;
    --moon-deep:  #420D4B;
    --moon-dark:  #210635;
}

/* Global font + background */
* {
    font-family: 'Poppins', sans-serif !important;
}
.stApp {
    background: radial-gradient(circle at top left, #7B337E 0%, #210635 45%, #000000 100%) !important;
    color: var(--moon-light) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #210635 0%, #420D4B 60%, #210635 100%) !important;
    color: var(--moon-light) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--moon-light) !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}

/* Headers */
h1 {
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em;
}
h2, h3 {
    font-weight: 600 !important;
}

/* Buttons */
.stButton>button, .stDownloadButton>button {
    background: linear-gradient(135deg, #7B337E, #6667AB) !important;
    color: #F5D5E0 !important;
    border: none !important;
    border-radius: 999px !important;
    padding: 0.35rem 1.1rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    background: linear-gradient(135deg, #F5D5E0, #7B337E) !important;
    color: #210635 !important;
    box-shadow: 0 0 12px rgba(245, 213, 224, 0.7) !important;
}

/* Inputs (number, text, select, sliders labels) */
input, select, textarea {
    background-color: #210635 !important;
    color: var(--moon-light) !important;
    border-radius: 8px !important;
    border: 1px solid rgba(245, 213, 224, 0.35) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}
input:focus, select:focus, textarea:focus {
    outline: none !important;
    box-shadow: 0 0 0 1px #F5D5E0 !important;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: rgba(33, 6, 53, 0.9) !important;
    border-radius: 12px !important;
    padding: 10px !important;
    border: 1px solid rgba(245, 213, 224, 0.25) !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    background-color: transparent !important;
    color: #F5D5E0 !important;
    border-radius: 999px !important;
    font-weight: 500 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #7B337E, #6667AB) !important;
    color: #F5D5E0 !important;
}

/* DataFrames / tables */
[data-testid="stDataFrame"] {
    background-color: rgba(15, 4, 30, 0.85) !important;
    border-radius: 12px !important;
    padding: 4px !important;
}
[data-testid="stDataFrame"] table {
    font-size: 0.85rem !important;
    color: var(--moon-light) !important;
    border-collapse: collapse !important;
}
[data-testid="stDataFrame"] thead tr th {
    background-color: rgba(10, 3, 25, 0.95) !important;
    font-weight: 600 !important;
    padding: 6px 8px !important;
    border-bottom: 1px solid rgba(245, 213, 224, 0.35) !important;
}
[data-testid="stDataFrame"] tbody tr:nth-child(odd) {
    background-color: rgba(33, 6, 53, 0.9) !important;
}
[data-testid="stDataFrame"] tbody tr:nth-child(even) {
    background-color: rgba(24, 4, 40, 0.9) !important;
}
[data-testid="stDataFrame"] tbody tr td {
    border-bottom: 1px solid rgba(245, 213, 224, 0.15) !important;
    padding: 5px 8px !important;
}
[data-testid="stDataFrame"] tbody tr:hover td {
    background-color: rgba(123, 51, 126, 0.6) !important;
}

/* Small caption text (disclaimer) */
footer, .stCaption, .stMarkdown p small {
    font-size: 0.72rem !important;
    opacity: 0.8;
}

</style>
""", unsafe_allow_html=True)
st.title("💹 Live Nifty Robo-Advisor — Version1")
st.write("Live allocations, macro tilt (inflation & GDP), Monte Carlo, and planner.")


# -------------------------
# Helpers
# -------------------------
def fmt_inr(v):
    try:
        return "₹{:,.0f}".format(v)
    except Exception:
        try:
            return f"₹{float(v):,.0f}"
        except Exception:
            return str(v)


# -------------------------
# Sidebar: profile & settings
# -------------------------
with st.sidebar:
    st.header("Profile & Settings")
    age = st.slider("Age", 18, 75, 34)
    monthly_income_input = st.number_input("Monthly income (₹)", min_value=0, value=70000, step=5000)
    self_declared_risk = st.selectbox("Risk appetite", ["Low", "Moderate", "High"])

    st.markdown("---")
    st.subheader("Goal settings")
    if "goals" not in st.session_state:
        st.session_state.goals = [
            {"name": "Retirement", "amount": 8000000, "years": 25},
            {"name": "Home", "amount": 3000000, "years": 8}
        ]
    with st.expander("View / edit goals"):
        goals_df = pd.DataFrame(st.session_state.goals)
        edited_goals = st.data_editor(goals_df, hide_index=True, use_container_width=True, key="goals_editor")
        # store back
        st.session_state.goals = edited_goals.to_dict("records")

    st.markdown("---")
    st.subheader("Investment inputs")
    current_investment = st.number_input("Current invested (lump sum ₹)", min_value=0, value=500000, step=10000)
    use_sip = st.checkbox("Use monthly SIP", value=True)
    monthly_sip = st.number_input("Monthly SIP (₹)", min_value=0, value=10000, step=500) if use_sip else 0
    default_horizon = st.slider("Default horizon (yrs)", 1, 40, 10)

    st.markdown("---")
    st.subheader("Computation & behavior")
    mc_sims = st.slider("Monte Carlo simulations", 200, 4000, 1200, step=100)
    frontier_samples = st.slider("Efficient frontier samples", 50, 2000, 400, step=50)
    lookback_years = st.selectbox("Live-data lookback (yrs)", [1, 3, 5], index=2)
    cache_mc = st.checkbox("Cache Monte Carlo (faster UX)", value=False)

    st.markdown("---")
    st.subheader("Ticker overrides (optional)")
    t_nifty = st.text_input("Nifty ticker (Yahoo)", value="^NSEI")
    t_gold = st.text_input("Gold ETF ticker (Yahoo)", value="GOLDBEES.NS")
    t_international = st.text_input("International ETF ticker (Yahoo)", value="VTI")

    st.markdown("---")
    st.subheader("Macro tilt settings")
    st.write("Macro tilt uses CPI & GDP (MOSPI/Quandl fallback -> World Bank).")
    apply_macro_tilt = st.checkbox("Apply macro tilt to expected returns", value=True)
    tilt_strength = st.slider("Macro tilt strength", 0.0, 2.0, 1.0, step=0.1)


# -------------------------
# Asset universe & baseline templates
# -------------------------
ASSET_CLASSES = [
    "Large Cap Equity", "Mid/Small Cap Equity", "International Equity", "Index ETFs",
    "Active Equity Funds", "Sectoral Funds", "Debt Funds", "Government Bonds",
    "Corporate Bonds", "Gold ETF", "REITs", "Real Estate (Direct)",
    "Cash / Liquid", "Fixed Deposits", "Commodities (other)", "Crypto (speculative)"
]

BASELINE_MAP = {
    "Low": {"Large Cap Equity": 25, "Mid/Small Cap Equity": 5, "International Equity": 5, "Index ETFs": 10,
            "Debt Funds": 30, "Gold ETF": 10, "REITs": 5, "Cash / Liquid": 10},
    "Moderate": {"Large Cap Equity": 35, "Mid/Small Cap Equity": 10, "International Equity": 8, "Index ETFs": 10,
                 "Debt Funds": 20, "Gold ETF": 7, "REITs": 5, "Cash / Liquid": 5},
    "High": {"Large Cap Equity": 45, "Mid/Small Cap Equity": 15, "International Equity": 10, "Index ETFs": 10,
             "Debt Funds": 10, "Gold ETF": 5, "REITs": 3, "Cash / Liquid": 2, "Crypto (speculative)": 0}
}


# -------------------------
# Robo-allocation engine
# -------------------------
def robo_allocation(age, income, risk_label, bases=BASELINE_MAP, extra_assets=[]):
    base = bases[risk_label].copy()
    for a in extra_assets:
        if a not in base and a in ASSET_CLASSES:
            base[a] = 2.0

    # age tilt: younger -> slightly more equity; older -> more debt
    age_tilt = 0
    if age < 35:
        age_tilt = 5
    elif age > 55:
        age_tilt = -5

    if age_tilt != 0 and "Debt Funds" in base:
        shift = max(0, age_tilt)
        avail = base.get("Debt Funds", 0)
        use_shift = min(shift, avail)
        base["Debt Funds"] = max(0, base.get("Debt Funds", 0) - use_shift)
        base["Large Cap Equity"] = base.get("Large Cap Equity", 0) + use_shift

    # income influence (higher income -> slightly more international)
    if income > 150000:
        base["International Equity"] = base.get("International Equity", 0) + 2

    total = sum(base.values())
    if total <= 0:
        base = bases["Moderate"].copy()
        total = sum(base.values())

    alloc = {k: round(v / total * 100, 2) for k, v in base.items()}
    return alloc


# recommended allocation (based on profile)
recommended_alloc = robo_allocation(age, monthly_income_input, self_declared_risk)
recommended_df = pd.DataFrame({"Asset Class": list(recommended_alloc.keys()), "Allocation (%)": list(recommended_alloc.values())})

st.markdown("## Recommended allocation (based on your profile)")
st.dataframe(recommended_df.set_index("Asset Class"), use_container_width=True)

# Editable allocation below recommendation (user may tweak)
st.markdown("### Edit allocation (tweak recommended allocation)")
alloc_df = st.data_editor(recommended_df, hide_index=True, use_container_width=True, key="alloc_editor")

# Normalize allocation for calculations: rescale to 100 if sum differs
total_alloc = alloc_df["Allocation (%)"].sum()
if abs(total_alloc - 100.0) > 0.001:
    st.info(f"Allocation sums to {total_alloc:.2f}%. Rescaling to 100% for calculations.")
    if total_alloc > 0:
        alloc_df["Allocation (%)"] = alloc_df["Allocation (%)"] / total_alloc * 100.0
    else:
        st.warning("Allocation invalid — reverting to recommended baseline.")
        alloc_df = recommended_df.copy()


# -------------------------
# Macro fetchers (MOSPI/Quandl -> World Bank fallback)
# -------------------------
WB_BASE = "https://api.worldbank.org/v2"


def wb_latest_indicator(country_code, indicator):
    try:
        url = f"{WB_BASE}/country/{country_code}/indicator/{indicator}?format=json&per_page=10"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        payload = r.json()
        if len(payload) < 2 or not payload[1]:
            return None, None
        for entry in payload[1]:
            if entry.get("value") is not None:
                return float(entry["value"]), int(entry["date"])
        return None, None
    except Exception:
        return None, None


def try_quandl_rbi_cpi(api_key):
    if not api_key:
        return None, None
    try:
        url = f"https://data.nasdaq.com/api/v3/datasets/RBI/CPI.json?api_key={api_key}"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        j = r.json()
        data = j.get("dataset", {}).get("data", [])
        if not data:
            return None, None
        for row in data:
            if row and row[1] is not None:
                return float(row[1]), int(row[0][:4])
        return None, None
    except Exception:
        return None, None


def try_mospi_cpi_scrape():
    # best-effort MOSPI CSV attempt; may fail for many deployments (kept as fallback)
    try:
        url = "http://mospi.gov.in/sites/default/files/publication_reports/Consumer%20Price%20Indices_0.csv"
        r = requests.get(url, timeout=8)
        if r.status_code == 200 and "CPI" in r.text[:2000]:
            lines = r.text.splitlines()
            for line in reversed(lines[-200:]):
                toks = [t.strip() for t in line.split(",") if t.strip()]
                for t in reversed(toks):
                    try:
                        v = float(t)
                        return v, None
                    except Exception:
                        continue
        return None, None
    except Exception:
        return None, None


def get_macro_indicators_combined():
    nasdaq_key = os.environ.get("NASDAQ_API_KEY") or None
    cpi_val, cpi_year = None, None

    if nasdaq_key:
        try:
            cpi_val, cpi_year = try_quandl_rbi_cpi(nasdaq_key)
        except Exception:
            cpi_val, cpi_year = None, None

    if cpi_val is None:
        cpi_val, cpi_year = try_mospi_cpi_scrape()

    if cpi_val is None:
        cpi_val, cpi_year = wb_latest_indicator("IND", "FP.CPI.TOTL.ZG")

    gdp_val, gdp_year = wb_latest_indicator("IND", "NY.GDP.MKTP.KD.ZG")
    return {"inflation": (cpi_val, cpi_year), "gdp_growth": (gdp_val, gdp_year)}


with st.spinner("Fetching macro indicators (MOSPI / RBI / World Bank fallback)..."):
    macros = get_macro_indicators_combined()
inflation_val, inflation_year = macros["inflation"]
gdp_val, gdp_year = macros["gdp_growth"]


# -------------------------
# Expected returns & volatilities (defaults + live blend)
# -------------------------
DEFAULT_RET = {
    "Large Cap Equity": 0.10, "Mid/Small Cap Equity": 0.13, "International Equity": 0.09,
    "Index ETFs": 0.095, "Debt Funds": 0.06, "Gold ETF": 0.07, "REITs": 0.08,
    "Cash / Liquid": 0.035, "Fixed Deposits": 0.05, "Corporate Bonds": 0.055,
    "Government Bonds": 0.04, "Crypto (speculative)": 0.20
}
DEFAULT_VOL = {k: 0.18 if "Equity" in k else 0.07 for k in DEFAULT_RET.keys()}

TICKER_MAP = {
    "Large Cap Equity": t_nifty or "^NSEI",
    "Gold ETF": t_gold or "GOLDBEES.NS",
    "International Equity": t_international or "VTI"
}


@st.cache_data(ttl=60 * 30)
def get_cagr_vol_for_ticker(ticker, years=5):
    if not ticker:
        return None, None
    try:
        hist = yf.Ticker(ticker).history(period=f"{years}y", interval="1d")
        close = hist["Close"].dropna()
        if len(close) < 10:
            return None, None
        total_years = (close.index[-1] - close.index[0]).days / 365.25
        if total_years <= 0:
            return None, None
        cagr = (close.iloc[-1] / close.iloc[0]) ** (1.0 / total_years) - 1.0
        vol = close.pct_change().dropna().std() * np.sqrt(252)
        return float(cagr), float(vol)
    except Exception:
        return None, None


def apply_macro_tilt_to_returns(base_returns, inflation_pct, gdp_pct, strength=1.0):
    adj = base_returns.copy()
    if inflation_pct is None:
        inflation = 0.02
    else:
        inflation = inflation_pct / 100.0
    if gdp_pct is None:
        gdp = 0.04
    else:
        gdp = gdp_pct / 100.0

    gdp_gap = gdp - 0.04

    for k, v in base_returns.items():
        new_v = v
        if "Debt" in k or "Bond" in k or "Cash" in k or "Fixed" in k:
            new_v = v - (inflation * 0.6 * strength)
        elif "Equity" in k or "Index" in k or "REIT" in k or "Active Equity" in k or "Sectoral" in k:
            new_v = v + (gdp_gap * 0.5 * strength)
            new_v = new_v - (inflation * 0.15 * strength)
        elif "Gold" in k or "Commodities" in k:
            if inflation > 0.04:
                new_v = v + (0.01 * strength)
            else:
                new_v = v + (0.002 * strength)
        elif "Crypto" in k:
            new_v = v - (inflation * 0.05 * strength)

        adj[k] = max(new_v, -0.2)
    return adj


# Build asset metrics using the (possibly edited) allocation list
asset_returns = {}
asset_vols = {}
for row in alloc_df.to_dict("records"):
    a = row["Asset Class"]
    ticker = TICKER_MAP.get(a)
    if ticker:
        c, v = get_cagr_vol_for_ticker(ticker, years=lookback_years)
        if c is not None:
            asset_returns[a] = 0.7 * c + 0.3 * DEFAULT_RET.get(a, 0.06)
            asset_vols[a] = 0.7 * v + 0.3 * DEFAULT_VOL.get(a, 0.15)
        else:
            asset_returns[a] = DEFAULT_RET.get(a, 0.06)
            asset_vols[a] = DEFAULT_VOL.get(a, 0.15)
    else:
        asset_returns[a] = DEFAULT_RET.get(a, 0.06)
        asset_vols[a] = DEFAULT_VOL.get(a, 0.15)

if apply_macro_tilt:
    asset_returns = apply_macro_tilt_to_returns(asset_returns, inflation_val, gdp_val, strength=tilt_strength)

alloc_df["Exp Return (%)"] = alloc_df["Asset Class"].map(lambda x: asset_returns.get(x, 0.06) * 100)
alloc_df["Volatility (%)"] = alloc_df["Asset Class"].map(lambda x: asset_vols.get(x, 0.15) * 100)
alloc_df["Allocation (₹)"] = alloc_df["Allocation (%)"] / 100.0 * current_investment

st.markdown("### Allocation details")
st.dataframe(alloc_df.style.format({
    "Allocation (%)": "{:.2f}",
    "Exp Return (%)": "{:.2f}%",
    "Volatility (%)": "{:.2f}%",
    "Allocation (₹)": "₹{:,.0f}"
}), use_container_width=True)


# -------------------------
# Portfolio expected return & covariance
# -------------------------
weights = np.array(alloc_df["Allocation (%)"] / 100.0)
means = np.array([asset_returns[a] for a in alloc_df["Asset Class"]])
vols = np.array([asset_vols[a] for a in alloc_df["Asset Class"]])

base_corr = 0.25
cov = np.outer(vols, vols) * base_corr
np.fill_diagonal(cov, vols ** 2)

port_return = float(np.dot(weights, means))
port_vol = float(np.sqrt(weights @ cov @ weights))


# -------------------------
# Risk metrics
# -------------------------
rf_rate = 0.04


def sharpe_like(mu, vol, rf=rf_rate):
    return (mu - rf) / (vol + 1e-9)


def sortino_like(means_vec, rf=rf_rate):
    downside = np.sqrt(np.mean(np.minimum(0, means_vec - rf) ** 2))
    mu = np.mean(means_vec)
    return (mu - rf) / (downside + 1e-9)


sh = sharpe_like(port_return, port_vol)
so = sortino_like(means)


# -------------------------
# Monte Carlo simulation
# -------------------------
def monte_carlo_sim(invest, monthly_sip, weights_vec, means_vec, cov_mat, years, sims, seed=None):
    if seed is not None:
        np.random.seed(seed)
    n = len(weights_vec)
    L = np.linalg.cholesky(cov_mat + np.eye(n) * 1e-12)
    results = np.zeros(sims)
    annual_sip = monthly_sip * 12.0
    base_alloc = weights_vec * invest
    for s in range(sims):
        asset_vals = base_alloc.copy()
        for y in range(years):
            z = np.random.normal(size=n)
            ret = means_vec + L @ z
            asset_vals = asset_vals * (1 + ret)
            if annual_sip > 0:
                asset_vals += annual_sip * weights_vec
        results[s] = asset_vals.sum()
    return results


if cache_mc:
    monte_carlo_sim = st.cache_data(ttl=60 * 5)(monte_carlo_sim)

with st.spinner("Running Monte Carlo simulation..."):
    mc = monte_carlo_sim(current_investment, monthly_sip, weights, means, cov, default_horizon, mc_sims)

prob_meet = float((mc >= sum([g["amount"] for g in st.session_state.goals])).sum() / len(mc) * 100.0)
median_end = float(np.median(mc))
p10 = float(np.percentile(mc, 10))
p90 = float(np.percentile(mc, 90))


# -------------------------
# Efficient frontier (random portfolios)
# -------------------------
def random_weights(n, samples):
    r = np.random.random((samples, n))
    r /= r.sum(axis=1)[:, None]
    return r


samples = min(max(50, frontier_samples), 2000)
rand_w = random_weights(len(weights), samples)
ef_returns = rand_w.dot(means)
ef_vols = np.sqrt(np.einsum('ij,jk,ik->i', rand_w, cov, rand_w))
ef_sharpe = (ef_returns - rf_rate) / (ef_vols + 1e-9)


# -------------------------
# UI Tabs: summary, visuals, planner
# -------------------------
tab1, tab2, tab3 = st.tabs(["Summary", "Visuals", "Planner & Actions"])

with tab1:
    st.header("Portfolio summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exp. annual return", f"{port_return*100:.2f}%")
    c2.metric("Est. volatility (σ)", f"{port_vol*100:.2f}%")
    c3.metric(f"Median MC end ({default_horizon}y)", fmt_inr(median_end))
    c4.metric("Prob. meet combined goals", f"{prob_meet:.1f}%")

    st.markdown("#### Allocation (editable)")
    st.dataframe(alloc_df[["Asset Class", "Allocation (%)", "Allocation (₹)"]].set_index("Asset Class"), use_container_width=True)

with tab2:
    st.header("Interactive visuals")
    col_a, col_b = st.columns([1.1, 1.0])
    with col_a:
        st.subheader("Allocation breakdown")
        fig_p = px.pie(alloc_df, names="Asset Class", values="Allocation (%)", hole=0.35)
        fig_p.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_p, use_container_width=True)

        st.subheader("Risk vs Return (assets)")
        fig_sc = px.scatter(alloc_df, x="Volatility (%)", y="Exp Return (%)", size="Allocation (%)", text="Asset Class")
        fig_sc.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_b:
        st.subheader("Efficient frontier (samples)")
        ef_df = pd.DataFrame({"Return": ef_returns * 100, "Volatility": ef_vols * 100, "Sharpe": ef_sharpe})
        fig_ef = px.scatter(ef_df, x="Volatility", y="Return", color="Sharpe", color_continuous_scale="Viridis")
        fig_ef.add_trace(go.Scatter(x=[port_vol * 100], y=[port_return * 100], mode="markers+text",
                                   marker=dict(size=14, color="gold"), text=["Your portfolio"], textposition="top center"))
        fig_ef.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_ef, use_container_width=True)

        st.subheader("Monte Carlo distribution (final corpus)")
        fig_mc = px.histogram(mc, nbins=60, title="Monte Carlo final value distribution")
        fig_mc.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_mc, use_container_width=True)

with tab3:
    st.header("Planner & actions")
    st.subheader("Multi-goal summary")
    goals_table = pd.DataFrame(st.session_state.goals)
    goals_table["Prob. (approx)"] = None
    for i, g in enumerate(st.session_state.goals):
        sims_goal = max(300, int(mc_sims / 4))
        mc_goal = monte_carlo_sim(current_investment, monthly_sip, weights, means, cov, g.get("years", default_horizon), sims_goal)
        p = float((mc_goal >= g["amount"]).sum() / len(mc_goal) * 100.0)
        goals_table.loc[i, "Prob. (approx)"] = f"{p:.1f}%"
    st.dataframe(goals_table, use_container_width=True)

    # Deterministic portfolio FV (SIP)
    def deterministic_portfolio_fv(current_investment, monthly_sip, weights, means, years):
        weighted_annual = float(np.dot(weights, means))
        monthly_return = (1 + weighted_annual) ** (1.0 / 12.0) - 1.0
        fv = current_investment
        total_months = int(years * 12)
        for _ in range(total_months):
            fv = (fv + monthly_sip) * (1 + monthly_return)
        return fv

    st.markdown("### SIP shortfall (deterministic approximation)")
    combined_target = goals_table["amount"].sum()
    det_fv = deterministic_portfolio_fv(current_investment, monthly_sip, weights, means, default_horizon)
    st.write(f"Deterministic future value (current SIP): {fmt_inr(det_fv)}")
    if det_fv >= combined_target:
        st.success("Current SIP + lump sum is estimated to meet combined goals (deterministic).")
    else:
        lo, hi = 0, 500000
        for _ in range(40):
            mid = (lo + hi) / 2
            if deterministic_portfolio_fv(current_investment, mid, weights, means, default_horizon) >= combined_target:
                hi = mid
            else:
                lo = mid
        suggested = int(ceil(hi))
        st.warning(f"Estimate: increase SIP to ~ {fmt_inr(suggested)} / month to meet combined goals (deterministic).")

    st.markdown("---")
    # Rebalance worksheet inside an expander
    with st.expander("Rebalance worksheet (paste your current holdings)"):
        st.write("Paste your current holdings (Asset Class, Current Value). App will calculate buy/sell to reach target weights.")
        cur_df = st.data_editor(
            pd.DataFrame(columns=["Asset Class", "Current Value (₹)"]),
            hide_index=True,
            use_container_width=True,
            key="cur_value_editor"
        )

        if not cur_df.empty:
            cur_df = cur_df[cur_df["Asset Class"].isin(alloc_df["Asset Class"])]
            total = cur_df["Current Value (₹)"].sum()
            if total > 0:
                target_vals = total * (alloc_df["Allocation (%)"] / 100.0).values
                cur_vals = cur_df.set_index("Asset Class")["Current Value (₹)"].reindex(alloc_df["Asset Class"]).fillna(0).values
                buy_sell = target_vals - cur_vals
                reb_df = pd.DataFrame({
                    "Asset Class": alloc_df["Asset Class"],
                    "Target Value (₹)": target_vals,
                    "Current Value (₹)": cur_vals,
                    "Buy(+)/Sell(-) (₹)": buy_sell
                })
                st.dataframe(reb_df.style.format({
                    "Target Value (₹)": "₹{:,.0f}",
                    "Current Value (₹)": "₹{:,.0f}",
                    "Buy(+)/Sell(-) (₹)": "₹{:,.0f}"
                }), use_container_width=True)
                st.download_button("Download rebalance CSV", reb_df.to_csv(index=False).encode("utf-8"), file_name="rebalance.csv", mime="text/csv")

    st.markdown("---")
    st.download_button("Download allocation CSV", alloc_df.to_csv(index=False).encode("utf-8"), file_name="allocation.csv", mime="text/csv")


st.markdown("---")
st.caption("Educational tool — not investment advice. Verify tickers, tax treatment and instrument choices before acting.")
