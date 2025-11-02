# Version1.py
"""
Version1.py - Live Nifty Robo-Advisor (Streamlit)
- Live Nifty stocks (yfinance)
- Smart robo-allocation by age/income/risk
- Multi-goal planner (data editor)
- SIP projection + Monte Carlo simulation (live)
- Risk-return metrics (Sharpe-like, Sortino-like, VaR)
- Efficient-frontier approximation (random portfolios)
- Macro tilt: pulls CPI & GDP (World Bank) and adjusts expected returns
Requirements: streamlit, plotly, yfinance, pandas, numpy, requests
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import requests
from math import ceil
from datetime import datetime

# -------------------------
# Page config & small styles
# -------------------------
st.set_page_config(page_title="Live Nifty Robo-Advisor (v1)", layout="wide", initial_sidebar_state="expanded")
st.markdown("## 💹 Live Nifty Robo-Advisor — Version1")
st.markdown("Live allocations, macro tilt (inflation & GDP), Monte Carlo, and planner.")
st.write("")

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
# Sidebar: profile, toggles & macro panel
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
        edited = st.data_editor(goals_df, num_rows="dynamic")
        st.session_state.goals = edited.to_dict("records")

    st.markdown("---")
    st.subheader("Investment inputs")
    current_investment = st.number_input("Current invested (lump sum ₹)", min_value=0, value=500000, step=10000)
    use_sip = st.checkbox("Use monthly SIP", value=True)
    if use_sip:
        monthly_sip = st.number_input("Monthly SIP (₹)", min_value=0, value=10000, step=500)
    else:
        monthly_sip = 0
    default_horizon = st.slider("Default horizon (yrs)", 1, 40, 10)

    st.markdown("---")
    st.subheader("Computation limits & behavior")
    mc_sims = st.slider("Monte Carlo simulations", 200, 4000, 1200, step=100)
    frontier_samples = st.slider("Efficient frontier samples", 50, 2000, 400, step=50)
    lookback_years = st.selectbox("Live-data lookback (yrs)", [1, 3, 5], index=2)
    cache_mc = st.checkbox("Cache Monte Carlo (faster UX, fewer recalcs)", value=False)

    st.markdown("---")
    st.subheader("Ticker overrides (optional)")
    t_nifty = st.text_input("Nifty ticker (Yahoo)", value="^NSEI")
    t_gold = st.text_input("Gold ETF ticker (Yahoo)", value="GOLDBEES.NS")
    t_international = st.text_input("International ETF ticker (Yahoo)", value="VTI")

    st.markdown("---")
    st.subheader("Macro tilt settings")
    st.write("Macro tilt uses CPI (inflation) & GDP growth pulled from World Bank.")
    apply_macro_tilt = st.checkbox("Apply macro tilt to expected returns", value=True)
    tilt_strength = st.slider("Macro tilt strength (0 = off, 1 = default)", 0.0, 2.0, 1.0, step=0.1)

# -------------------------
# Asset universe & baseline
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
# Robo-allocation engine (unchanged behavior improved)
# -------------------------
def robo_allocation(age, income, risk_label, bases=BASELINE_MAP, extra_assets=[]):
    base = bases[risk_label].copy()
    for a in extra_assets:
        if a not in base and a in ASSET_CLASSES:
            base[a] = 2.0

    # age tilt
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

    if income > 150000:
        base["International Equity"] = base.get("International Equity", 0) + 2

    total = sum(base.values())
    if total <= 0:
        # fallback
        base = bases["Moderate"].copy()
        total = sum(base.values())
    alloc = {k: round(v / total * 100, 2) for k, v in base.items()}
    return alloc

allocation = robo_allocation(age, monthly_income_input, self_declared_risk)
alloc_df = pd.DataFrame({"Asset Class": list(allocation.keys()), "Allocation (%)": list(allocation.values())})

# editable allocation (live)
alloc_df = st.data_editor(alloc_df, num_rows="dynamic", use_container_width=True)

# ensure sum ~100 -> rescale automatically
total_alloc = alloc_df["Allocation (%)"].sum()
if abs(total_alloc - 100.0) > 0.001:
    st.info(f"Allocation sums to {total_alloc:.2f}%. Rescaling to 100% for calculations.")
    if total_alloc > 0:
        alloc_df["Allocation (%)"] = alloc_df["Allocation (%)"] / total_alloc * 100.0
    else:
        st.warning("Allocation invalid — reverting to baseline.")
        allocation = robo_allocation(age, monthly_income_input, self_declared_risk)
        alloc_df = pd.DataFrame({"Asset Class": list(allocation.keys()), "Allocation (%)": list(allocation.values())})

# -------------------------
# Macro fetchers (World Bank)
# -------------------------
WB_BASE = "https://api.worldbank.org/v2"

def wb_latest_indicator(country_code, indicator):
    """
    Fetch recent value for an indicator from World Bank API.
    Returns tuple (value, year) or (None, None).
    """
    try:
        url = f"{WB_BASE}/country/{country_code}/indicator/{indicator}?format=json&per_page=10"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        payload = r.json()
        # payload[1] is list of data points
        if len(payload) < 2 or not payload[1]:
            return None, None
        for entry in payload[1]:
            # find the most recent non-null
            if entry.get("value") is not None:
                return float(entry["value"]), int(entry["date"])
        return None, None
    except Exception:
        return None, None

def get_macro_indicators():
    """
    Returns dict { 'inflation': (val,year), 'gdp_growth': (val,year) }
    inflation (annual %): FP.CPI.TOTL.ZG
    GDP growth (annual %): NY.GDP.MKTP.KD.ZG
    """
    inflation, infl_year = wb_latest_indicator("IND", "FP.CPI.TOTL.ZG")
    gdpg, gdp_year = wb_latest_indicator("IND", "NY.GDP.MKTP.KD.ZG")
    return {"inflation": (inflation, infl_year), "gdp_growth": (gdpg, gdp_year)}

# fetch macro data (safe)
with st.spinner("Fetching macro indicators (World Bank)..."):
    macros = get_macro_indicators()
inflation_val, inflation_year = macros["inflation"]
gdp_val, gdp_year = macros["gdp_growth"]

# show macro summary in sidebar
with st.sidebar.expander("Macro indicators (World Bank)", expanded=True):
    st.write("Inflation (CPI, annual %) — World Bank latest:")
    if inflation_val is not None:
        st.write(f"{inflation_val:.2f}% (year {inflation_year})")
    else:
        st.write("Not available")
    st.write("Real GDP growth (annual %) — World Bank latest:")
    if gdp_val is not None:
        st.write(f"{gdp_val:.2f}% (year {gdp_year})")
    else:
        st.write("Not available")
    st.caption("World Bank is used as an approachable, reliable public API. Values may lag official releases by a few months.")

# -------------------------
# Expected returns & vols (defaults + live blend)
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

@st.cache_data(ttl=60*30)
def get_cagr_vol_for_ticker(ticker, years=5):
    """Return (cagr, vol) for ticker using yfinance; cached short TTL."""
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
    """
    Adjust base_returns dict in-place using macro indicators.
    Rules (configurable):
      - Convert inflation_pct & gdp_pct from percent to decimal
      - Debt returns reduced when inflation rises (real return compression)
      - Equity returns get modest boost when GDP > 4% and trimmed when GDP < 2%
      - Gold gets a small bump when inflation is high
    Multiplicative 'strength' scales the effect.
    """
    adj = base_returns.copy()
    if inflation_pct is None:
        inflation = 0.02
    else:
        inflation = inflation_pct / 100.0
    if gdp_pct is None:
        gdp = 0.04
    else:
        gdp = gdp_pct / 100.0

    # GDP gap relative to 4% baseline
    gdp_gap = gdp - 0.04

    for k, v in base_returns.items():
        new_v = v
        # Debt: penalize by ~ inflation * 0.6 * strength
        if "Debt" in k or "Bond" in k or "Cash" in k or "Fixed" in k:
            new_v = v - (inflation * 0.6 * strength)
        # Equity: modest boost if GDP gap positive, small trim if negative
        elif "Equity" in k or "Index" in k or "REIT" in k or "Active Equity" in k or "Sectoral" in k:
            new_v = v + (gdp_gap * 0.5 * strength)
            # also reduce real by inflation effect slightly
            new_v = new_v - (inflation * 0.15 * strength)
        # Gold: hedge vs inflation
        elif "Gold" in k or "Commodities" in k:
            if inflation > 0.04:
                new_v = v + (0.01 * strength)
            else:
                new_v = v + (0.002 * strength)
        # Crypto: keep speculative premium but slightly hurt by high inflation
        elif "Crypto" in k:
            new_v = v - (inflation * 0.05 * strength)

        # don't let adjusted returns go absurd negative beyond -0.2
        adj[k] = max(new_v, -0.2)
    return adj

# Build asset metrics
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

# Apply macro tilt if requested
if apply_macro_tilt:
    asset_returns = apply_macro_tilt_to_returns(asset_returns, inflation_val, gdp_val, strength=tilt_strength)

# Attach to table
alloc_df["Exp Return (%)"] = alloc_df["Asset Class"].map(lambda x: asset_returns.get(x, 0.06) * 100)
alloc_df["Volatility (%)"] = alloc_df["Asset Class"].map(lambda x: asset_vols.get(x, 0.15) * 100)
alloc_df["Allocation (₹)"] = alloc_df["Allocation (%)"] / 100.0 * current_investment

st.markdown("### Allocation details")
st.dataframe(alloc_df.style.format({"Allocation (%)": "{:.2f}", "Exp Return (%)": "{:.2f}%", "Volatility (%)": "{:.2f}%", "Allocation (₹)": "₹{:,.0f}"}), use_container_width=True)

# -------------------------
# Portfolio expected return & covariance
# -------------------------
weights = np.array(alloc_df["Allocation (%)"] / 100.0)
means = np.array([asset_returns[a] for a in alloc_df["Asset Class"]])
vols = np.array([asset_vols[a] for a in alloc_df["Asset Class"]])

# approximate cov with base correlation
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
# Monte Carlo (annual correlated) - no cache when user wants live
# -------------------------
def monte_carlo_sim(invest, monthly_sip, weights_vec, means_vec, cov_mat, years, sims, seed=None):
    if seed is not None:
        np.random.seed(seed)
    n = len(weights_vec)
    # stable Cholesky
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

# If caching toggled on, use st.cache_data wrapper dynamically
if cache_mc:
    monte_carlo_sim = st.cache_data(ttl=60*5)(monte_carlo_sim)

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
# UI: Tabs with outputs
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
    st.dataframe(alloc_df[["Asset Class","Allocation (%)","Allocation (₹)"]].set_index("Asset Class"), use_container_width=True)

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
        ef_df = pd.DataFrame({"Return": ef_returns*100, "Volatility": ef_vols*100, "Sharpe": ef_sharpe})
        fig_ef = px.scatter(ef_df, x="Volatility", y="Return", color="Sharpe", color_continuous_scale="Viridis")
        fig_ef.add_trace(go.Scatter(x=[port_vol*100], y=[port_return*100], mode="markers+text",
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
        sims_goal = max(300, int(mc_sims/4))
        mc_goal = monte_carlo_sim(current_investment, monthly_sip, weights, means, cov, g.get("years", default_horizon), sims_goal)
        p = float((mc_goal >= g["amount"]).sum() / len(mc_goal) * 100.0)
        goals_table.loc[i, "Prob. (approx)"] = f"{p:.1f}%"
    st.dataframe(goals_table, use_container_width=True)

    # Deterministic FV (fixed monthly conv)
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
            target_vals = total * (alloc_df["Allocation (%)"]/100.0).values
            cur_vals = cur_df.set_index("Asset Class")["Current Value (₹)"].reindex(alloc_df["Asset Class"]).fillna(0).values
            buy_sell = target_vals - cur_vals
            reb_df = pd.DataFrame({
                "Asset Class": alloc_df["Asset Class"],
                "Target Value (₹)": target_vals,
                "Current Value (₹)": cur_vals,
                "Buy(+)/Sell(-) (₹)": buy_sell
            })
            st.dataframe(reb_df.style.format({"Target Value (₹)":"₹{:,.0f}","Current Value (₹)":"₹{:,.0f}","Buy(+)/Sell(-) (₹)":"₹{:,.0f}"}), use_container_width=True)
            st.download_button("Download rebalance CSV", reb_df.to_csv(index=False).encode("utf-8"), file_name="rebalance.csv", mime="text/csv")

    st.markdown("---")
    st.download_button("Download allocation CSV", alloc_df.to_csv(index=False).encode("utf-8"), file_name="allocation.csv", mime="text/csv")

st.markdown("---")
st.caption("Educational tool — not investment advice. Verify tickers, tax treatment and instrument choices before acting.")
