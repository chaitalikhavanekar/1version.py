# app.py
"""
Live Nifty + Robo-Advisor Streamlit app
- Live Nifty stocks (yfinance)
- Smart robo-allocation by age/income/risk
- Multi-goal planner (data editor)
- SIP projection + Monte Carlo simulation
- Risk-return metrics (Sharpe-like, Sortino-like, VaR)
- Efficient-frontier approximation (random portfolios)
- Plotly interactive charts, CSV downloads
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from math import ceil
from datetime import datetime

# -------------------------
# Page config & styles
# -------------------------
st.set_page_config(page_title="Live Nifty Robo-Advisor", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg,#041422 0%, #062b3a 100%); color: #E8F0F2;}
      .title { color:#6FF0B0; font-size:28px; font-weight:700; }
      .muted { color:#9FB4C8; }
      .card { background: rgba(255,255,255,0.03); padding:12px; border-radius:10px; border:1px solid rgba(255,255,255,0.03); }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<div class="title">💹 Live Nifty Robo-Advisor — Smart Asset Allocation</div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Combines live market data, automated allocation, risk analytics, SIP & Monte Carlo planning.</div>', unsafe_allow_html=True)
st.write("")  # spacing

# -------------------------
# Small helper utils
# -------------------------
def fmt_inr(v):
    return "₹{:,.0f}".format(v)

# -------------------------
# Sidebar: basic profile & settings
# -------------------------
with st.sidebar:
    st.header("Profile & Settings")
    age = st.slider("Age", 18, 75, 34)
    monthly_income = st.number_input("Monthly income (₹)", min_value=0, value=70000, step=5000)
    self_declared_risk = st.selectbox("Risk appetite", ["Low", "Moderate", "High"])
    # numeric risk score for logic
    risk_score = {"Low": 25, "Moderate": 50, "High": 75}[self_declared_risk]
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
        # store back
        st.session_state.goals = edited.to_dict("records")
    st.markdown("---")
    st.subheader("Investment inputs")
    current_investment = st.number_input("Current invested (lump sum ₹)", min_value=0, value=500000, step=10000)
    use_sip = st.checkbox("Use monthly SIP", value=True)
    monthly_sip = st.number_input("Monthly SIP (₹)", min_value=0, value=10000, step=500) if use_sip else 0
    default_horizon = st.slider("Default horizon (yrs)", 1, 40, 10)
    st.markdown("---")
    st.subheader("Computation limits")
    mc_sims = st.slider("Monte Carlo simulations", 200, 4000, 1200, step=100)
    frontier_samples = st.slider("Efficient frontier samples", 50, 2000, 400, step=50)
    lookback_years = st.selectbox("Live-data lookback (yrs)", [1, 3, 5], index=2)
    st.markdown("---")
    st.subheader("Ticker overrides (optional)")
    t_nifty = st.text_input("Nifty ticker (Yahoo)", value="^NSEI")
    t_gold = st.text_input("Gold ETF ticker (Yahoo)", value="GOLDBEES.NS")
    t_international = st.text_input("International ETF ticker", value="VTI")

# -------------------------
# Asset universe & baselines
# -------------------------
ASSET_CLASSES = [
    "Large Cap Equity", "Mid/Small Cap Equity", "International Equity",
    "Index ETFs", "Active Equity Funds", "Sectoral Funds",
    "Debt Funds", "Government Bonds", "Corporate Bonds",
    "Gold ETF", "REITs", "Real Estate (Direct)", "Cash / Liquid", "Fixed Deposits",
    "Commodities (other)", "Crypto (speculative)"
]

# baseline allocation templates keyed by risk band
BASELINE_MAP = {
    "Low": {"Large Cap Equity": 25, "Mid/Small Cap Equity": 5, "International Equity": 5, "Index ETFs": 10, "Debt Funds": 30, "Gold ETF": 10, "REITs": 5, "Cash / Liquid": 10},
    "Moderate": {"Large Cap Equity": 35, "Mid/Small Cap Equity": 10, "International Equity": 8, "Index ETFs": 10, "Debt Funds": 20, "Gold ETF": 7, "REITs": 5, "Cash / Liquid": 5},
    "High": {"Large Cap Equity": 45, "Mid/Small Cap Equity": 15, "International Equity": 10, "Index ETFs": 10, "Debt Funds": 10, "Gold ETF": 5, "REITs": 3, "Cash / Liquid": 2, "Crypto (speculative)": 0}
}

# -------------------------
# Robo-allocation engine
# -------------------------
def robo_allocation(age, income, risk_label, bases=BASELINE_MAP, extra_assets=[]):
    # start from baseline
    base = bases[risk_label].copy()
    # include any user extra assets (if not present) at small default
    for a in extra_assets:
        if a not in base and a in ASSET_CLASSES:
            base[a] = 2.0
    # age tilt: younger -> more equity, older -> more debt
    age_tilt = 0
    if age < 35:
        age_tilt = 5
    elif age > 55:
        age_tilt = -5
    # apply tilt by shifting from debt to equity
    if age_tilt != 0:
        shift = min(age_tilt, base.get("Debt Funds", 0))
        base["Debt Funds"] = max(0, base.get("Debt Funds", 0) - shift)
        base["Large Cap Equity"] = base.get("Large Cap Equity", 0) + shift
    # income influence: higher income -> slightly more international
    if income > 150000:
        base["International Equity"] = base.get("International Equity", 0) + 2
    # normalize to 100
    total = sum(base.values())
    alloc = {k: round(v / total * 100, 2) for k, v in base.items()}
    return alloc

# compute allocation and show toggle for editing
allocation = robo_allocation(age, monthly_income, self_declared_risk)
alloc_df = pd.DataFrame({"Asset Class": list(allocation.keys()), "Allocation (%)": list(allocation.values())})
alloc_df = st.data_editor(alloc_df, num_rows="dynamic", use_container_width=True)
# ensure sum = 100
if abs(alloc_df["Allocation (%)"].sum() - 100) > 0.5:
    st.warning("Total allocation should be ~100%. You can edit the allocation table above.")

# -------------------------
# Live Nifty quick view (top stocks)
# -------------------------
st.markdown("## Live Nifty snapshot")
with st.spinner("Fetching live prices (Yahoo)..."):
    try:
        top_tickers = ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS"]
        live = yf.download(top_tickers, period="5d", interval="1d")["Close"].dropna(axis=0, how="all")
        if not live.empty:
            latest = live.iloc[-1]
            df_live = pd.DataFrame({"Ticker": latest.index, "Price": latest.values})
            st.dataframe(df_live, use_container_width=True)
        else:
            st.info("No live data available for the sample tickers.")
    except Exception as e:
        st.info("Live data fetch failed: " + str(e))

# -------------------------
# Estimate expected returns & vols (blend live + defaults)
# -------------------------
DEFAULT_RET = {
    "Large Cap Equity": 0.10, "Mid/Small Cap Equity": 0.13, "International Equity": 0.09,
    "Index ETFs": 0.095, "Debt Funds": 0.06, "Gold ETF": 0.07, "REITs": 0.08, "Cash / Liquid": 0.035,
    "Fixed Deposits": 0.05, "Corporate Bonds": 0.055, "Government Bonds": 0.04, "Crypto (speculative)": 0.20
}
DEFAULT_VOL = {k: 0.18 if "Equity" in k else 0.07 for k in DEFAULT_RET.keys()}

# mapping asset -> possible ticker for live fetch
TICKER_MAP = {
    "Large Cap Equity": t_nifty or "^NSEI",
    "Gold ETF": t_gold or "GOLDBEES.NS",
    "International Equity": t_international or "VTI"
}

@st.cache_data(ttl=60*30)
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
        cagr = (close.iloc[-1] / close.iloc[0]) ** (1.0/total_years) - 1.0
        vol = close.pct_change().dropna().std() * np.sqrt(252)
        return float(cagr), float(vol)
    except Exception:
        return None, None

# build asset metrics
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

# attach to table
alloc_df["Exp Return (%)"] = alloc_df["Asset Class"].map(lambda x: asset_returns.get(x, 0.06) * 100)
alloc_df["Volatility (%)"] = alloc_df["Asset Class"].map(lambda x: asset_vols.get(x, 0.15) * 100)
alloc_df["Allocation (₹)"] = alloc_df["Allocation (%)"] / 100.0 * current_investment

st.markdown("### Allocation details")
st.dataframe(alloc_df.style.format({"Allocation (%)":"{:.2f}", "Exp Return (%)":"{:.2f}%", "Volatility (%)":"{:.2f}%","Allocation (₹)":"₹{:,.0f}"}), use_container_width=True)

# -------------------------
# Portfolio expected return & covariance
# -------------------------
weights = np.array(alloc_df["Allocation (%)"] / 100.0)
means = np.array([asset_returns[a] for a in alloc_df["Asset Class"]])
vols = np.array([asset_vols[a] for a in alloc_df["Asset Class"]])

# if we don't have series to compute cov, approximate with base correlation
base_corr = 0.25
cov = np.outer(vols, vols) * base_corr
np.fill_diagonal(cov, vols ** 2)

port_return = float(np.dot(weights, means))
port_vol = float(np.sqrt(weights @ cov @ weights))

# -------------------------
# Risk metrics: Sharpe-like, Sortino-like (approx), VaR from MC
# -------------------------
rf_rate = 0.04

def sharpe_like(mu, vol, rf=rf_rate):
    return (mu - rf) / (vol + 1e-9)

def sortino_like(means_vec, rf=rf_rate):
    # approximate downside dev using returns below rf
    downside = np.sqrt(np.mean(np.minimum(0, means_vec - rf) ** 2))
    mu = np.mean(means_vec)
    return (mu - rf) / (downside + 1e-9)

sh = sharpe_like(port_return, port_vol)
so = sortino_like(means)

# -------------------------
# Monte Carlo (annual correlated) - cached
# -------------------------
@st.cache_data(ttl=60*10)
def monte_carlo_sim(invest, monthly_sip, weights_vec, means_vec, cov_mat, years, sims):
    n = len(weights_vec)
    L = np.linalg.cholesky(cov_mat)
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
# Output panels (tabs)
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
    col_a, col_b = st.columns([1.1, 1])
    with col_a:
        st.subheader("Allocation breakdown")
        fig_p = px.pie(alloc_df, names="Asset Class", values="Allocation (%)", hole=0.35, color_discrete_sequence=px.colors.sequential.Tealgrn)
        fig_p.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_p, use_container_width=True)

        st.subheader("Risk vs Return (assets)")
        fig_sc = px.scatter(alloc_df, x="Volatility (%)", y="Exp Return (%)", size="Allocation (%)", text="Asset Class")
        fig_sc.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_b:
        st.subheader("Efficient frontier (samples)")
        ef_df = pd.DataFrame({"Return": ef_returns*100, "Volatility": ef_vols*100, "Sharpe": ef_sharpe})
        fig_ef = px.scatter(ef_df, x="Volatility", y="Return", color="Sharpe", color_continuous_scale="Viridis")
        fig_ef.add_trace(go.Scatter(x=[port_vol*100], y=[port_return*100], mode="markers+text", marker=dict(size=14, color="gold"), text=["Your portfolio"], textposition="top center"))
        fig_ef.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_ef, use_container_width=True)

        st.subheader("Monte Carlo distribution (final corpus)")
        fig_mc = px.histogram(mc, nbins=60, title="Monte Carlo final value distribution")
        fig_mc.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_mc, use_container_width=True)

with tab3:
    st.header("Planner & actions")
    st.subheader("Multi-goal summary")
    goals_table = pd.DataFrame(st.session_state.goals)
    goals_table["Prob. (approx)"] = None
    # quick per-goal probability (same portfolio, different horizon; small MC per goal to save CPU)
    for i, g in enumerate(st.session_state.goals):
        sims_goal = max(300, int(mc_sims/4))
        mc_goal = monte_carlo_sim(current_investment, monthly_sip, weights, means, cov, g.get("years", default_horizon), sims_goal)
        p = float((mc_goal >= g["amount"]).sum() / len(mc_goal) * 100.0)
        goals_table.loc[i, "Prob. (approx)"] = f"{p:.1f}%"
    st.dataframe(goals_table, use_container_width=True)

    st.markdown("### SIP shortfall (deterministic approximation)")
    def deterministic_portfolio_fv(current_investment, monthly_sip, weights, means, default_horizon):
    """
    Calculate deterministic portfolio future value based on weighted average returns.
    """
    total_months = int(default_horizon * 12)
    fv = current_investment
    # Calculate average monthly return from weighted means
    avg_monthly_return = sum(w * m for w, m in zip(weights, means)) / 12

    for _ in range(total_months):
        fv = (fv + monthly_sip) * (1 + avg_monthly_return)

    return fv
    combined_target = goals_table["amount"].sum()
    det_fv = deterministic_portfolio_fv(current_investment, monthly_sip, weights, means, default_horizon)
    st.write(f"Deterministic future value (current SIP): {fmt_inr(det_fv)}")
    if det_fv >= combined_target:
        st.success("Current SIP + lump sum is estimated to meet combined goals (deterministic).")
    else:
        suggested = find_required_sip = None
        # binary search required SIP
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
    st.subheader("Rebalance worksheet")
    st.write("Paste your current holdings (Asset Class, Current Value). App will calculate buy/sell to reach target weights.")
    cur_df = st.experimental_data_editor(pd.DataFrame(columns=["Asset Class", "Current Value (₹)"]), num_rows="dynamic")
    if not cur_df.empty:
        # filter and compute
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

# Footer
st.markdown("---")
st.caption("Educational tool — not investment advice. Verify tickers, tax treatment and instrument choices before acting.")
