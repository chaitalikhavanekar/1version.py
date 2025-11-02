import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from data_fetcher import get_macro_data
from math import ceil

st.set_page_config(page_title="Live Robo-Advisor", layout="wide")

# ---- Sidebar Inputs ----
with st.sidebar:
    st.header("User Profile")
    age = st.slider("Age", 18, 75, 30)
    income = st.number_input("Monthly Income (₹)", value=70000, step=5000)
    risk = st.selectbox("Risk Appetite", ["Low", "Moderate", "High"])
    st.markdown("---")
    st.subheader("Investment")
    invest = st.number_input("Current Investment (₹)", value=500000)
    sip = st.number_input("Monthly SIP (₹)", value=10000)
    years = st.slider("Investment Horizon (Years)", 1, 40, 10)

# ---- Fetch Macro Data ----
macro = get_macro_data()
st.sidebar.markdown(f"**Inflation:** {macro['inflation']:.1f}%")
st.sidebar.markdown(f"**GDP Growth:** {macro['gdp_growth']:.1f}%")
st.sidebar.markdown(f"**Risk-free rate:** {macro['rf_rate']*100:.2f}%")

# ---- Allocation Logic ----
BASELINE = {
    "Low": {"Equity": 25, "Debt": 50, "Gold": 15, "Cash": 10},
    "Moderate": {"Equity": 45, "Debt": 35, "Gold": 10, "Cash": 10},
    "High": {"Equity": 65, "Debt": 20, "Gold": 10, "Cash": 5},
}

def adjust_for_age(base_alloc, age):
    if age < 30:
        base_alloc["Equity"] += 5
        base_alloc["Debt"] -= 5
    elif age > 55:
        base_alloc["Debt"] += 5
        base_alloc["Equity"] -= 5
    return base_alloc

alloc = adjust_for_age(BASELINE[risk].copy(), age)
alloc_df = pd.DataFrame(list(alloc.items()), columns=["Asset", "Allocation (%)"])

# ---- Expected Return Logic ----
base_returns = {"Equity": 0.12, "Debt": 0.06, "Gold": 0.07, "Cash": 0.03}
infl_adj = (macro["gdp_growth"] - macro["inflation"]/2) / 100
exp_returns = {k: v + infl_adj for k, v in base_returns.items()}

alloc_df["Exp Return (%)"] = alloc_df["Asset"].map(lambda x: exp_returns[x] * 100)
alloc_df["Alloc (₹)"] = alloc_df["Allocation (%)"] / 100 * invest

st.markdown("## 📊 Recommended Allocation")
st.dataframe(alloc_df.style.format({"Exp Return (%)": "{:.2f}", "Alloc (₹)": "₹{:,.0f}"}), use_container_width=True)

# ---- Projection ----
weights = alloc_df["Allocation (%)"] / 100
returns = alloc_df["Exp Return (%)"] / 100
expected_growth = float(np.dot(weights, returns))
future_val = invest * (1 + expected_growth)**years + sip * (((1 + expected_growth)**years - 1) / expected_growth) * 12

st.metric("Expected Annualized Return", f"{expected_growth*100:.2f}%")
st.metric("Projected Portfolio Value", f"₹{future_val:,.0f}")

# ---- Charts ----
fig = px.pie(alloc_df, names="Asset", values="Allocation (%)", title="Asset Distribution", hole=0.3)
st.plotly_chart(fig, use_container_width=True)
