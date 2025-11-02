import yfinance as yf
import pandas as pd
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def get_index_data(ticker="^NSEI", years=5):
    """Fetch historical OHLC for given index"""
    df = yf.download(ticker, period=f"{years}y", interval="1d", progress=False)
    df.reset_index(inplace=True)
    return df

def get_cpi_data():
    """Fetch Indian CPI data from Quandl or fallback RBI API"""
    api_key = os.getenv("QUANDL_KEY")
    url = f"https://data.nasdaq.com/api/v3/datasets/RBI/CPI.json?api_key={api_key}"
    try:
        res = requests.get(url)
        data = res.json()
        df = pd.DataFrame(data["dataset"]["data"], columns=["Date", "CPI"])
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)
        return df
    except Exception as e:
        print("CPI fetch failed:", e)
        return pd.DataFrame()

def get_macro_data():
    """Simple macro snapshot with inflation, GDP, and risk-free rate"""
    inflation = 5.2  # default India inflation %
    gdp_growth = 6.8  # India GDP growth %
    rf_rate = 0.04    # risk-free (approx 10Y GSec)
    cpi_df = get_cpi_data()
    if not cpi_df.empty:
        recent = cpi_df.iloc[-1]["CPI"]
        old = cpi_df.iloc[-13]["CPI"] if len(cpi_df) > 12 else cpi_df.iloc[0]["CPI"]
        inflation = (recent - old) / old * 100
    return {"inflation": inflation, "gdp_growth": gdp_growth, "rf_rate": rf_rate}

def get_mutual_fund_nav(keyword="Index Fund"):
    """Fetch Indian mutual fund NAVs using AMFI"""
    url = "https://www.amfiindia.com/spages/NAVAll.txt"
    df = pd.read_csv(url, sep=';', header=None, on_bad_lines='skip', encoding='utf-8')
    df.columns = ["Scheme Code", "ISIN Div Payout/ISIN Growth", "ISIN Div Reinvestment", 
                  "Scheme Name", "NAV", "Repurchase Price", "Sale Price", "Date"]
    df = df[df["Scheme Name"].str.contains(keyword, case=False, na=False)]
    df.reset_index(drop=True, inplace=True)
    return df
