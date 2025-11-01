import yfinance as yf
import requests
import pandas as pd

def get_index_data(ticker: str, start: str = "2021-01-01", end: str = None) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.reset_index(inplace=True)
    return df

def get_inflation_data(api_key: str) -> pd.DataFrame:
    url = f"https://data.nasdaq.com/api/v3/datasets/RBI/CPI.json?api_key={api_key}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data["dataset"]["data"], columns=["Date", "CPI"])
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    return df

def get_mutual_fund_nav(fund_keyword: str) -> pd.DataFrame:
    url = "https://www.amfiindia.com/spages/NAVAll.txt"
    df = pd.read_csv(url, sep=';', header=None, on_bad_lines='skip', encoding='utf-8')
    df.columns = ["Scheme Code", "ISIN Div Payout/ISIN Growth", "ISIN Div Reinvestment", 
                  "Scheme Name", "NAV", "Repurchase Price", "Sale Price", "Date"]
    df = df[df["Scheme Name"].str.contains(fund_keyword, case=False, na=False)]
    df.reset_index(drop=True, inplace=True)
    return df
