import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import datetime
import plotly.express as px
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CAPM",
    page_icon="📈",
    layout="wide"
)

st.title("Capital Asset Pricing Model (CAPM)")

# ---------------- USER INPUT ----------------
col1, col2 = st.columns(2)

with col1:
    stock_list = st.multiselect(
        "Choose stocks",
        ('TSLA', 'AAPL', 'NFLX', 'MSFT', 'AMZN', 'NVDA', 'GOOGL'),
        ['TSLA', 'AAPL', 'AMZN', 'GOOGL']
    )

with col2:
    num_years = st.number_input("Number of years", 1, 10, value=1)

if not stock_list:
    st.warning("Please select at least one stock")
    st.stop()

# ---------------- DATE RANGE ----------------
end = datetime.date.today()
start = end - datetime.timedelta(days=365 * num_years)

# ---------------- LOAD SP500 ----------------
@st.cache_data(ttl=3600)
def load_sp500(start, end):
    df = web.DataReader("sp500", "fred", start, end)
    df = df.reset_index()
    df.columns = ["Date", "sp500"]
    df["Date"] = pd.to_datetime(df["Date"])
    return df

SP500 = load_sp500(start, end)

# ---------------- LOAD STOCK DATA ----------------
@st.cache_data(ttl=3600)
def load_stock_data(stocks, years):
    df = pd.DataFrame()
    date_index = None

    for stock in stocks:
        data = yf.download(
            stock,
            period=f"{years}y",
            auto_adjust=True,
            progress=False
        )

        if data.empty:
            continue

        if date_index is None:
            date_index = data.index

        df[stock] = data["Close"].values

    if df.empty:
        return None

    df["Date"] = pd.to_datetime(date_index)
    df["Date"] = df["Date"].tz_localize(None)

    return df

stocks_df = load_stock_data(stock_list, num_years)

if stocks_df is None:
    st.error("No stock data available")
    st.stop()

# ---------------- FIX TIMEZONE & MERGE ----------------
SP500["Date"] = SP500["Date"].dt.tz_localize(None)

stocks_df = pd.merge(stocks_df, SP500, on="Date", how="inner")

if stocks_df.empty:
    st.error("Merged dataframe is empty")
    st.stop()

# ---------------- SHOW DATA ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Head")
    st.dataframe(stocks_df.head(), use_container_width=True)

with col2:
    st.subheader("Data Tail")
    st.dataframe(stocks_df.tail(), use_container_width=True)

# ---------------- PLOT FUNCTION ----------------
def interactive_plot(df, title):
    fig = px.line()
    for col in df.columns:
        if col != "Date":
            fig.add_scatter(x=df["Date"], y=df[col], name=col)
    fig.update_layout(height=450, title=title)
    return fig

st.subheader("Stock Prices")
st.plotly_chart(interactive_plot(stocks_df, "Stock Prices"), use_container_width=True)

# ---------------- NORMALIZED PRICES ----------------
def normalize(df):
    df_norm = df.copy()
    for col in df_norm.columns:
        if col != "Date":
            df_norm[col] = df_norm[col] / df_norm[col].iloc[0]
    return df_norm

st.subheader("Normalized Prices")
st.plotly_chart(
    interactive_plot(normalize(stocks_df), "Normalized Prices"),
    use_container_width=True
)

# ---------------- DAILY RETURNS ----------------
def daily_return(df):
    df_ret = df.copy()
    for col in df.columns:
        if col != "Date":
            df_ret[col] = df[col].pct_change()
    df_ret.dropna(inplace=True)
    return df_ret

stocks_daily_return = daily_return(stocks_df)

# ---------------- BETA CALCULATION ----------------
def calculate_beta(df, stock):
    b, a = np.polyfit(df["sp500"], df[stock], 1)
    return b, a

beta = {}
alpha = {}

for stock in stock_list:
    b, a = calculate_beta(stocks_daily_return, stock)
    beta[stock] = b
    alpha[stock] = a

beta_df = pd.DataFrame({
    "Stock": beta.keys(),
    "Beta": [round(v, 3) for v in beta.values()]
})

st.subheader("Beta Values")
st.dataframe(beta_df, use_container_width=True)

# ---------------- CAPM EXPECTED RETURN ----------------
rf = 0.02  # risk-free rate (2%)
rm = stocks_daily_return["sp500"].mean() * 252

return_df = pd.DataFrame({
    "Stock": beta.keys(),
    "Expected Annual Return (%)": [
        round((rf + b * (rm - rf)) * 100, 2) for b in beta.values()
    ]
})

st.subheader("CAPM Expected Returns")
st.dataframe(return_df, use_container_width=True)
