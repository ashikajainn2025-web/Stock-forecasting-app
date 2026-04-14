import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import datetime

# Page setup
st.set_page_config(page_title="CAPM Beta Calculator", page_icon="📊", layout="wide")
st.title("📊 CAPM Beta & Return Calculator")

# Sidebar for input
col1, col2 = st.columns([1, 1])
with col1:
    stock = st.selectbox("Choose a stock", ('TSLA', 'AAPL', 'NFLX', 'MSFT', 'MGM', 'AMZN', 'NVDA', 'GOOGL'))
with col2:
    num_years = st.number_input("Number of Years", 1, 10, value=1)

try:
    # Define date range
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365 * num_years)

    # Download stock & market data
    stock_data = yf.download(stock, start=start_date, end=end_date, progress=False)
    market_data = yf.download('^GSPC', start=start_date, end=end_date, progress=False)

    # Handle MultiIndex columns - yfinance returns MultiIndex when downloading single ticker
    # Convert to regular columns by taking the first level
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = [col[0] if isinstance(col, tuple) else col for col in stock_data.columns]
    if isinstance(market_data.columns, pd.MultiIndex):
        market_data.columns = [col[0] if isinstance(col, tuple) else col for col in market_data.columns]

    # Validate data was downloaded
    if stock_data.empty or market_data.empty:
        st.error("❌ Failed to download data. Please check your internet connection and try again.")
        st.stop()

    # Reset index to get 'Date' column (index might be DatetimeIndex)
    # yfinance returns DatetimeIndex, reset_index() converts it to a column
    stock_data = stock_data.reset_index()
    market_data = market_data.reset_index()

    # Find the date column - it's typically the first column after reset_index
    # Handle cases where column might be unnamed (empty string) or have different names
    price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    
    # Find first column that's not a price column (should be the date)
    date_col_stock = None
    date_col_market = None
    
    for col in stock_data.columns:
        if col not in price_columns:
            date_col_stock = col
            break
    if date_col_stock is None:
        date_col_stock = stock_data.columns[0]  # Fallback to first column
    
    for col in market_data.columns:
        if col not in price_columns:
            date_col_market = col
            break
    if date_col_market is None:
        date_col_market = market_data.columns[0]  # Fallback to first column
    
    # Rename date columns to 'Date' for consistency
    if date_col_stock != 'Date':
        stock_data.rename(columns={date_col_stock: 'Date'}, inplace=True)
    if date_col_market != 'Date':
        market_data.rename(columns={date_col_market: 'Date'}, inplace=True)

    # Ensure Date columns are datetime and remove timezone if present
    stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.tz_localize(None)
    market_data['Date'] = pd.to_datetime(market_data['Date']).dt.tz_localize(None)

    # Validate 'Close' column exists
    if 'Close' not in stock_data.columns:
        st.error(f"❌ Missing 'Close' column in stock data. Available columns: {list(stock_data.columns)}")
        st.stop()
    if 'Close' not in market_data.columns:
        st.error(f"❌ Missing 'Close' column in market data. Available columns: {list(market_data.columns)}")
        st.stop()

    # Calculate daily returns
    stock_data['Stock Return'] = stock_data['Close'].pct_change() * 100
    market_data['Market Return'] = market_data['Close'].pct_change() * 100

    # Merge on Date
    merged_data = pd.merge(
        stock_data[['Date', 'Stock Return']],
        market_data[['Date', 'Market Return']],
        on='Date',
        how='inner'
    )

    # Drop NaNs
    merged_data = merged_data.dropna()

    # Validate merged data
    if merged_data.empty:
        st.error("❌ No overlapping data between stock and market. Please try a different date range.")
        st.stop()

    if 'Market Return' not in merged_data.columns:
        st.error(f"❌ Missing 'Market Return' column after merge. Available columns: {list(merged_data.columns)}")
        st.stop()
    if 'Stock Return' not in merged_data.columns:
        st.error(f"❌ Missing 'Stock Return' column after merge. Available columns: {list(merged_data.columns)}")
        st.stop()

    # CAPM Regression
    X = sm.add_constant(merged_data['Market Return'])
    Y = merged_data['Stock Return']
    model = sm.OLS(Y, X).fit()

    # Get alpha, beta
    alpha = model.params['const']
    beta = model.params['Market Return']

    # CAPM Expected Return
    rf = 0.0  # risk-free rate
    rm = merged_data['Market Return'].mean() * 252  # Annualized market return
    expected_return = rf + beta * (rm - rf)

    # Display results
    st.subheader("📈 Results")
    st.markdown(f"**Beta (β): `{beta:.4f}`**")
    st.markdown(f"**Alpha (α): `{alpha:.4f}`**")
    st.markdown(f"**Expected Annual Return: `{expected_return:.2f}%`**")

    # Flatten column names if needed
    merged_data.columns = merged_data.columns.map(lambda x: x if isinstance(x, str) else x[0])

    # Plot
    fig = px.scatter(
        merged_data,
        x='Market Return',
        y='Stock Return',
        title=f'{stock} vs Market (CAPM Analysis)',
        trendline='ols',
        labels={'Market Return': 'Market Return (%)', 'Stock Return': 'Stock Return (%)'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Explanation
    st.markdown("---")
    st.markdown("### ℹ️ Interpretation")
    st.markdown("""
    - **Beta > 1** → More volatile than the market.
    - **Beta < 1** → Less volatile than the market.
    - **Beta ≈ 1** → Moves similar to the market.
    - **Alpha** shows extra return beyond market prediction.
    """)

except Exception as e:
    import traceback
    st.error(f"❌ Error fetching or processing data: `{e}`")
    with st.expander("🔍 Error Details (Click to expand)"):
        st.code(traceback.format_exc())