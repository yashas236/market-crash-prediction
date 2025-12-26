import yfinance as yf
import pandas as pd

print("Starting data download (OHLCV + VIX)...")

# Define the tickers
# ^GSPC is the S&P 500
# ^VIX is the CBOE Volatility Index
tickers = ["^GSPC", "^VIX"]

# Define the time period
start_date = "2000-01-01"
end_date = pd.to_datetime("today").strftime('%Y-%m-%d') # Get data up to today

# Download full OHLCV data for S&P 500 and VIX
data = yf.download(tickers, start=start_date, end=end_date)

# --- Save the required data ---

# Get all columns for S&P 500 (^GSPC)
sp500_data = data.loc[:, (slice(None), '^GSPC')]
sp500_data.columns = sp500_data.columns.droplevel(1) # Remove the '^GSPC' level from column names

# Get only the 'Close' column for VIX (^VIX)
vix_close = data.loc[:, ('Close', '^VIX')].rename('VIX_Close')

# Join them together
final_market_data = sp500_data.join(vix_close)

# Save to the new CSV file
final_market_data.to_csv("data/market_data_ohlcv.csv")

print("Downloaded S&P 500 (OHLCV) and VIX (Close) data.")
print("Saved combined data to data/market_data_ohlcv.csv")