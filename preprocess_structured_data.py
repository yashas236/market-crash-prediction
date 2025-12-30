import pandas as pd
import numpy as np
import pandas_ta as ta # Make sure you have run 'pip install pandas-ta'

print("Starting Phase 2: Preprocessing structured data (17-Feature Advanced)...")

# --- 1. Load All Structured Data ---
try:
    market_data = pd.read_csv("data/market_data_ohlcv.csv") 
except FileNotFoundError:
    print("Error: data/market_data_ohlcv.csv not found. Please run download_market_data.py.")
    exit()

gpr_file_name = "data/data_gpr_daily_recent.xls"
try:
    gpr_data = pd.read_excel(gpr_file_name)
except FileNotFoundError:
    print(f"Error: {gpr_file_name} not found.")
    exit()
print("Loaded market and GPR data.")

# --- 2. Clean and Standardize Data ---
market_data['Date'] = pd.to_datetime(market_data['Date'])
market_data = market_data.set_index('Date')
market_data = market_data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
if 'VIX_Close' not in market_data.columns and '^VIX' in market_data.columns:
     market_data = market_data.rename(columns={'^VIX': 'VIX_Close'})

if 'DAY' in gpr_data.columns:
    gpr_data['Date'] = pd.to_datetime(gpr_data['DAY'].astype(str), format='%Y%m%d')
    gpr_data = gpr_data.set_index('Date')
else:
    print("Error: Could not find 'DAY' column in the GPR file.")
    exit()

gpr_features = gpr_data[['GPRD', 'GPRD_THREAT', 'GPRD_ACT']].copy()
gpr_features = gpr_features.rename(columns={
    'GPRD': 'GPR_Composite', 'GPRD_THREAT': 'GPR_Threats', 'GPRD_ACT': 'GPR_Acts'
})
print("Standardized dates and column names.")

# --- 3. Engineer Advanced GPR Features (12 total) ---
print("Engineering long-term 'gradual' features for GPR...")
gpr_features['GPR_Threats_MA_63'] = gpr_features['GPR_Threats'].rolling(window=63).mean()
gpr_features['GPR_Threats_MA_126'] = gpr_features['GPR_Threats'].rolling(window=126).mean()
gpr_features['GPR_Threats_Trend'] = gpr_features['GPR_Threats_MA_63'] - gpr_features['GPR_Threats_MA_126']
gpr_features['GPR_Acts_MA_63'] = gpr_features['GPR_Acts'].rolling(window=63).mean()

print("Engineering GPR Rate of Change features...")
gpr_features['GPR_Threats_1D_PctChange'] = gpr_features['GPR_Threats'].pct_change(1) * 100
gpr_features['GPR_Acts_1D_PctChange'] = gpr_features['GPR_Acts'].pct_change(1) * 100
gpr_features['GPR_Threats_5D_PctChange'] = gpr_features['GPR_Threats'].pct_change(5) * 100

print("Engineering GPR Volatility features...")
gpr_features['GPR_Threats_Vol_21D'] = gpr_features['GPR_Threats'].rolling(window=21).std()
gpr_features['GPR_Acts_Vol_21D'] = gpr_features['GPR_Acts'].rolling(window=21).std()

change_cols = [col for col in gpr_features.columns if 'PctChange' in col]
gpr_features[change_cols] = gpr_features[change_cols].replace([np.inf, -np.inf], np.nan)
gpr_features[change_cols] = gpr_features[change_cols].fillna(0)
print("Advanced GPR features engineered.")

# --- 4. Engineer Technical Indicators ---
print("Engineering technical indicators (RSI, MACD)...")
market_data['RSI_14'] = ta.rsi(market_data['close'], length=14)
macd = ta.macd(market_data['close'])
market_data = market_data.join(macd[['MACD_12_26_9', 'MACDh_12_26_9']])
print("Technical indicators added.")

# --- 5. Define and Calculate Target Variable (Y) (Identical) ---
market_data['Log_Return'] = np.log(market_data['close'] / market_data['close'].shift(1))
rolling_window = 252
market_data['Rolling_1st_Percentile'] = market_data['Log_Return'].rolling(window=rolling_window).quantile(0.01)
market_data['Crash_Event'] = (market_data['Log_Return'] < market_data['Rolling_1st_Percentile']).astype(int)
print(f"Calculated target variable 'Crash_Event' (Y).")

# --- 6. Merge All Data Streams ---
master_df = market_data.join(gpr_features, how='left')

# --- 7. Handle Missing Values (FFill & Drop) ---
print("Forward-filling all GPR and engineered features...")
features_to_process = [
    'GPR_Composite', 'GPR_Threats', 'GPR_Acts',
    'GPR_Threats_MA_63', 'GPR_Threats_MA_126', 'GPR_Threats_Trend', 'GPR_Acts_MA_63',
    'GPR_Threats_1D_PctChange', 'GPR_Acts_1D_PctChange', 'GPR_Threats_5D_PctChange',
    'GPR_Threats_Vol_21D', 'GPR_Acts_Vol_21D',
    'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'VIX_Close'
]
for feature in features_to_process:
    if feature in master_df.columns:
        master_df[feature] = master_df[feature].ffill()

master_df = master_df.dropna()
print("Merged, ffilled, and dropped NaNs from master_df.")

# --- 8. Chronological Split (Identical) ---
train_end = '2015-12-31'
validation_end = '2018-12-31'
train_set = master_df.loc[:train_end]
validation_set = master_df.loc[pd.to_datetime(train_end) + pd.Timedelta(days=1) : validation_end]
test_set = master_df.loc[pd.to_datetime(validation_end) + pd.Timedelta(days=1) :]

# --- 9. Save Processed Datasets (local) ---
train_set.to_csv("data/train_structured.csv")
validation_set.to_csv("data/validation_structured.csv")
test_set.to_csv("data/test_structured.csv")
print("\nSuccessfully processed and saved 17-feature structured data.")