import os
# Suppress TensorFlow logs before importing it
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import config

def predict_latest():
    print("--- Loading Models and Data ---")
    
    # 1. Load Configuration and Data
    try:
        # Load the test data to get the very latest available date
        df = pd.read_csv(config.DATA_TEST, index_col="Date", parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Could not find {config.DATA_TEST}")
        return

    # Check if enough data exists for the LSTM lookback
    if len(df) < config.LOOKBACK_DAYS:
        print(f"Error: Not enough data in {config.DATA_TEST}. Need at least {config.LOOKBACK_DAYS} days.")
        return

    # 2. Load Scalers and Models
    try:
        scaler_gpr = joblib.load(config.SCALER_GPR)
        scaler_market = joblib.load(config.SCALER_MARKET)
        spark_model = load_model(config.MODEL_SPARK)
        fuel_model = joblib.load(config.MODEL_FUEL)
    except Exception as e:
        print(f"Error loading models/scalers: {e}")
        print("Ensure you have run 'train_lstm_model.py' and 'train_market_svm.py' first.")
        return

    # 3. Prepare Data for the Latest Date
    latest_date = df.index[-1]
    print(f"Analyzing market conditions for date: {latest_date.strftime('%Y-%m-%d')}")

    # --- Spark (GPR) Data Prep ---
    # LSTM needs a sequence of the last LOOKBACK_DAYS (e.g., 30 days)
    df_spark_seq = df.iloc[-config.LOOKBACK_DAYS:]
    X_spark_raw = df_spark_seq[config.GPR_FEATURES].values
    
    # Scale using the GPR scaler
    X_spark_scaled = scaler_gpr.transform(X_spark_raw)
    
    # Reshape to (1, Time, Features) for the LSTM
    X_spark_input = X_spark_scaled.reshape(1, config.LOOKBACK_DAYS, len(config.GPR_FEATURES))

    # --- Fuel (Market) Data Prep ---
    # SVM needs just the specific day's features (the last row)
    df_fuel_latest = df.iloc[[-1]] 
    X_fuel_raw = df_fuel_latest[config.MARKET_FEATURES]
    
    # Scale using the Market scaler
    X_fuel_input = scaler_market.transform(X_fuel_raw)

    # 4. Generate Predictions
    # Spark (LSTM) output is a probability
    gpr_prob = spark_model.predict(X_spark_input, verbose=0)[0][0]
    
    # Fuel (SVM) output is [prob_class_0, prob_class_1]. We want class 1 (Crash).
    market_prob = fuel_model.predict_proba(X_fuel_input)[0][1]

    # 5. Apply Gated Logic (Thresholds from config.py)
    gpr_signal = gpr_prob > config.GPR_PANIC_THRESHOLD
    market_signal = market_prob > config.MARKET_CRITICAL_THRESHOLD
    
    # OR Logic: If either is high risk, trigger warning
    crash_warning = gpr_signal or market_signal
    
    # 6. Output Results
    print("\n" + "="*60)
    print(f"{'Model':<20} | {'Probability':<12} | {'Threshold':<10} | {'Status'}")
    print("-" * 60)
    
    gpr_status = "PANIC ⚠️" if gpr_signal else "Normal"
    print(f"{'Spark (Geopolitical)':<20} | {gpr_prob:.2%}      | {config.GPR_PANIC_THRESHOLD:.2f}       | {gpr_status}")
    
    market_status = "FRAGILE ⚠️" if market_signal else "Stable"
    print(f"{'Fuel (Market)':<20} | {market_prob:.2%}      | {config.MARKET_CRITICAL_THRESHOLD:.2f}       | {market_status}")
    
    print("-" * 60)
    
    # Combined Probability (Max of the two risks)
    combined_prob = max(gpr_prob, market_prob)
    
    print(f"\nOverall Crash Probability: {combined_prob:.2%}")
    
    if crash_warning:
        print("\n>>> 🔴 CRASH WARNING ISSUED 🔴 <<<")
        print(f"The system predicts a high likelihood of a crash within the next {config.WARNING_WINDOW_DAYS} days.")
    else:
        print("\n>>> 🟢 NO CRASH PREDICTED 🟢 <<<")
        print("Market conditions are currently within safety parameters.")
    print("="*60 + "\n")

if __name__ == "__main__":
    predict_latest()
