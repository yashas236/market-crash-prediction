import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import warnings
import joblib
import re
import config
import utils

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

print("Starting Step 1: OR Logic Threshold Optimization (High Precision Mode)...")

# --- 1. SET PARAMETERS ---
LOOKBACK_DAYS = config.LOOKBACK_DAYS
WARNING_WINDOW_DAYS = config.WARNING_WINDOW_DAYS

# [CRITICAL CHANGE] Optimization Metric (F-Beta Score)
# Beta = 1.0 calculates the standard F1-Score (Balanced Precision & Recall).
# Set Beta < 1 (e.g., 0.5) to prioritize Precision (reducing false alarms).
OPTIMIZATION_BETA = 1

# [CRITICAL CHANGE] Minimum acceptable precision. 
# Any model with precision below 33% (1 true event for every 2 false alarms) is rejected.
MIN_PRECISION_FLOOR = 0.33 

# --- 2. LOAD MODELS AND VALIDATION DATA ---
try:
    spark_model = load_model(config.MODEL_SPARK)
    fuel_model = joblib.load(config.MODEL_FUEL)
    market_scaler = joblib.load(config.SCALER_MARKET)
    gpr_scaler = joblib.load(config.SCALER_GPR)
    df_val = pd.read_csv(config.DATA_VAL, index_col="Date", parse_dates=True)
except FileNotFoundError as e:
    print(f"Error loading files: {e}. Ensure models and data exist.")
    exit()

print("Loaded models and validation data.")

# --- 3. GENERATE PREDICTIONS ---

# A. Spark (GPR) Predictions
print("Generating Spark (GPR) predictions...")
X_gpr_val = df_val[config.GPR_FEATURES]
X_gpr_val_scaled = gpr_scaler.transform(X_gpr_val)

def create_sequences(X, lookback_period):
    X_sequences = []
    for i in range(len(X) - lookback_period):
        X_sequences.append(X[i:(i + lookback_period)])
    return np.array(X_sequences)

X_gpr_val_seq = create_sequences(X_gpr_val_scaled, LOOKBACK_DAYS)
gpr_probs_val = spark_model.predict(X_gpr_val_seq, verbose=0).flatten()

# B. Fuel (Market) Predictions
print("Generating Fuel (Market) predictions...")
X_market_val = df_val[config.MARKET_FEATURES]
X_market_val_aligned = X_market_val.iloc[LOOKBACK_DAYS:] 
X_market_val_scaled = market_scaler.transform(X_market_val_aligned)
market_fragility_probs_val = fuel_model.predict_proba(X_market_val_scaled)[:, 1]

# C. Align Ground Truth Labels
y_true_val = df_val['Crash_Event'].values[LOOKBACK_DAYS:]

# --- 4. IDENTIFY EVENTS (For Metric Calculation) ---
val_event_groups = utils.group_crash_events(y_true_val)

def calculate_event_f_beta(y_true, y_pred, event_groups, beta=1.0):
    """
    Calculates F-Beta score based on EVENTS.
    Beta < 1 prioritizes PRECISION.
    Beta > 1 prioritizes RECALL.
    """
    if not event_groups: 
        return 0, 0, 0, 0, 0

    total_events = len(event_groups)
    events_hit = 0
    hit_event_windows = []

    # 1. Evaluate Hits (Recall)
    for start_day, end_day in event_groups:
        warning_start = max(0, start_day - WARNING_WINDOW_DAYS)
        warning_end = start_day - 1
        
        if warning_start <= warning_end:
            window_preds = y_pred[warning_start : warning_end + 1]
            if np.sum(window_preds) > 0:
                events_hit += 1
                hit_event_windows.append((warning_start, end_day)) 

    # 2. Evaluate False Alarms (Precision)
    pred_alarm_indices = np.where(y_pred == 1)[0]
    false_alarms = 0

    if pred_alarm_indices.size > 0:
        # Create mask of allowed periods (windows where alarms are "correct")
        allowed_alarm_mask = np.zeros_like(y_true)
        for start, end in hit_event_windows:
            allowed_alarm_mask[start : end + 1] = 1 
        
        # Identify false alarms (alarms outside allowed windows)
        false_alarm_indices = pred_alarm_indices[allowed_alarm_mask[pred_alarm_indices] == 0]
        
        if len(false_alarm_indices) > 0:
            # Group consecutive false alarms into single "Events" to avoid double counting
            # e.g., Day 1, 2, 3 all alarming = 1 False Alarm Event, not 3.
            false_alarms = 1 + np.sum(np.diff(false_alarm_indices) > 1)

    # 3. Metrics
    recall = events_hit / total_events if total_events > 0 else 0
    
    # Precision: Hit Events / (Hit Events + False Alarm Events)
    if (events_hit + false_alarms) > 0:
        precision = events_hit / (events_hit + false_alarms)
    else:
        precision = 0
    
    # F-Beta Score formula
    if (precision + recall) > 0:
        numerator = (1 + beta**2) * (precision * recall)
        denominator = (beta**2 * precision) + recall
        f_beta = numerator / denominator
    else:
        f_beta = 0

    return f_beta, precision, recall, events_hit, false_alarms

# --- 5. PERCENTILE-BASED OPTIMIZATION LOOP ---

print("\n--- Analysing Prediction Distributions ---")

# [CRITICAL CHANGE] Scanning higher percentiles only.
# Previous code scanned 50-99. Now scanning 85-99.9.
# We are looking for "Extreme" events to reduce false positives.
gpr_percentiles = np.linspace(85, 99.9, 40) 
gpr_candidates = np.percentile(gpr_probs_val, gpr_percentiles)
gpr_candidates = np.unique(gpr_candidates)
gpr_candidates = gpr_candidates[gpr_candidates > 0.001]

fuel_percentiles = np.linspace(85, 99.9, 40)
fuel_candidates = np.percentile(market_fragility_probs_val, fuel_percentiles)
fuel_candidates = np.unique(fuel_candidates)

print(f"Testing {len(gpr_candidates)} GPR thresholds (Tail Range: {gpr_candidates.min():.4f} - {gpr_candidates.max():.4f})")
print(f"Testing {len(fuel_candidates)} Fuel thresholds (Tail Range: {fuel_candidates.min():.4f} - {fuel_candidates.max():.4f})")

best_score = -1
best_params = {}

print(f"\nRunning Grid Search (Target: Beta={OPTIMIZATION_BETA}, Min Prec={MIN_PRECISION_FLOOR})...")

for market_thresh in fuel_candidates:
    for gpr_thresh in gpr_candidates:
        
        # OR Logic: Alarm if EITHER GPR > gpr_thresh OR Market > market_thresh
        y_pred = ((gpr_probs_val > gpr_thresh) | (market_fragility_probs_val > market_thresh)).astype(int)

        # Calculate F-Beta with Precision bias
        f_beta, prec, rec, hits, fas = calculate_event_f_beta(
            y_true_val, y_pred, val_event_groups, beta=OPTIMIZATION_BETA
        )

        # [CRITICAL CHANGE] Stricter Precision Floor
        if prec < MIN_PRECISION_FLOOR: 
            continue

        if f_beta > best_score:
            best_score = f_beta
            best_params = {
                'market': market_thresh,
                'gpr': gpr_thresh,
                'score': f_beta,
                'prec': prec,
                'rec': rec,
                'hits': hits,
                'fas': fas
            }

# --- 6. REPORT AND SAVE ---

print("\n" + "="*40)
if best_score == -1:
    print("Optimization Failed: No combination met minimum precision requirements.")
    print("Consider lowering MIN_PRECISION_FLOOR or checking model quality.")
    # Safe default fallback (High percentiles)
    best_params = {
        'market': np.percentile(market_fragility_probs_val, 95), 
        'gpr': np.percentile(gpr_probs_val, 95)
    }
else:
    print(f"OPTIMIZATION SUCCESS (High Precision Mode)")
    print(f"Best F({OPTIMIZATION_BETA})-Score: {best_params['score']:.4f}")
    print(f"Recall: {best_params['rec']:.2%} ({best_params['hits']} events captured)")
    print(f"Precision: {best_params['prec']:.2%} ({best_params['fas']} false alarms)")
    print("-" * 20)
    print(f"GPR Panic Threshold: {best_params['gpr']:.4f}")
    print(f"Market Critical Threshold: {best_params['market']:.4f}")

# Update Config
config_path = "config.py"
with open(config_path, "r") as f:
    content = f.read()

# Regex substitution
content = re.sub(r"(GPR_PANIC_THRESHOLD\s*=\s*)[\d\.]+", f"\\g<1>{best_params['gpr']:.4f}", content)
content = re.sub(r"(MARKET_CRITICAL_THRESHOLD\s*=\s*)[\d\.]+", f"\\g<1>{best_params['market']:.4f}", content)

with open(config_path, "w") as f:
    f.write(content)
    
print("\nconfig.py successfully updated with optimized values.")