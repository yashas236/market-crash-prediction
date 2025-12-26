import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import warnings
import joblib
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import re
import config

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print("Starting Step 1: Gating Threshold Optimization...")

# --- 1. SET PARAMETERS ---
LOOKBACK_DAYS = config.LOOKBACK_DAYS
WARNING_WINDOW_DAYS = config.WARNING_WINDOW_DAYS

# --- 2. LOAD MODELS AND VALIDATION DATA ---
try:
    # Load Spark (LSTM) model
    spark_model = load_model(config.MODEL_SPARK)

    # Load Fuel (SVM) model and its scaler
    fuel_model = joblib.load(config.MODEL_FUEL)
    market_scaler = joblib.load(config.SCALER_MARKET)
    gpr_scaler = joblib.load(config.SCALER_GPR)

    # Load validation data
    df_val = pd.read_csv(config.DATA_VAL, index_col="Date", parse_dates=True)

    # Define feature sets
    # Imported from config

except FileNotFoundError as e:
    print(f"Error loading files: {e}. Make sure all models and data are in the correct directories.")
    exit()

print("Loaded models and validation data.")

# --- 3. PREPARE DATA AND GENERATE BASE PREDICTIONS ---

# 1. GPR/Spark Predictions
# We need a scaler for the GPR data. We'll fit it on the training data to be consistent.

X_gpr_val_scaled = gpr_scaler.transform(df_val[config.GPR_FEATURES])

def create_sequences(X, lookback_period):
    X_sequences = []
    for i in range(len(X) - lookback_period):
        X_sequences.append(X[i:(i + lookback_period)])
    return np.array(X_sequences)

X_gpr_val_seq = create_sequences(X_gpr_val_scaled, LOOKBACK_DAYS)
gpr_probs_val = spark_model.predict(X_gpr_val_seq).flatten()

# 2. Market/Fuel Predictions
X_market_val = df_val[config.MARKET_FEATURES]
X_market_val_scaled = market_scaler.transform(X_market_val)
market_fragility_probs_val = fuel_model.predict_proba(X_market_val_scaled)[:, 1]

# 3. Align data
y_true_val = df_val['Crash_Event'].values[LOOKBACK_DAYS:]
# The market fragility probabilities don't need sequencing, just alignment
market_fragility_probs_val_aligned = market_fragility_probs_val[LOOKBACK_DAYS:]

print("Generated base predictions on validation data.")

# --- 3.5. VISUALIZE PREDICTION DISTRIBUTIONS ---
print("\nVisualizing prediction distributions...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Histogram for Spark (GPR) probabilities
ax1.hist(gpr_probs_val, bins=50, color='skyblue', edgecolor='black')
ax1.set_title("Distribution of 'Spark' (GPR-LSTM) Probabilities")
ax1.set_xlabel("Predicted Probability")
ax1.set_ylabel("Frequency")
ax1.axvline(np.mean(gpr_probs_val), color='r', linestyle='--', label=f'Mean: {np.mean(gpr_probs_val):.3f}')
ax1.legend()

# Histogram for Fuel (Market Fragility) probabilities
ax2.hist(market_fragility_probs_val_aligned, bins=50, color='salmon', edgecolor='black')
ax2.set_title("Distribution of 'Fuel' (Market-SVM) Probabilities")
ax2.set_xlabel("Predicted Probability")
ax2.set_ylabel("Frequency")
ax2.axvline(np.mean(market_fragility_probs_val_aligned), color='r', linestyle='--', label=f'Mean: {np.mean(market_fragility_probs_val_aligned):.3f}')
ax2.legend()

plt.tight_layout()
plt.savefig("prediction_distributions_validation.png")
print("Saved prediction distribution plot to 'prediction_distributions_validation.png'")

# --- 4. DEFINE EVENT-BASED METRIC CALCULATION ---
# Pre-calculate validation event groups once to speed up the loop
val_event_indices = np.where(y_true_val == 1)[0]
val_event_groups = []
if val_event_indices.any():
    current_event_start = val_event_indices[0]
    for i in range(1, len(val_event_indices)):
        if val_event_indices[i] > val_event_indices[i-1] + 1:
            val_event_groups.append(current_event_start)
            current_event_start = val_event_indices[i]
    val_event_groups.append(current_event_start)

def calculate_event_f1_fast(y_true, y_pred, event_groups):
    if not event_groups: return 0, 0, 0, 0, 0
    
    total_events = len(event_groups)
    events_hit = 0
    hit_event_windows = []

    # Evaluate Hits
    for start_day in event_groups:
        warning_start = max(0, start_day - WARNING_WINDOW_DAYS)
        warning_end = start_day - 1
        if warning_start <= warning_end and np.sum(y_pred[warning_start : warning_end + 1]) > 0:
            events_hit += 1
            hit_event_windows.append((warning_start, start_day))

    # Evaluate False Alarms
    pred_alarm_indices = np.where(y_pred == 1)[0]
    false_alarms = 0
    if pred_alarm_indices.any():
        # Create a mask of "allowed" alarm periods
        allowed_alarm_mask = np.zeros_like(y_true)
        for start, end in hit_event_windows:
            allowed_alarm_mask[start : end + 1] = 1
        allowed_alarm_mask[y_true == 1] = 1

        # Fast boolean indexing
        false_alarm_indices = pred_alarm_indices[allowed_alarm_mask[pred_alarm_indices] == 0]
        
        if len(false_alarm_indices) > 0:
            false_alarms = 1 + np.sum(np.diff(false_alarm_indices) > 1)

    # Calculate metrics
    recall = events_hit / total_events if total_events > 0 else 0
    precision = events_hit / (events_hit + false_alarms) if (events_hit + false_alarms) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1_score, precision, recall, events_hit, false_alarms

# --- 5. OPTIMIZATION LOOP ---
print("\nSearching for optimal thresholds (Vectorized & Fine-Grained)...")

# Finer grid steps (0.01) for better precision, enabled by vectorized speed
fragility_thresholds = np.arange(0.04, 0.25, 0.01)      
high_sens_thresholds = np.arange(0.05, 0.40, 0.01)      
low_sens_thresholds = np.arange(0.25, 0.70, 0.01)       

best_f1 = -1
best_params = {}

total_iterations = len(fragility_thresholds) * len(high_sens_thresholds) * len(low_sens_thresholds)
print(f"Evaluating {total_iterations} combinations...")

for f_thresh in fragility_thresholds:
    for hs_thresh in high_sens_thresholds:
        for ls_thresh in low_sens_thresholds:
            # --- LOGIC ENFORCEMENT ---
            # Force a meaningful gap (e.g., 15%) between the two states.
            # If the thresholds are too close, the "Hybrid" logic is redundant.
            if (ls_thresh - hs_thresh) < 0.15:
                continue

            # Vectorized Gating Logic (100x faster than loop)
            # Create array of thresholds based on fragility
            thresholds = np.where(market_fragility_probs_val_aligned > f_thresh, hs_thresh, ls_thresh)
            # Compare GPR probabilities to these thresholds
            y_pred_binary = (gpr_probs_val > thresholds).astype(int)

            # Calculate F1 score
            f1, precision, recall, hits, false_alarms = calculate_event_f1_fast(y_true_val, y_pred_binary, val_event_groups)

            if f1 > best_f1:
                best_f1 = f1
                best_params = {
                    'fragility_threshold': f_thresh,
                    'high_sensitivity_gpr_threshold': hs_thresh,
                    'low_sensitivity_gpr_threshold': ls_thresh,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'hits': hits,
                    'false_alarms': false_alarms
                }

# --- 6. REPORT RESULTS ---
print("\n--- Optimization Complete ---")
if best_f1 > -1:
    print("Best Parameters Found:")
    print(f"  Fragility Threshold: {best_params['fragility_threshold']:.2f}")
    print(f"  High-Sensitivity GPR Threshold: {best_params['high_sensitivity_gpr_threshold']:.2f}")
    print(f"  Low-Sensitivity GPR Threshold: {best_params['low_sensitivity_gpr_threshold']:.2f}")
    print("-" * 30)
    print(f"Validation Event-Based F1-Score: {best_params['f1_score']:.2%}")
    print(f"Validation Event-Based Precision: {best_params['precision']:.2%}")
    print(f"Validation Event-Based Recall: {best_params['recall']:.2%}")
    print(f"Hits / False Alarms on Validation Set: {best_params['hits']} / {best_params['false_alarms']}")

    # --- 7. UPDATE CONFIG.PY ---
    print("\nUpdating config.py with new thresholds...")
    
    config_path = "config.py"
    with open(config_path, "r") as f:
        content = f.read()
    
    # Update Fragility Threshold
    content = re.sub(
        r"(FRAGILITY_THRESHOLD\s*=\s*)[\d\.]+", 
        f"\\g<1>{best_params['fragility_threshold']:.2f}", 
        content
    )
    
    # Update High Sens Threshold
    content = re.sub(
        r"(HIGH_SENS_GPR_THRESHOLD\s*=\s*)[\d\.]+", 
        f"\\g<1>{best_params['high_sensitivity_gpr_threshold']:.2f}", 
        content
    )
    
    # Update Low Sens Threshold
    content = re.sub(
        r"(LOW_SENS_GPR_THRESHOLD\s*=\s*)[\d\.]+", 
        f"\\g<1>{best_params['low_sensitivity_gpr_threshold']:.2f}", 
        content
    )
    
    with open(config_path, "w") as f:
        f.write(content)
        
    print(f"Successfully updated {config_path} with optimized values.")

else:
    print("No suitable parameters found. You may need to adjust the search ranges.")

print("\nStep 1 complete.")