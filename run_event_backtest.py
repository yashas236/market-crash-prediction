import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import warnings
import joblib
import shap
import config
import utils

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# We will also ignore the specific runtime warnings from SHAP
warnings.filterwarnings("ignore", category=RuntimeWarning, module="shap")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

print("Starting Full Backtest (Gated Spark & Fuel Model)...")

# --- 1. SET PARAMETERS ---
WARNING_WINDOW_DAYS = config.WARNING_WINDOW_DAYS
LOOKBACK_DAYS = config.LOOKBACK_DAYS

# --- 2. Load Model and Test Data (from local folder) ---
try:
    # --- Load the "Spark" model (LSTM trained on GPR) ---
    model = load_model(config.MODEL_SPARK) 
    y_test_seq = np.load("y_test_seq_gpr.npy") 
    X_test_seq = np.load("X_test_seq_gpr.npy") 
    
    # --- Load the "Fuel" model (SVM trained on Market Fragility) ---
    svm_fuel_model = joblib.load(config.MODEL_FUEL)
    market_scaler = joblib.load(config.SCALER_MARKET)
    
    # Load dates and market data for SVM
    df_test = pd.read_csv(config.DATA_TEST, index_col="Date", parse_dates=True)
    test_dates = df_test.index[LOOKBACK_DAYS:]

    X_market_test = df_test[config.MARKET_FEATURES]

except FileNotFoundError:
    print("Error: Model or data files not found. Run train_lstm_model.py and train_market_svm.py first.")
    exit()
print("Loaded GPR-LSTM ('Spark') and Market-SVM ('Fuel') models and test data.")

# --- 3. Get Model Predictions (Gated Logic) ---
print("Generating gated predictions...")

# 1. Get the "Spark" signal (LSTM, trained ONLY on GPR)
gpr_prob = model.predict(X_test_seq).flatten()

# 2. Get the "Fuel" signal (SVM, trained ONLY on VIX/Market)
# We need to align the market data with the LSTM sequences
X_market_test_aligned = X_market_test.iloc[LOOKBACK_DAYS:]
X_market_test_scaled = market_scaler.transform(X_market_test_aligned)
market_fragility_prob = svm_fuel_model.predict_proba(X_market_test_scaled)[:, 1]

# 3. The "Gated" Signal - Using Optimized Thresholds
# These values were determined by running `optimize_gate.py` on the validation set.
FRAGILITY_THRESHOLD = config.FRAGILITY_THRESHOLD
HIGH_SENS_GPR_THRESHOLD = config.HIGH_SENS_GPR_THRESHOLD
LOW_SENS_GPR_THRESHOLD = config.LOW_SENS_GPR_THRESHOLD

y_pred_binary = []
dynamic_thresholds = []
for gpr_p, fragility in zip(gpr_prob, market_fragility_prob):
    # If Market Fragility is HIGH, use a lower (more sensitive) threshold for the GPR signal.
    threshold = HIGH_SENS_GPR_THRESHOLD if fragility > FRAGILITY_THRESHOLD else LOW_SENS_GPR_THRESHOLD
    dynamic_thresholds.append(threshold)
    y_pred_binary.append(1 if gpr_p > threshold else 0)

y_pred_binary = np.array(y_pred_binary)
dynamic_thresholds = np.array(dynamic_thresholds)
# For ROC AUC, we will use the original GPR probability, as the gated logic is a binary rule.
y_pred_proba = gpr_prob

# --- Save predictions for visualization script ---
np.save("y_pred_proba.npy", y_pred_proba)
np.save("y_pred_fuel.npy", market_fragility_prob)
np.save("y_pred_binary.npy", y_pred_binary)
np.save("dynamic_thresholds.npy", dynamic_thresholds)
print("Saved prediction probabilities to 'y_pred_proba.npy'")

y_true = y_test_seq

# --- 4. Evaluate Daily Performance (Identical) ---
print("\n--- Daily Performance Evaluation ---")
print("\nDaily Classification Report:")
print(classification_report(y_true, y_pred_binary, target_names=['No Crash', 'Crash'], zero_division=0))
print("\nDaily Area Under the ROC Curve (AUROC):")
print(f"AUROC: {roc_auc_score(y_true, y_pred_proba):.4f}")
print("\nDaily Confusion Matrix:"); print(confusion_matrix(y_true, y_pred_binary))

# --- 5. Find Ground-Truth Crash Events (Using Utils) ---
event_groups = utils.group_crash_events(y_true)
if not event_groups: 
    print("No crash events found in test data.")
    exit()
print(f"\nFound {len(event_groups)} distinct crash events in the test data.")

# --- 6. Evaluate Event-Based Hits and Misses (Identical) ---
events_hit, events_missed, hit_event_windows = 0, 0, []
event_results_table = []

# Known crash reasons are now imported from config
for start_day, end_day in event_groups:
    warning_start, warning_end = max(0, start_day - WARNING_WINDOW_DAYS), start_day - 1
    hit = False
    if warning_start <= warning_end and np.sum(y_pred_binary[warning_start : warning_end + 1]) > 0:
        events_hit += 1; hit = True
    if not hit: events_missed += 1
    else: hit_event_windows.append((warning_start, end_day))
    
    # Record event details for the table
    s_date = test_dates[start_day]
    e_date = test_dates[end_day]
    
    reason = "Other / Unclassified"
    for k_start, k_end, k_name in config.KNOWN_CRASHES:
        if (s_date <= k_end) and (e_date >= k_start):
            reason = k_name
            break
            
    status = "Hit (Predicted)" if hit else "Miss (Not Predicted)"
    event_results_table.append({
        "Start Date": s_date.strftime('%Y-%m-%d'), 
        "End Date": e_date.strftime('%Y-%m-%d'), 
        "Reason": reason,
        "Status": status
    })

# --- 7. Evaluate Event-Based False Alarms (Identical) ---
pred_alarm_indices = np.where(y_pred_binary == 1)[0]
false_alarm_count = 0
false_alarm_start_dates = []

if pred_alarm_indices.any():
    allowed_alarm_mask = np.zeros_like(y_true)
    for start, end in hit_event_windows: allowed_alarm_mask[start : end + 1] = 1
    allowed_alarm_mask[y_true == 1] = 1
    false_alarm_indices = [i for i in pred_alarm_indices if allowed_alarm_mask[i] == 0]
    
    if false_alarm_indices:
        # Load test dates to log the exact date of the false alarm
        current_alarm_start = false_alarm_indices[0]
        for i in range(1, len(false_alarm_indices)):
            if false_alarm_indices[i] > false_alarm_indices[i-1] + 1:
                false_alarm_count += 1
                false_alarm_start_dates.append(test_dates[current_alarm_start].strftime('%Y-%m-%d'))
                current_alarm_start = false_alarm_indices[i]
        false_alarm_count += 1
        false_alarm_start_dates.append(test_dates[current_alarm_start].strftime('%Y-%m-%d'))

# --- 8. Report Event-Based Metrics (Identical) ---
print("\n--- Event-Based Backtest Results ---")
print(f"Warning Window: {WARNING_WINDOW_DAYS} days | Gated Threshold Logic")
print("-" * 40)
print(f"Total Actual Crash Events: {len(event_groups)}")
print(f"Events Predicted Early (Hits): {events_hit}")
print(f"Events Missed: {events_missed}")
print(f"Event-Based Recall (Hit Rate): {events_hit / len(event_groups):.2%}")
print(f"Total False Alarm Events: {false_alarm_count}")
print("-" * 40)
print("\n--- Detailed Event Prediction Table ---")
df_events_report = pd.DataFrame(event_results_table)
print(df_events_report.to_string(index=False))
print("-" * 40)

if false_alarm_start_dates:
    print("\n--- Detailed False Alarm Analysis ---")
    false_alarm_report_data = []
    for date_str in false_alarm_start_dates:
        date = pd.Timestamp(date_str)
        found = False
        for start, end, event, reason in config.KNOWN_FALSE_ALARMS:
            if start <= date <= end:
                false_alarm_report_data.append({
                    "Date Range (Approx)": f"{start.strftime('%Y-%m-%d')} / {end.strftime('%Y-%m-%d')}",
                    "Event (The \"Spark\")": event,
                    "Why No Crash? (The \"Firewall\")": reason
                })
                found = True
                break
    
    df_false_alarms = pd.DataFrame(false_alarm_report_data)
    # To prevent duplicate entries for the same event period
    df_false_alarms.drop_duplicates(subset=["Event (The \"Spark\")"], inplace=True)
    print(df_false_alarms.to_string(index=False))

# --- 9. Model Interpretability (SHAP Analysis) ---
print("\nStarting SHAP analysis... (This will take a long time)")

try:
    # --- 12-FEATURE LIST ---
    N_FEATURES = len(config.GPR_FEATURES)
    
    # 1. Reshape our 3D data to 2D for SHAP
    X_test_2d = X_test_seq.reshape(X_test_seq.shape[0], LOOKBACK_DAYS * N_FEATURES)

    # --- THIS IS THE FIX ---
    print("Cleaning data for SHAP (replacing inf/nan with 0)...")
    X_test_2d = np.nan_to_num(X_test_2d, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    print("Data cleaned.")
    # --- END FIX ---

    # 2. Summarize the 2D data.
    # Use 100 samples for a stable result
    background_data_2d = shap.kmeans(X_test_2d[np.random.choice(X_test_2d.shape[0], 20, replace=False)], 20)
    
    # 3. Create the prediction function
    def model_predict_proba(data_2d):
        data_3d = data_2d.reshape(-1, LOOKBACK_DAYS, N_FEATURES)
        return model.predict(data_3d, verbose=0)
        
    # 4. Create the explainer
    explainer = shap.KernelExplainer(model_predict_proba, background_data_2d)
    
    # 5. Explain a 2D sample (Use 100 samples for a stable result)
    samples_to_explain_2d = X_test_2d[np.random.choice(X_test_2d.shape[0], 20, replace=False)]
    
    shap_values_2d = explainer.shap_values(samples_to_explain_2d, nsamples="auto") 
    
    # 6. Reshape the SHAP values back to 3D for analysis
    shap_values_3d = shap_values_2d[0].reshape(-1, LOOKBACK_DAYS, N_FEATURES)
    
    # 7. Manually plot the global feature importance
    axes_to_average = tuple(range(shap_values_3d.ndim - 1))
    global_importance = np.mean(np.abs(shap_values_3d), axis=axes_to_average)
    
    print(f"Calculated global importance: {global_importance}")
    
    feature_importance_series = pd.Series(
        global_importance, 
        index=config.GPR_FEATURES
    )
    feature_importance_series = feature_importance_series.sort_values(ascending=True)
    
    plt.figure(figsize=(10, 6))
    feature_importance_series.plot(kind='barh', color='skyblue')
    plt.title('Global Feature Importance (Mean Absolute SHAP Value)')
    plt.xlabel('Mean SHAP Value (Impact on model output)')
    plt.savefig("shap_summary.png", bbox_inches='tight')
    print("Saved 'shap_summary.png'")
    
except Exception as e:
    print(f"Error generating SHAP plot: {e}")

print("\nPhase 5 complete.")