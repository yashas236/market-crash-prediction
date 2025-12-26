import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib
import os
import random
import tensorflow as tf

# --- SET RANDOM SEED FOR REPRODUCIBILITY ---
SEED_VALUE = 42
os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

print("Starting Phase: Training Market Fragility SVM Model (The 'Fuel')...")

# --- 1. SET PARAMETERS ---
WARNING_WINDOW_DAYS = 10 
# These are the features that will determine "market fragility"
# We are adding credit-based indicators to make the model more robust.
MARKET_FEATURES = [
    'VIX_Close', 
    'RSI_14',
    'Rolling_1st_Percentile' # A measure of recent downside volatility
]
TARGET_COLUMN = 'Crash_Event'

# --- 2. Helper Function to create warning labels ---
def create_warning_labels(y_true, window_size):
    y_warning = np.zeros_like(y_true)
    true_event_indices = np.where(y_true == 1)[0]
    if not true_event_indices.any(): return y_warning
    event_groups = []
    current_event_start = true_event_indices[0]
    for i in range(1, len(true_event_indices)):
        if true_event_indices[i] > true_event_indices[i-1] + 1:
            event_groups.append(current_event_start)
            current_event_start = true_event_indices[i]
    event_groups.append(current_event_start) 
    for event_start_day in event_groups:
        warning_start = max(0, event_start_day - window_size)
        warning_end = event_start_day
        y_warning[warning_start : warning_end] = 1
    y_warning[y_true == 1] = 0 # Don't train on the crash day itself
    return y_warning

# --- 3. Load Datasets ---
try:
    train_df = pd.read_csv("data/train_final.csv", index_col='Date', parse_dates=True)
    test_df = pd.read_csv("data/test_final.csv", index_col='Date', parse_dates=True)
except FileNotFoundError:
    print("Error: Data files not found. Make sure 'data/train_final.csv' and 'data/test_final.csv' exist.")
    exit()
print("Loaded train and test datasets.")

# --- 4. Prepare Data for Modeling ---
X_train = train_df[MARKET_FEATURES]
y_train_raw = train_df[TARGET_COLUMN]
X_test = test_df[MARKET_FEATURES]
y_test_raw = test_df[TARGET_COLUMN]

# --- 5. Create Warning-Based Target (Y) ---
# We train the SVM to recognize the fragile period *before* a crash.
y_train = create_warning_labels(y_train_raw.values, WARNING_WINDOW_DAYS)
y_test = create_warning_labels(y_test_raw.values, WARNING_WINDOW_DAYS)
print(f"Training on {np.sum(y_train)} total 'fragility' days.")

# --- 6. Normalize Features ---
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Market features normalized.")

# --- 7. Train and Save SVM "Fuel" Model ---
print("\nStarting Market SVM ('Fuel' Model) Training...")
# Define the SVM with balanced class weights, as fragile periods are rare.
svm_fuel_model = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=SEED_VALUE)

# Train the model
svm_fuel_model.fit(X_train_scaled, y_train)

# Save the model and the scaler
joblib.dump(svm_fuel_model, "svm_fuel_model.pkl")
joblib.dump(scaler, "market_features_scaler.pkl")
print("Trained SVM 'Fuel' model saved as 'svm_fuel_model.pkl'")
print("Market feature scaler saved as 'market_features_scaler.pkl'")

# --- 8. Evaluate SVM ---
print("\nEvaluating SVM 'Fuel' Model on Test Data...")
y_pred_svm = svm_fuel_model.predict(X_test_scaled)
print("SVM 'Fuel' Model Classification Report:")
# We evaluate against the same 'warning' labels, not the raw crash event
print(classification_report(y_test, y_pred_svm, zero_division=0))

print("\nPhase complete.")