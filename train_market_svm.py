import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib
import os
import random
import tensorflow as tf
import config
import utils

# --- SET RANDOM SEED FOR REPRODUCIBILITY ---
SEED_VALUE = 42
os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

print("Starting Phase: Training Market Fragility SVM Model (The 'Fuel')...")

# --- 1. SET PARAMETERS ---
WARNING_WINDOW_DAYS = config.WARNING_WINDOW_DAYS
TARGET_COLUMN = 'Crash_Event'

# --- 3. Load Datasets ---
try:
    train_df = pd.read_csv(config.DATA_TRAIN, index_col='Date', parse_dates=True)
    test_df = pd.read_csv(config.DATA_TEST, index_col='Date', parse_dates=True)
except FileNotFoundError:
    print("Error: Data files not found. Make sure 'data/train_final.csv' and 'data/test_final.csv' exist.")
    exit()
print("Loaded train and test datasets.")

# --- 4. Prepare Data for Modeling ---
X_train = train_df[config.MARKET_FEATURES]
y_train_raw = train_df[TARGET_COLUMN]
X_test = test_df[config.MARKET_FEATURES]
y_test_raw = test_df[TARGET_COLUMN]

# --- 5. Create Warning-Based Target (Y) ---
# We train the SVM to recognize the fragile period *before* a crash.
y_train = utils.create_warning_labels(y_train_raw.values, WARNING_WINDOW_DAYS)
y_test = utils.create_warning_labels(y_test_raw.values, WARNING_WINDOW_DAYS)
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
joblib.dump(svm_fuel_model, config.MODEL_FUEL)
joblib.dump(scaler, config.SCALER_MARKET)
print(f"Trained SVM 'Fuel' model saved as '{config.MODEL_FUEL}'")
print(f"Market feature scaler saved as '{config.SCALER_MARKET}'")

# --- 8. Evaluate SVM ---
print("\nEvaluating SVM 'Fuel' Model on Test Data...")
y_pred_svm = svm_fuel_model.predict(X_test_scaled)
print("SVM 'Fuel' Model Classification Report:")
# We evaluate against the same 'warning' labels, not the raw crash event
print(classification_report(y_test, y_pred_svm, zero_division=0))

print("\nPhase complete.")