import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import random
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.svm import SVC
import joblib
from sklearn.metrics import classification_report
import config
import utils


# --- SET RANDOM SEED FOR REPRODUCIBILITY ---
import tensorflow as tf
import os
# --- SET RANDOM SEED FOR REPRODUCIBILITY ---
SEED_VALUE = 42 # <-- FIXED SEED
os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)
# --- END SEED BLOCK ---

print("Starting Phase 4: Training EWS Model (GPR Features Only)...")

# --- 1. SET PARAMETERS ---
LOOKBACK_DAYS = 30
WARNING_WINDOW_DAYS = 10 
EPOCHS = 100 
PATIENCE = 20

# --- 3. Load Final Datasets (from local folder) ---
try:
    train_df = pd.read_csv("data/train_final.csv", index_col='Date', parse_dates=True)
    val_df = pd.read_csv("data/validation_final.csv", index_col='Date', parse_dates=True)
    test_df = pd.read_csv("data/test_final.csv", index_col='Date', parse_dates=True)
except FileNotFoundError:
    print("Error: Final data files not found. (Did you copy the 'data' folder to this new directory?)")
    exit()
print("Loaded train, validation, and test datasets.")

# --- 4. Prepare Data for Modeling (GPR-ONLY FEATURE LIST) ---
feature_columns = [
    'GPR_Composite', 'GPR_Threats', 'GPR_Acts',
    'GPR_Threats_MA_63', 'GPR_Threats_MA_126', 'GPR_Threats_Trend', 'GPR_Acts_MA_63',
    'GPR_Threats_1D_PctChange', 'GPR_Acts_1D_PctChange', 'GPR_Threats_5D_PctChange',
    'GPR_Threats_Vol_21D', 'GPR_Acts_Vol_21D'
]
target_column = 'Crash_Event'
# (Rest of data prep is identical)
X_train = train_df[feature_columns]
y_train_raw = train_df[target_column]
X_val = val_df[feature_columns]
y_val_raw = val_df[target_column]
X_test = test_df[feature_columns]
y_test = test_df[target_column] 

# --- 5. Create New Warning-Based Target (Y) (Identical) ---
y_train = utils.create_warning_labels(y_train_raw.values, WARNING_WINDOW_DAYS)
y_val = utils.create_warning_labels(y_val_raw.values, WARNING_WINDOW_DAYS)
print(f"Training on {np.sum(y_train)} total 'warning' days.")

# --- 6. Normalize Features (Identical) ---
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
joblib.dump(scaler, config.SCALER_GPR)
print(f"GPR feature scaler saved as '{config.SCALER_GPR}'")
X_test_scaled = scaler.transform(X_test)
print("Features normalized.")

# --- 7. Create Sequences (Identical) ---
def create_sequences(X, y, lookback_period):
    X_sequences, y_sequences = [], []
    for i in range(len(X) - lookback_period):
        X_sequences.append(X[i:(i + lookback_period)])
        y_sequences.append(y[i + lookback_period])
    return np.array(X_sequences), np.array(y_sequences)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, LOOKBACK_DAYS)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, LOOKBACK_DAYS)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, LOOKBACK_DAYS)
print(f"Created sequences with lookback of {LOOKBACK_DAYS} days.")

# --- 8. Calculate Class Weights (Identical) ---
print("Calculating class weights...")
if len(np.unique(y_train_seq)) > 1:
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train_seq), y=y_train_seq)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"Using class weights: {class_weight_dict}")
else:
    class_weight_dict = None

# --- 9. Build the STACKED Bi-LSTM Model ---
model = Sequential()
model.add(Bidirectional(LSTM( # <-- Wrap the first layer
    units=50, 
    return_sequences=True, 
    input_shape=(LOOKBACK_DAYS, len(feature_columns))
)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False)) # The second layer can stay the same
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

# --- 10. Compile the Model (Identical) ---
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Model compiled.")

# --- 11. Train the Model with Early Stopping (Identical) ---
print("\nStarting model training with Early Stopping...")
early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1, mode='min', restore_best_weights=True)
history = model.fit(
    X_train_seq, y_train_seq,
    epochs=EPOCHS,
    batch_size=32,
    validation_data=(X_val_seq, y_val_seq),
    class_weight=class_weight_dict,
    callbacks=[early_stopping], 
    verbose=1
)
print("Model training complete.")

# --- 12. Save Model & Test Data (local) ---
model.save("trained_ews_model_gpr.h5")
np.save("X_test_seq_gpr.npy", X_test_seq)
np.save("y_test_seq_gpr.npy", y_test_seq)
print("\nTrained model saved as 'trained_ews_model_gpr.h5'")

# --- 12. Plot Training Curves (Identical) ---
print("\nGenerating training history plots...")
try:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(history.history['loss'], label='Training Loss'); plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2); plt.plot(history.history['accuracy'], label='Training Accuracy'); plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig("training_history_gpr.png")
    print("Saved training history plot to 'training_history_gpr.png'")
except Exception as e:
    print(f"Error generating plot: {e}")
print("\nPhase 4 complete.")
