import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import random
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, Concatenate, Activation, Dot, Flatten, Softmax
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
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
LOOKBACK_DAYS = config.LOOKBACK_DAYS
WARNING_WINDOW_DAYS = config.WARNING_WINDOW_DAYS 
EPOCHS = 100 
PATIENCE = 20

# --- 2. Load Final Datasets (from local folder) ---
try:
    train_df = pd.read_csv(config.DATA_TRAIN, index_col='Date', parse_dates=True)
    val_df = pd.read_csv(config.DATA_VAL, index_col='Date', parse_dates=True)
    test_df = pd.read_csv(config.DATA_TEST, index_col='Date', parse_dates=True)
except FileNotFoundError:
    print("Error: Final data files not found. (Did you copy the 'data' folder to this new directory?)")
    exit()
print("Loaded train, validation, and test datasets.")

# --- 3. Prepare Data for Modeling (GPR-ONLY FEATURE LIST) ---
feature_columns = config.GPR_FEATURES
target_column = 'Crash_Event'
# (Rest of data prep is identical)
X_train = train_df[feature_columns]
y_train_raw = train_df[target_column]
X_val = val_df[feature_columns]
y_val_raw = val_df[target_column]
X_test = test_df[feature_columns]
y_test = test_df[target_column] 

# --- 4. Create New Warning-Based Target (Y) (Using Utils) ---
y_train = utils.create_warning_labels(y_train_raw.values, WARNING_WINDOW_DAYS)
y_val = utils.create_warning_labels(y_val_raw.values, WARNING_WINDOW_DAYS)
print(f"Training on {np.sum(y_train)} total 'warning' days.")

# --- 5. Normalize Features & Save Scaler ---
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
joblib.dump(scaler, config.SCALER_GPR)
print(f"GPR feature scaler saved as '{config.SCALER_GPR}'")
X_test_scaled = scaler.transform(X_test)
print("Features normalized.")

# --- 6. Create Sequences (Identical) ---
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

# --- 7. Calculate Class Weights (Identical) ---
print("Calculating class weights...")
if len(np.unique(y_train_seq)) > 1:
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train_seq), y=y_train_seq)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"Using class weights: {class_weight_dict}")
else:
    class_weight_dict = None

# --- 8. Build the STACKED Bi-LSTM Model ---
def build_attention_model(input_shape):
    inputs = Input(shape=input_shape)

    # 1. Bi-Directional LSTM Layer
    # Returns (Batch, Time, 128) because units=64 and it's bidirectional
    lstm_out = Bidirectional(LSTM(units=64, return_sequences=True))(inputs)
    lstm_out = Dropout(0.3)(lstm_out)

    # 2. Attention Mechanism
    # Compute a score for every time step: (Batch, Time, 1)
    attention_score = Dense(1, activation='tanh')(lstm_out)
    
    # Convert scores to probabilities summing to 1 over the Time axis (axis=1)
    attention_weights = Softmax(axis=1)(attention_score)

    # 3. Context Vector
    # Weighted sum of LSTM outputs. 
    # Dot(axes=1) sums over the Time dimension.
    # Shape becomes (Batch, 1, 128)
    context_vector = Dot(axes=1)([attention_weights, lstm_out])

    # Flatten to remove the '1' dimension, resulting in (Batch, 128)
    context_vector = Flatten()(context_vector)

    # 4. Classification Head
    x = Dense(32, activation='relu')(context_vector)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Initialize the model
input_shape = (LOOKBACK_DAYS, len(feature_columns))
model = build_attention_model(input_shape)

model.summary()

# --- 9. Compile the Model (Identical) ---
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Model compiled.")

# --- 10. Train the Model with Early Stopping (Identical) ---
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

# --- 11. Save Model & Test Data (local) ---
model.save(config.MODEL_SPARK)
np.save(config.SEQ_X_TEST, X_test_seq)
np.save(config.SEQ_Y_TEST, y_test_seq)
print(f"\nTrained model saved as '{config.MODEL_SPARK}'")

# --- 12. Plot Training Curves ---
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
