import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import config

print("Generating backtest visualization...")

# --- 1. Load Data ---
try:
    # Load the complete test dataset to get the price data
    df_test = pd.read_csv(config.DATA_TEST, index_col="Date", parse_dates=True)
    
    # Load the sequences used for prediction
    y_pred_proba = np.load("y_pred_proba.npy")
    y_pred_fuel = np.load("y_pred_fuel.npy")
    y_pred_binary = np.load("y_pred_binary.npy")
    y_true = np.load("y_test_seq_gpr.npy")
    dynamic_thresholds = np.load("dynamic_thresholds.npy")

except FileNotFoundError as e:
    print(f"Error loading data files: {e}")
    print("Please run 'run_event_backtest.py' first to generate the prediction files.")
    exit()

# Align the predictions with the correct dates from the test set
# The first prediction corresponds to the end of the first lookback period
LOOKBACK_DAYS = config.LOOKBACK_DAYS
prediction_dates = df_test.index[LOOKBACK_DAYS:]

if len(prediction_dates) != len(y_pred_proba):
    print("Error: Mismatch between prediction length and date range.")
    exit()

df_results = pd.DataFrame({
    'close': df_test['close'].loc[prediction_dates],
    'y_pred_proba': y_pred_proba,
    'y_pred_fuel': y_pred_fuel,
    'y_pred_binary': y_pred_binary,
    'dynamic_threshold': dynamic_thresholds,
    'y_true': y_true
})

# --- 2. Create the Plot ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 15), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})

# --- PANEL 1: S&P 500 Price & Events ---
ax1.plot(df_results.index, df_results['close'], color='black', label='S&P 500 Close', linewidth=1.5)
ax1.set_ylabel('S&P 500 Close Price', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_yscale('log')
ax1.set_title('S&P 500 Price & Geopolitical Events')

# --- Define Known Geopolitical Events for Highlighting ---
# Imported from config

# Shade the ground-truth crash events with distinction
crash_label_added = False
geo_label_added = False

for _, crash_period in df_results[df_results['y_true'] == 1].groupby((df_results['y_true'] != df_results['y_true'].shift()).cumsum()):
    start_date = crash_period.index[0]
    end_date = crash_period.index[-1]
    
    # Check if this crash overlaps with a known geopolitical event
    is_geopolitical = False
    for geo_start, geo_end in config.GEOPOLITICAL_EVENTS:
        # Check for overlap
        if (start_date <= geo_end) and (end_date >= geo_start):
            is_geopolitical = True
            break
    
    if is_geopolitical:
        label = 'Geopolitical Crash' if not geo_label_added else None
        ax1.axvspan(start_date, end_date, color="#6C0909", alpha=1, label=label) # Dark Red
        geo_label_added = True
    else:
        label = 'Other Crash' if not crash_label_added else None
        ax1.axvspan(start_date, end_date, color='green', alpha=0.5, label=label) # Lighter Red
        crash_label_added = True

ax1.legend(loc="upper left")
ax1.grid(True, which="both", ls="--", linewidth=0.5)

# --- PANEL 2: The "Fuel" (Market Fragility) ---
# Plot the SVM probability and the static threshold for fragility
FRAGILITY_THRESHOLD = config.FRAGILITY_THRESHOLD

ax2.plot(df_results.index, df_results['y_pred_fuel'], color='royalblue', label='Market Fragility (SVM)', linewidth=1.5)
ax2.axhline(y=FRAGILITY_THRESHOLD, color='red', linestyle='--', linewidth=1.5, label=f'Fragility Threshold ({FRAGILITY_THRESHOLD})')

# Fill area where market is fragile
ax2.fill_between(df_results.index, df_results['y_pred_fuel'], FRAGILITY_THRESHOLD, 
                 where=(df_results['y_pred_fuel'] >= FRAGILITY_THRESHOLD), 
                 color='red', alpha=0.1, interpolate=True)

ax2.set_ylabel('Fragility Score')
ax2.set_title('The "Fuel": Market Fragility (SVM Model)')
ax2.set_ylim(0, 1)
ax2.legend(loc="upper left")
ax2.grid(True, ls="--", linewidth=0.5)

# --- PANEL 3: The "Spark" (GPR) & Dynamic Gating ---
# Plot the LSTM probability and the DYNAMIC threshold
ax3.plot(df_results.index, df_results['y_pred_proba'], color='#FF8C00', label='Geopolitical Risk (LSTM)', linewidth=1.5) # Dark Orange
ax3.plot(df_results.index, df_results['dynamic_threshold'], color='black', linestyle='-', linewidth=1.5, label='Dynamic Gating Threshold')

# Fill area where GPR > Dynamic Threshold (The actual crash signal)
ax3.fill_between(df_results.index, df_results['y_pred_proba'], df_results['dynamic_threshold'], 
                 where=(df_results['y_pred_proba'] >= df_results['dynamic_threshold']), 
                 color='red', alpha=0.3, label='Crash Signal (Gated)', interpolate=True)

ax3.set_ylabel('Risk Score')
ax3.set_title('The "Spark": Geopolitical Risk (LSTM) vs. Dynamic Gating')
ax3.set_ylim(0, 1)
ax3.legend(loc="upper left")
ax3.grid(True, ls="--", linewidth=0.5)

plt.tight_layout()
plt.savefig("backtest_visualization.png", dpi=300)
print("Saved plot to 'backtest_visualization.png'")
