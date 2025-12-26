# Hybrid Geopolitical and Market Fragility Model for Crash Prediction

## 1. Objective

The primary goal of this project is to develop and backtest a sophisticated early warning system for predicting major stock market crashes. The system uses a hybrid model that combines geopolitical risk indicators with market fragility metrics to create a more nuanced and accurate prediction mechanism than either data source could provide alone.

## 2. Model Architecture

The system is built on a two-part "Spark" and "Fuel" analogy, where a geopolitical event acts as the "spark" and a vulnerable market acts as the "fuel" necessary for a full-blown crash.

### The "Spark" Model: GPR-LSTM

This is a Bidirectional Long Short-Term Memory (LSTM) neural network trained exclusively on Geopolitical Risk (GPR) data. Its function is to analyze time-series patterns in geopolitical news and identify periods of escalating tension that could act as a catalyst for a market sell-off. Key features include:

- GPR Composite, Threats, and Acts indices.
- Moving Averages (MAs) and trend indicators for these indices.
- Volatility and percentage change metrics to capture sudden spikes in risk.

### The "Fuel" Model: Market-SVM

This is a Support Vector Machine (SVM) model trained on technical indicators that measure market fragility. Its purpose is to assess how susceptible the market is to a shock. A fragile, over-leveraged, or complacent market is considered to have more "fuel" for a crash. The features used are:

- **VIX (CBOE Volatility Index):** Measures market fear and expected volatility.
- **RSI (Relative Strength Index):** A momentum oscillator that measures the speed and change of price movements.
- **Rolling 1st Percentile:** A measure of recent downside volatility.

### The Gated Mechanism

The core innovation of this project is the "gated" logic that combines the two models. Instead of using a single, static threshold for a crash prediction, the system dynamically adjusts its sensitivity based on the market's condition.

- The "Fuel" model first determines if the market is in a **High Fragility** or **Low Fragility** state.
- If the market is fragile, a **lower, more sensitive GPR threshold** is used. This means even a minor geopolitical "spark" can trigger a warning.
- If the market is stable, a **higher, less sensitive GPR threshold** is used, requiring a much more significant geopolitical event to trigger an alarm.

This dual-threshold system is designed to maximize the F1-Score, balancing the trade-off between correctly predicting crashes (Recall) and avoiding false alarms (Precision).

## 3. Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Data Setup**: Ensure your training and testing data CSV files are placed in a `data/` directory.

## 4. Usage

The project workflow consists of three main stages: Optimization, Backtesting, and Visualization.

### Step 1: Optimize Thresholds

Run the optimization script to find the best dynamic gating thresholds using your validation set.

```bash
python optimize_gate.py
```

_Output: Prints the optimal Fragility, High-Sensitivity, and Low-Sensitivity thresholds._

### Step 2: Run Backtest

Execute the full event-based backtest on the test dataset. This generates predictions and evaluates performance against known crash events.

```bash
python run_event_backtest.py
```

_Output: Generates `.npy` prediction files and prints a detailed classification report and event hit/miss table._

### Step 3: Visualize Results

Generate the final 3-panel visualization (Price, Fuel, and Spark).

```bash
python plot_backtest.py
```

_Output: Saves `backtest_visualization.png`._

## 5. Performance & Analysis

### Hits and Misses

The model successfully predicted the **2020 COVID-19 Crash** and parts of the **2022 Bear Market**. However, it missed several smaller, macro-driven crashes (e.g., inflation spikes), highlighting the need for future integration of macroeconomic data (like credit spreads).

### The Value of "False Alarms"

Many "false alarms" were correctly identified geopolitical threats (e.g., **2019 Trade War**, **2022 Russia-Ukraine Pre-Invasion**) that did not result in a crash due to external interventions (e.g., Fed stimulus). These demonstrate the model's ability to detect the "spark" even if the "fire" was put out by policymakers.

### Model Interpretability (SHAP)

SHAP analysis reveals that the **1-day percentage changes in GPR Acts and Threats** are the most significant predictors. This confirms the model is highly attuned to sudden, sharp escalations in risk rather than baseline tension levels.

## 6. Future Work

- **Macro Integration**: Incorporate BAML High-Yield Spreads to detect credit-driven crises.
- **Transformer Models**: Experiment with Transformer architectures for better long-term dependency capture in GPR sequences.
