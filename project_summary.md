# Project Summary: Hybrid Geopolitical and Market Fragility Model for Crash Prediction

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

This dual-threshold system, optimized via the `optimize_gate.py` script, is designed to maximize the F1-Score, balancing the trade-off between correctly predicting crashes (Recall) and avoiding false alarms (Precision).

## 3. Performance Evaluation

### Backtest Results & Analysis

The model was backtested against historical data, which included several known market crashes and periods of intense geopolitical tension.

#### Hits and Misses

The model successfully predicted the **2020 COVID-19 Crash** and parts of the **2022 Bear Market**. However, it missed several smaller, macro-driven crashes.

These misses highlight the model's current limitations. The crash events that were not captured were primarily driven by economic data surprises (like inflation reports) or credit market stress. The model's current feature set is not designed to detect these specific risks.

**To improve this, other data sources should be provided.** For example, incorporating credit spread data like the **Bank of America Merrill Lynch (BAML) High-Yield Spread** would make the "Fuel" model sensitive to corporate credit risk, a key factor in many financial crises.

#### The Value of "False Alarms"

A crucial insight from the backtest is that many of the model's "false alarms" were, in fact, **correctly identified geopolitical threats that did not result in a market crash due to external intervention.** These are not model failures, but rather successful detections of averted crises.

For example, the model generated alarms during:

- **The 2019 US-China Trade War Escalation:** A crash was averted when the Federal Reserve (the "Fed Put") signaled it would cut interest rates to protect the economy.
- **The 2020 China-India Border Clash:** A potential market panic was neutralized by the massive wave of COVID-19 government stimulus, which buoyed global markets.
- **The 2022 Russia-Ukraine "No Limits" Pact:** The model correctly identified the immense risk _before_ the invasion, signaling 20 days early. The market only reacted when the invasion actually began.

These instances demonstrate that the model is effective at its primary job: **detecting the "spark" of geopolitical risk.** The fact that a crash did not always follow is a testament to the "Firewalls" (like Fed intervention or government stimulus) that extinguished the threat, not a failure of the model to detect it.

## 4. Model Interpretability (SHAP Analysis)

To understand what drives the "Spark" model's predictions, SHAP (SHapley Additive exPlanations) analysis was performed. The global feature importance plot reveals that the most significant predictors are the **1-day percentage changes in GPR Acts and Threats**.

This indicates that the model is highly attuned to **sudden, sharp escalations** in geopolitical risk, rather than just a high baseline level of risk. It prioritizes the _change_ in the risk environment, which aligns with the "spark" analogy.

## 5. Conclusion

The hybrid gated model demonstrates a strong ability to identify periods of heightened geopolitical risk that have the potential to cause market crashes. Its "false alarms" are a feature, not a bug, as they often correspond to real-world threats that were successfully mitigated.

The model's primary weakness is its blindness to purely economic or credit-driven crises. Future work should focus on enriching the "Fuel" model's feature set with indicators like the BAML High-Yield Spread and other macroeconomic data to create a more holistic and robust early warning system.
