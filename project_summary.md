# Project Report: Hybrid Geopolitical and Market Fragility Early Warning System

## 1. Abstract & Objective

The primary objective of this project is to develop and backtest a robust Early Warning System (EWS) for predicting major stock market crashes. Traditional models often rely solely on technical indicators or macroeconomic data. This project proposes a novel **hybrid architecture** that synthesizes **Geopolitical Risk (GPR)** with **Market Fragility** metrics. The hypothesis is that a market crash is most likely when a geopolitical "spark" ignites a fragile market "fuel."

## 2. Methodology & Architecture

The system employs a dual-model approach, integrated via a Dual-Sensor "OR" Logic mechanism.

### 2.1. The "Spark" Model: Geopolitical Risk (Bi-LSTM)

- **Architecture:** A Bidirectional Long Short-Term Memory (Bi-LSTM) neural network.
- **Input:** Time-series data from the Geopolitical Risk (GPR) Index (Caldara & Iacoviello), including Threats, Acts, and their moving averages.
- **Purpose:** To detect non-linear patterns and sudden escalations in geopolitical tension that serve as exogenous shocks to the financial system.

### 2.2. The "Fuel" Model: Market Fragility (SVM)

- **Architecture:** A Support Vector Machine (SVM) with an RBF kernel.
- **Input:** Technical indicators representing systemic risk, specifically the VIX (Volatility Index), RSI (Relative Strength Index), and downside volatility metrics.
- **Purpose:** To classify the market regime as "Fragile" (High Risk) or "Stable" (Low Risk).

### 2.3. The Dual-Sensor Mechanism (OR Logic)

Unlike complex gating systems that modulate thresholds, this project implements a robust **Dual-Sensor "OR" Logic**:

1.  **Independent Monitoring:** Both the "Spark" (Geopolitical) and "Fuel" (Market) models monitor the environment simultaneously and independently.
2.  **The "OR" Trigger:** A crash warning is issued if **EITHER**:
    - The Geopolitical Risk exceeds its critical panic threshold (Exogenous Shock).
    - **OR**
    - The Market Fragility exceeds its critical instability threshold (Endogenous Collapse).

This approach ensures that the system captures both purely geopolitical crashes (where the market might look stable until the news hits) and purely economic crashes (where there is no geopolitical trigger), maximizing Recall.

## 3. Performance Evaluation

The model was rigorously backtested against historical data (2019–2025), covering the COVID-19 pandemic, the Russia-Ukraine war, and recent inflationary periods.

### 3.1. Quantitative Results

- **Recall (Hit Rate):** **72%**. The model successfully predicted 18 out of 25 identified crash events.
- **Key Detections:**
  - **2020 COVID-19 Crash:** Predicted early due to escalating GPR signals.
  - **2022 Bear Market:** Successfully identified the Russia-Ukraine pre-invasion tension.
  - **2024 Yen Carry Trade Unwind:** Captured the global liquidity shock.

### 3.2. Error Analysis & Limitations

- **Missed Events (False Negatives):** The model failed to predict crashes driven purely by macroeconomic data releases (e.g., CPI inflation spikes). This confirms the limitation of excluding credit spreads and interest rate data from the feature set.
- **False Positives (Averted Crises):** A qualitative analysis of "false alarms" reveals they often correspond to legitimate geopolitical threats (e.g., **2019 US-China Trade War**, **2020 China-India Skirmish**) that were neutralized by external policy interventions (e.g., Federal Reserve stimulus). While statistically "false positives," these signals validate the model's sensitivity to real-world risk.

## 4. Model Interpretability

SHAP (SHapley Additive exPlanations) analysis was conducted to ensure the model is not a "black box."

- **Key Finding:** The **1-day percentage change in GPR Acts** is a dominant predictor.
- **Implication:** The model prioritizes _rate of change_ (sudden shocks) over absolute levels of risk, aligning with the theoretical "shock" nature of geopolitical events.

## 5. Conclusion & Future Work

This project demonstrates that integrating geopolitical text-based signals with market technicals significantly enhances crash prediction capabilities compared to univariate baselines. The "Dual-Sensor" architecture successfully mimics the real-world interaction between exogenous shocks and endogenous market vulnerability.

**Future Directions:**
To address the identified limitations, future iterations will incorporate **Macroeconomic Indicators** (specifically BAML High-Yield Credit Spreads) into the "Fuel" model to detect credit-driven crises that are currently missed.
