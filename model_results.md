# Model Performance Summary Matrix

| Warning Window | Lookback | Recall (Hit Rate) | False Alarms | AUROC  | Notes                                        |
| :------------- | :------- | :---------------- | :----------- | :----- | :------------------------------------------- |
| **10 Days**    | 30 Days  | 68%               | 28           | 0.8402 | High false alarms, lower recall.             |
| **20 Days**    | 30 Days  | 72%               | 13           | 0.8640 | Balanced.                                    |
| **30 Days**    | 30 Days  | 80%               | 23           | 0.8140 | High recall, but AUROC dropped.              |
| **30 Days**    | 42 Days  | 80%               | 20           | 0.9160 | **Current Best** (High Recall + High AUROC). |
| 10 Days        | 21 Days  | TBD               | TBD          | TBD    | _1 Trading Month_                            |
| 20 Days        | 21 Days  | TBD               | TBD          | TBD    | _1 Trading Month_                            |
| 30 Days        | 21 Days  | TBD               | TBD          | TBD    | _1 Trading Month_                            |
| 10 Days        | 42 Days  | TBD               | TBD          | TBD    | _2 Trading Months_                           |
| 20 Days        | 42 Days  | TBD               | TBD          | TBD    | _2 Trading Months_                           |
| 10 Days        | 63 Days  | TBD               | TBD          | TBD    | _1 Trading Quarter_                          |
| 20 Days        | 63 Days  | TBD               | TBD          | TBD    | _1 Trading Quarter_                          |
| 30 Days        | 63 Days  | TBD               | TBD          | TBD    | _1 Trading Quarter_                          |

---

# Detailed Reports

## warning_window_days=10 && lookback_window=30

a
Found 25 distinct crash events in the test data.

--- Event-Based Backtest Results ---
Warning Window: 10 days |Lookback: 30 days| Dual-Sensor OR Logic

---

Total Actual Crash Events: 25
Events Predicted Early (Hits): 17
Events Missed: 8
Event-Based Recall (Hit Rate): 68.00%
Total False Alarm Events: 28

---

--- Detailed Event Prediction Table ---
Start Date End Date Reason Status
2020-02-24 2020-02-25 COVID-19 Crash Hit (Predicted)
2020-02-27 2020-02-27 COVID-19 Crash Miss (Not Predicted)
2020-03-05 2020-03-05 COVID-19 Crash Hit (Predicted)
2020-03-09 2020-03-09 COVID-19 Crash Hit (Predicted)
2020-03-11 2020-03-12 COVID-19 Crash Hit (Predicted)
2020-03-16 2020-03-16 COVID-19 Crash Hit (Predicted)
2021-11-26 2021-11-26 Omicron Variant Scare Miss (Not Predicted)
2022-02-03 2022-02-03 2022 Bear Market / Russia-Ukraine Hit (Predicted)
2022-03-07 2022-03-07 2022 Bear Market / Russia-Ukraine Hit (Predicted)
2022-04-22 2022-04-22 2022 Bear Market / Russia-Ukraine Hit (Predicted)
2022-04-26 2022-04-26 2022 Bear Market / Russia-Ukraine Hit (Predicted)
2022-04-29 2022-04-29 2022 Bear Market / Russia-Ukraine Hit (Predicted)
2022-05-05 2022-05-05 2022 Bear Market / Russia-Ukraine Hit (Predicted)
2022-05-09 2022-05-09 2022 Bear Market / Russia-Ukraine Hit (Predicted)
2022-05-18 2022-05-18 Inflation/Retail Earnings Crash (Target/Walmart Miss) Hit (Predicted)
2022-06-13 2022-06-13 CPI Inflation Spike (Fed Hike Fears) Miss (Not Predicted)
2022-09-13 2022-09-13 CPI Hot Print Shock (Worst day since 2020) Miss (Not Predicted)
2024-04-30 2024-04-30 Q4 Earnings Miss / Profit Booking Hit (Predicted)
2024-07-24 2024-07-24 Tech Sector Selloff / Budget Capital Gains Tax Fears Miss (Not Predicted)
2024-08-02 2024-08-05 Yen Carry Trade Unwinding (Global Crash) Hit (Predicted)
2024-09-03 2024-09-03 US Recession Fears (Weak Manufacturing Data) Miss (Not Predicted)
2024-12-18 2024-12-18 Fed "Higher for Longer" Hawkish Signal Miss (Not Predicted)
2025-03-10 2025-03-10 China Deflation / US Trade Tariff Uncertainty Hit (Predicted)
2025-04-03 2025-04-04 Trump 25% "Reciprocal Tariff" on Imports Miss (Not Predicted)
2025-04-10 2025-04-10 Trade War Escalation (US-China-India) Hit (Predicted)

---

--- Detailed False Alarm Analysis ---
Date Range (Approx) Event (The "Spark") Why No Crash? (The "Firewall")
2019-04-30 / 2019-05-29 US-China Trade War Escalation The "Fed Put": The Federal Reserve signaled it would cut rates to save the economy, so investors bought stocks despite the bad news.
2020-06-05 / 2020-07-21 China-India Border Clash & HK Security Law Covid Stimulus: The outcome of the clash was local. Global markets were drunk on trillions of dollars of Covid relief money and tech stock rallies.
2021-01-05 / 2021-01-26 US Capitol Riots (Jan 6) Peaceful Transfer: Biden was certified quickly. The market saw it as a "one-off" event rather than a systemic collapse of the US government.
2022-02-04 / 2022-02-04 Russia-Ukraine "No Limits" Pact Pre-Signal: Your model was actually Right. It predicted the crash 20 days early. The market didn't collapse until the invasion actually started on Feb 24.
2023-10-19 / 2023-10-19 Israel-Gaza Hospital Blast Aftermath Flight to Safety: Paradoxically, when wars start, people buy US stocks/bonds as a "safe haven," keeping prices up even if risk is high.

# warning_window_days = 20 && lookback_window=30

--- Daily Performance Evaluation ---

Daily Classification Report:
precision recall f1-score support

    No Crash       0.99      0.93      0.96      1701
       Crash       0.14      0.69      0.24        29

    accuracy                           0.93      1730

macro avg 0.57 0.81 0.60 1730
weighted avg 0.98 0.93 0.95 1730

Daily Area Under the ROC Curve (AUROC):
AUROC: 0.8640
Daily Confusion Matrix:
[[1581 120]

Found 25 distinct crash events in the test data.

--- Event-Based Backtest Results ---
Warning Window: 20 days |Lookback: 30 days| Dual-Sensor OR Logic

---

Total Actual Crash Events: 25
Events Predicted Early (Hits): 18
Events Missed: 7
Event-Based Recall (Hit Rate): 72.00%
Total False Alarm Events: 13

---

--- Detailed Event Prediction Table ---
Start Date End Date Reason Status
2020-02-24 2020-02-25 COVID-19 Crash Miss (Not Predicted)
2020-02-27 2020-02-27 COVID-19 Crash Miss (Not Predicted)
2020-03-05 2020-03-05 COVID-19 Crash Hit (Predicted)
2020-03-09 2020-03-09 COVID-19 Crash Hit (Predicted)
2020-03-11 2020-03-12 COVID-19 Crash Hit (Predicted)
2020-03-16 2020-03-16 COVID-19 Crash Hit (Predicted)
2021-11-26 2021-11-26 Omicron Variant Scare Miss (Not Predicted)
2022-02-03 2022-02-03 2022 Bear Market / Russia-Ukraine Hit (Predicted)
2022-03-07 2022-03-07 2022 Bear Market / Russia-Ukraine Hit (Predicted)
2022-04-22 2022-04-22 2022 Bear Market / Russia-Ukraine Hit (Predicted)
2022-04-26 2022-04-26 2022 Bear Market / Russia-Ukraine Hit (Predicted)
2022-04-29 2022-04-29 2022 Bear Market / Russia-Ukraine Hit (Predicted)
2022-05-05 2022-05-05 2022 Bear Market / Russia-Ukraine Hit (Predicted)
2022-05-09 2022-05-09 2022 Bear Market / Russia-Ukraine Hit (Predicted)
2022-05-18 2022-05-18 Inflation/Retail Earnings Crash (Target/Walmart Miss) Hit (Predicted)
2022-06-13 2022-06-13 CPI Inflation Spike (Fed Hike Fears) Miss (Not Predicted)
2022-09-13 2022-09-13 CPI Hot Print Shock (Worst day since 2020) Miss (Not Predicted)
2024-04-30 2024-04-30 Q4 Earnings Miss / Profit Booking Hit (Predicted)
2024-07-24 2024-07-24 Tech Sector Selloff / Budget Capital Gains Tax Fears Miss (Not Predicted)
2024-08-02 2024-08-05 Yen Carry Trade Unwinding (Global Crash) Hit (Predicted)
2024-09-03 2024-09-03 US Recession Fears (Weak Manufacturing Data) Hit (Predicted)
2024-12-18 2024-12-18 Fed "Higher for Longer" Hawkish Signal Miss (Not Predicted)
2025-03-10 2025-03-10 China Deflation / US Trade Tariff Uncertainty Hit (Predicted)
2025-04-03 2025-04-04 Trump 25% "Reciprocal Tariff" on Imports Hit (Predicted)
2025-04-10 2025-04-10 Trade War Escalation (US-China-India) Hit (Predicted)

---

--- Detailed False Alarm Analysis ---
Empty DataFrame
Columns: []
Index: []

# warning_window_days = 30 && lookback_window=30

--- Daily Performance Evaluation ---

Daily Classification Report:
precision recall f1-score support

    No Crash       0.99      0.88      0.94      1701
       Crash       0.09      0.69      0.16        29

    accuracy                           0.88      1730

macro avg 0.54 0.79 0.55 1730
weighted avg 0.98 0.88 0.92 1730

Daily Area Under the ROC Curve (AUROC):
AUROC: 0.8140

Daily Confusion Matrix:
[[1503  198]
 [   9   20]]

--- Event-Based Backtest Results ---
Warning Window: 30 days |Lookback: 30 days| Dual-Sensor OR Logic

---

Total Actual Crash Events: 25
Events Predicted Early (Hits): 20
Events Missed: 5
Event-Based Recall (Hit Rate): 80.00%
Total False Alarm Events: 23

---

--- Detailed Event Prediction Table ---
Start Date End Date Reason Status
2020-02-24 2020-02-25 COVID-19 Crash Miss (Not Predicted)
2020-02-27 2020-02-27 COVID-19 Crash Miss (Not Predicted)
2020-03-05 2020-03-05 COVID-19 Crash Hit (Predicted)
2020-03-09 2020-03-09 COVID-19 Crash Hit (Predicted)
2020-03-11 2020-03-12 COVID-19 Crash Hit (Predicted)
2020-03-16 2020-03-16 COVID-19 Crash Hit (Predicted)
2021-11-26 2021-11-26 Omicron Variant Scare Miss (Not Predicted)
2022-02-03 2022-02-03 2022 Bear Market / Russia-Ukraine Hit (Predicted)
2022-03-07 2022-03-07 2022 Bear Market / Russia-Ukraine Hit (Predicted)
2022-04-22 2022-04-22 2022 Bear Market / Russia-Ukraine Hit (Predicted)
2022-04-26 2022-04-26 2022 Bear Market / Russia-Ukraine Hit (Predicted)
2022-04-29 2022-04-29 2022 Bear Market / Russia-Ukraine Hit (Predicted)
2022-05-05 2022-05-05 2022 Bear Market / Russia-Ukraine Hit (Predicted)
2022-05-09 2022-05-09 2022 Bear Market / Russia-Ukraine Hit (Predicted)
2022-05-18 2022-05-18 Inflation/Retail Earnings Crash (Target/Walmart Miss) Hit (Predicted)
2022-06-13 2022-06-13 CPI Inflation Spike (Fed Hike Fears) Hit (Predicted)
2022-09-13 2022-09-13 CPI Hot Print Shock (Worst day since 2020) Miss (Not Predicted)
2024-04-30 2024-04-30 Q4 Earnings Miss / Profit Booking Hit (Predicted)
2024-07-24 2024-07-24 Tech Sector Selloff / Budget Capital Gains Tax Fears Hit (Predicted)
2024-08-02 2024-08-05 Yen Carry Trade Unwinding (Global Crash) Hit (Predicted)
2024-09-03 2024-09-03 US Recession Fears (Weak Manufacturing Data) Hit (Predicted)
2024-12-18 2024-12-18 Fed "Higher for Longer" Hawkish Signal Miss (Not Predicted)
2025-03-10 2025-03-10 China Deflation / US Trade Tariff Uncertainty Hit (Predicted)
2025-04-03 2025-04-04 Trump 25% "Reciprocal Tariff" on Imports Hit (Predicted)
2025-04-10 2025-04-10 Trade War Escalation (US-China-India) Hit (Predicted)

---

--- Detailed False Alarm Analysis ---
Date Range (Approx) Event (The "Spark") Why No Crash? (The "Firewall")
2021-01-05 / 2021-01-26 US Capitol Riots (Jan 6) Peaceful Transfer: Biden was certified quickly. The market saw it as a "one-off" event rather than a systemic collapse of the US government.
2023-10-19 / 2023-10-19 Israel-Gaza Hospital Blast Aftermath Flight to Safety: Paradoxically, when wars start, people buy US stocks/bonds as a "safe haven," keeping prices up even if risk is high.

# warning_window_days = 30 && lookback_window=42

--- Daily Performance Evaluation ---

Daily Classification Report:
precision recall f1-score support

    No Crash       0.99      0.91      0.95      1689
       Crash       0.12      0.72      0.20        29

    accuracy                           0.91      1718

macro avg 0.56 0.82 0.58 1718
weighted avg 0.98 0.91 0.94 1718

Daily Area Under the ROC Curve (AUROC):
AUROC: 0.9160

Daily Confusion Matrix:
[[1534  155]
 [   8   21]]

Found 25 distinct crash events in the test data.
--- Event-Based Backtest Results ---
Warning Window: 30 days |Lookback: 42 days| Dual-Sensor OR Logic

---

Total Actual Crash Events: 25
Events Predicted Early (Hits): 20
Events Missed: 5
Event-Based Recall (Hit Rate): 80.00%
Total False Alarm Events: 20

---

--- Detailed Event Prediction Table ---
Start Date End Date Reason Status
2020-02-24 2020-02-25 COVID-19 Crash Miss (Not Predicted)
2020-02-27 2020-02-27 COVID-19 Crash Miss (Not Predicted)
2020-03-05 2020-03-05 COVID-19 Crash Hit (Predicted)
2020-03-09 2020-03-09 COVID-19 Crash Hit (Predicted)
2020-03-11 2020-03-12 COVID-19 Crash Hit (Predicted)
2020-03-16 2020-03-16 COVID-19 Crash Hit (Predicted)
2021-11-26 2021-11-26 Omicron Variant Scare Miss (Not Predicted)
2022-02-03 2022-02-03 2022 Bear Market / Russia-Ukraine Hit (Predicted)
2022-03-07 2022-03-07 2022 Bear Market / Russia-Ukraine Hit (Predicted)
2022-04-22 2022-04-22 2022 Bear Market / Russia-Ukraine Hit (Predicted)
2022-04-26 2022-04-26 2022 Bear Market / Russia-Ukraine Hit (Predicted)
2022-04-29 2022-04-29 2022 Bear Market / Russia-Ukraine Hit (Predicted)
2022-05-05 2022-05-05 2022 Bear Market / Russia-Ukraine Hit (Predicted)
2022-05-09 2022-05-09 2022 Bear Market / Russia-Ukraine Hit (Predicted)
2022-05-18 2022-05-18 Inflation/Retail Earnings Crash (Target/Walmart Miss) Hit (Predicted)
2022-06-13 2022-06-13 CPI Inflation Spike (Fed Hike Fears) Hit (Predicted)
2022-09-13 2022-09-13 CPI Hot Print Shock (Worst day since 2020) Miss (Not Predicted)
2024-04-30 2024-04-30 Q4 Earnings Miss / Profit Booking Hit (Predicted)
2024-07-24 2024-07-24 Tech Sector Selloff / Budget Capital Gains Tax Fears Hit (Predicted)
2024-08-02 2024-08-05 Yen Carry Trade Unwinding (Global Crash) Hit (Predicted)
2024-09-03 2024-09-03 US Recession Fears (Weak Manufacturing Data) Hit (Predicted)
2024-12-18 2024-12-18 Fed "Higher for Longer" Hawkish Signal Miss (Not Predicted)
2025-03-10 2025-03-10 China Deflation / US Trade Tariff Uncertainty Hit (Predicted)
2025-04-03 2025-04-04 Trump 25% "Reciprocal Tariff" on Imports Hit (Predicted)
2025-04-10 2025-04-10 Trade War Escalation (US-China-India) Hit (Predicted)

---

--- Detailed False Alarm Analysis ---
Date Range (Approx) Event (The "Spark") Why No Crash? (The "Firewall")
2023-10-19 / 2023-10-19 Israel-Gaza Hospital Blast Aftermath Flight to Safety: Paradoxically, when wars start, people buy US stocks/bonds as a "safe haven," keeping prices up even if risk is high.
