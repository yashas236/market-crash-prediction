import pandas as pd

# --- Model Parameters ---
LOOKBACK_DAYS = 30
WARNING_WINDOW_DAYS = 10

# --- Feature Definitions ---
GPR_FEATURES = [
    'GPR_Composite', 'GPR_Threats', 'GPR_Acts', 
    'GPR_Threats_MA_63', 'GPR_Threats_MA_126', 'GPR_Threats_Trend', 
    'GPR_Acts_MA_63', 'GPR_Threats_1D_PctChange', 'GPR_Acts_1D_PctChange',
    'GPR_Threats_5D_PctChange', 'GPR_Threats_Vol_21D', 'GPR_Acts_Vol_21D'
]

MARKET_FEATURES = [
    'VIX_Close', 'RSI_14', 'Rolling_1st_Percentile',
]

# --- Optimized Thresholds (Default) ---
# These can be updated after running optimize_gate.py
FRAGILITY_THRESHOLD = 0.04        # Low because SVM probabilities are skewed low
HIGH_SENS_GPR_THRESHOLD = 0.37    # Be sensitive when market is fragile
LOW_SENS_GPR_THRESHOLD = 0.52     # Be tolerant when market is stable

# --- File Paths ---
DATA_TRAIN = "data/train_final.csv"
DATA_TEST = "data/test_final.csv"
DATA_VAL = "data/validation_final.csv"
MODEL_SPARK = "trained_ews_model_gpr.h5"
MODEL_FUEL = "svm_fuel_model.pkl"
SCALER_MARKET = "market_features_scaler.pkl"
SCALER_GPR = "gpr_features_scaler.pkl"

# --- Event Definitions ---

# Used in run_event_backtest.py
KNOWN_CRASHES = [ 
    (pd.Timestamp("2020-02-24"), pd.Timestamp("2020-03-16"), "COVID-19 Crash"),
    (pd.Timestamp("2021-11-26"), pd.Timestamp("2021-11-26"), "Omicron Variant Scare"),
    (pd.Timestamp("2022-05-18"), pd.Timestamp("2022-05-18"), "Inflation/Retail Earnings Crash (Target/Walmart Miss)"),
    (pd.Timestamp("2022-06-13"), pd.Timestamp("2022-06-13"), "CPI Inflation Spike (Fed Hike Fears)"),
    (pd.Timestamp("2022-09-13"), pd.Timestamp("2022-09-13"), "CPI Hot Print Shock (Worst day since 2020)"),
    (pd.Timestamp("2024-04-30"), pd.Timestamp("2024-04-30"), "Q4 Earnings Miss / Profit Booking"),
    (pd.Timestamp("2024-07-24"), pd.Timestamp("2024-07-24"), "Tech Sector Selloff / Budget Capital Gains Tax Fears"),
    (pd.Timestamp("2024-08-02"), pd.Timestamp("2024-08-05"), "Yen Carry Trade Unwinding (Global Crash)"),
    (pd.Timestamp("2024-09-03"), pd.Timestamp("2024-09-03"), "US Recession Fears (Weak Manufacturing Data)"),
    (pd.Timestamp("2024-12-18"), pd.Timestamp("2024-12-18"), 'Fed "Higher for Longer" Hawkish Signal'),
    (pd.Timestamp("2025-03-10"), pd.Timestamp("2025-03-10"), "China Deflation / US Trade Tariff Uncertainty"),
    (pd.Timestamp("2025-04-03"), pd.Timestamp("2025-04-04"), 'Trump 25% "Reciprocal Tariff" on Imports'),
    (pd.Timestamp("2025-04-10"), pd.Timestamp("2025-04-10"), "Trade War Escalation (US-China-India)"),
    (pd.Timestamp("2022-02-03"), pd.Timestamp("2022-06-13"), "2022 Bear Market / Russia-Ukraine"),
]

# Used in run_event_backtest.py
KNOWN_FALSE_ALARMS = [
    (pd.Timestamp("2019-04-30"), pd.Timestamp("2019-05-29"), "US-China Trade War Escalation", 'The "Fed Put": The Federal Reserve signaled it would cut rates to save the economy, so investors bought stocks despite the bad news.'),
    (pd.Timestamp("2019-11-05"), pd.Timestamp("2019-11-05"), "Iran Nuclear Escalation", 'Market Apathy: The market was focused on the upcoming "Phase One" trade deal with China and ignored Middle East tensions.'),
    (pd.Timestamp("2020-01-17"), pd.Timestamp("2020-01-17"), "Soleimani Aftermath / Iran Missiles", "De-escalation: Both Trump and Iran signaled they didn't want full war. The fear remained (Model stayed high), but the immediate threat passed."),
    (pd.Timestamp("2020-06-05"), pd.Timestamp("2020-07-21"), "China-India Border Clash & HK Security Law", "Covid Stimulus: The outcome of the clash was local. Global markets were drunk on trillions of dollars of Covid relief money and tech stock rallies."),
    (pd.Timestamp("2020-12-22"), pd.Timestamp("2020-12-22"), "SolarWinds Hack / Trump Veto Threat", "Resolution: The bill passed anyway. The hack was serious but didn't affect corporate earnings, so Wall Street ignored it."),
    (pd.Timestamp("2021-01-05"), pd.Timestamp("2021-01-26"), "US Capitol Riots (Jan 6)", 'Peaceful Transfer: Biden was certified quickly. The market saw it as a "one-off" event rather than a systemic collapse of the US government.'),
    (pd.Timestamp("2021-02-02"), pd.Timestamp("2021-02-02"), "Myanmar Coup", "Irrelevance: A classic example of GPR vs. Market. A tragedy for human rights, but Myanmar has zero impact on the S&P 500 earnings."),
    (pd.Timestamp("2021-05-25"), pd.Timestamp("2021-05-25"), "Israel-Hamas Conflict (11-Day War)", "Containment: The conflict didn't spread to Oil producing nations (Iran/Saudi), so the global economy was unaffected."),
    (pd.Timestamp("2021-08-13"), pd.Timestamp("2021-08-26"), "Fall of Kabul / Airport Bombing", 'Priced In: The US withdrawal was known. The chaos was ugly, but it marked the end of a war, which markets often interpret as "reducing future spending."'),
    (pd.Timestamp("2022-02-04"), pd.Timestamp("2022-02-04"), 'Russia-Ukraine "No Limits" Pact', "Pre-Signal: Your model was actually Right. It predicted the crash 20 days early. The market didn't collapse until the invasion actually started on Feb 24."),
    (pd.Timestamp("2023-10-19"), pd.Timestamp("2023-10-19"), "Israel-Gaza Hospital Blast Aftermath", 'Flight to Safety: Paradoxically, when wars start, people buy US stocks/bonds as a "safe haven," keeping prices up even if risk is high.'),
]

# Used in plot_backtest.py
GEOPOLITICAL_EVENTS = [
    # --- 1990s: Oil & Emerging Market Crises ---
    (pd.Timestamp("1990-08-02"), pd.Timestamp("1991-02-28")), # Gulf War (Invasion of Kuwait)
    (pd.Timestamp("1997-07-02"), pd.Timestamp("1997-12-31")), # Asian Financial Crisis (Baht Devaluation)
    (pd.Timestamp("1998-08-17"), pd.Timestamp("1998-11-01")), # Russian Financial Crisis (Default)

    # --- 2000s: Terror & The Great Recession ---
    (pd.Timestamp("2001-09-11"), pd.Timestamp("2001-10-31")), # 9/11 Terror Attacks
    (pd.Timestamp("2003-03-19"), pd.Timestamp("2003-05-01")), # US Invasion of Iraq
    (pd.Timestamp("2008-09-15"), pd.Timestamp("2008-11-01")), # Lehman Brothers Collapse (Systemic Shock)

    # --- 2010s: European Instability & Trade Wars ---
    (pd.Timestamp("2011-02-15"), pd.Timestamp("2011-10-20")), # Libyan Civil War / Arab Spring (Oil Shock)
    (pd.Timestamp("2014-02-20"), pd.Timestamp("2014-03-21")), # Crimea Annexation (Russia-Ukraine Phase 1)
    (pd.Timestamp("2016-06-23"), pd.Timestamp("2016-06-30")), # Brexit Referendum Shock
    (pd.Timestamp("2018-03-22"), pd.Timestamp("2018-12-01")), # US-China Trade War Escalation (Section 301 Tariffs)

    # --- 2020s: Pandemic & Modern Conflicts ---
    (pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-10")), # Soleimani Assassination (US-Iran Tension)
    (pd.Timestamp("2020-02-20"), pd.Timestamp("2020-04-07")), # COVID-19 Pandemic Crash
    (pd.Timestamp("2022-02-24"), pd.Timestamp("2022-04-01")), # Russia-Ukraine War Invasion (Phase 2)
    (pd.Timestamp("2023-10-07"), pd.Timestamp("2023-11-15")), # Israel-Hamas War Start

    # --- 2024-2025: The "Missed" Events (Macro-Geopolitics) ---
    (pd.Timestamp("2024-04-13"), pd.Timestamp("2024-04-19")), # Iran Missile Attack on Israel
    (pd.Timestamp("2024-08-02"), pd.Timestamp("2024-08-07")), # Yen Carry Trade Unwind (Global Liquidity Shock)
    (pd.Timestamp("2025-04-03"), pd.Timestamp("2025-04-15")), # Trump Reciprocal Tariffs (Trade War II)
]