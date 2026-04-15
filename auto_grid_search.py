import os
import re
import subprocess
import sys

# --- CONFIGURATION GRID ---
# Financial Logic:
# 21 Days = ~1 Trading Month
# 30 Days = Baseline (Calendar Month - Keep for comparison)
# 42 Days = ~2 Trading Months
# 63 Days = ~1 Trading Quarter (Earnings Cycle)

GRID = [
    # 1. Short-Term Memory (1 Trading Month)
    {'warning': 10, 'lookback': 21},
    {'warning': 20, 'lookback': 21},
    {'warning': 30, 'lookback': 21},


    {'warning': 10, 'lookback': 30},
    {'warning': 20, 'lookback': 30},
    {'warning': 30, 'lookback': 30},

    # 2. Medium-Term Memory (2 Trading Months)
    # (We fill the gaps for 42, assuming 30/42 is already done but running it ensures consistency)
    {'warning': 10, 'lookback': 42},
    {'warning': 20, 'lookback': 42},
    {'warning': 30, 'lookback': 42},
    
    # 3. Long-Term Memory (1 Trading Quarter)
    {'warning': 10, 'lookback': 63},
    {'warning': 20, 'lookback': 63},
    {'warning': 30, 'lookback': 63},
]

CONFIG_PATH = "config.py"
LOG_FILE = "grid_search_log.txt"

def update_config(warning, lookback):
    """Updates the config.py file with new parameters."""
    with open(CONFIG_PATH, "r") as f:
        content = f.read()
    
    # Regex to replace the specific lines ensuring we match the variable assignment
    content = re.sub(r"(WARNING_WINDOW_DAYS\s*=\s*)\d+", f"\\g<1>{warning}", content)
    content = re.sub(r"(LOOKBACK_DAYS\s*=\s*)\d+", f"\\g<1>{lookback}", content)
    
    with open(CONFIG_PATH, "w") as f:
        f.write(content)
    print(f"Updated config: Warning={warning}, Lookback={lookback}")

def run_make_all():
    """Runs 'make all' and captures output."""
    print("Running make all...")
    try:
        result = subprocess.run(
            ["make", "all"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Error running make all!")
        # Print the error output to help debug if a script fails
        print(e.stderr) 
        return None

def main():
    with open(LOG_FILE, "a") as f:
        f.write("\n" + "="*50 + "\nNEW GRID SEARCH SESSION\n" + "="*50 + "\n")

    for params in GRID:
        w = params['warning']
        l = params['lookback']
        
        header = f"\n{'='*40}\nTESTING: Warning={w} | Lookback={l}\n{'='*40}\n"
        print(header)
        
        # 1. Update Config
        update_config(w, l)
        
        # 2. Run Full Pipeline (Train -> Optimize -> Backtest) via Make
        backtest_out = run_make_all()
        if not backtest_out: continue
        
        # 5. Save relevant output to log
        with open(LOG_FILE, "a") as f:
            f.write(header)
            
            # Extract AUROC
            auroc_match = re.search(r"AUROC:\s*([0-9\.]+)", backtest_out)
            if auroc_match:
                f.write(f"AUROC: {auroc_match.group(1)}\n")

            # Extract the Event-Based Results section
            if "--- Event-Based Backtest Results ---" in backtest_out:
                start = backtest_out.find("--- Event-Based Backtest Results ---")
                end = backtest_out.find("--- Detailed Event Prediction Table ---")
                summary = backtest_out[start:end] if end != -1 else backtest_out[start:]
                f.write(summary + "\n")
                print("Results logged.")
            else:
                f.write("Error: Could not parse backtest output.\n")

    print(f"\nGrid search complete. Check {LOG_FILE} for results.")

if __name__ == "__main__":
    main()