# Variables
PYTHON = python

# Targets
.PHONY: all train optimize run plot test lint

all: test train optimize run plot

train:
	@echo "--- Training Models ---"
	$(PYTHON) train_lstm_model.py
	$(PYTHON) train_market_svm.py

optimize:
	@echo "--- Optimizing Thresholds ---"
	$(PYTHON) optimize_gate.py

run:
	@echo "--- Running Backtest ---"
	$(PYTHON) run_event_backtest.py

plot:
	@echo "--- Generating Visualization ---"
	$(PYTHON) plot_backtest.py

test:
	@echo "--- Running Unit Tests ---"
	$(PYTHON) -m unittest discover tests

lint:
	@echo "--- Running Linting ---"
	# Stop the build if there are Python syntax errors or undefined names
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	# exit-zero treats all errors as warnings.
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics