PYTHON ?= python3
.PHONY: test test-comparison reproduce-comparison
test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v
test-comparison:
	PYTHONPATH=src:experiments/comparison $(PYTHON) -m unittest discover -s tests/comparison -v
reproduce-comparison:
	PYTHONPATH=src:experiments/comparison $(PYTHON) experiments/comparison/reproduce.py
