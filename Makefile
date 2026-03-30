PYTHON := python3
PIP    := pip3
SRC    := slicenet
TESTS  := tests
BENCH  := benchmarks

.PHONY: all install demo bench test clean lint

all: install

install:
	$(PIP) install -e ".[dev]"

demo:
	$(PYTHON) scripts/demo.py

bench:
	$(PYTHON) -m $(BENCH).run_benchmarks

test:
	$(PYTHON) -m pytest $(TESTS) -v --tb=short

test-cov:
	$(PYTHON) -m pytest $(TESTS) -v --tb=short \
		--cov=$(SRC) --cov-report=term-missing --cov-report=html

lint:
	$(PYTHON) -m ruff check $(SRC) $(TESTS)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage dist build *.egg-info

help:
	@echo "Targets:"
	@echo "  install    - Install package and dev dependencies"
	@echo "  demo       - Run the demo script"
	@echo "  bench      - Run all benchmarks"
	@echo "  test       - Run pytest suite"
	@echo "  test-cov   - Run pytest with coverage report"
	@echo "  lint       - Run ruff linter"
	@echo "  clean      - Remove build artifacts and caches"
