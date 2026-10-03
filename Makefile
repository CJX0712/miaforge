# MiaForge Makefile
# 作者：晨星

PY ?= python
VENV ?= .venv

.PHONY: venv install lint test demo clean

venv:
	$(PY) -m venv $(VENV)

install:
	$(VENV)/bin/pip install -r requirements.lock.txt
	$(VENV)/bin/pip install pytest ruff

lint:
	$(VENV)/bin/ruff check . || true

test:
	$(VENV)/bin/pytest -q

demo:
	$(PY) cli.py demo --out examples/benchmark.json

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -f examples/benchmark_ci.json examples/benchmark_docker.json
