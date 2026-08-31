VENV  := .venv
PY    := $(VENV)/bin/python
PIP   := $(VENV)/bin/pip

STAGE ?= A
N     ?= 8

.PHONY: help setup smoke test snapshot rollout fanout train necessity api sandboxes clean

help: ## show available targets
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  variables:  STAGE=$(STAGE)  N=$(N)"

setup: ## create .venv and install requirements
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "torch is NOT installed - see requirements-train.txt. It is only needed for"
	@echo "the trainer (step A8); everything through A7 runs on numpy alone."

smoke: ## stage 00 gate - 4 parallel Daytona sandboxes
	$(PY) smoke_test.py

test: ## stage A gates - physics + REINFORCE, local, no network
	$(PY) stage_a_selftest.py
	@if ls tests/test_*.py >/dev/null 2>&1; then $(PY) -m pytest tests -q; \
	 else echo "(no pytest files in tests/ yet - stage_a_selftest.py is the gate)"; fi

snapshot: ## bake the gravity-gauntlet-worker snapshot
	$(PY) -m controller.build_snapshot

rollout: ## one real Daytona rollout          (A6 gate)
	$(PY) run_one_sandbox.py

fanout: ## one generation of N sandboxes      (STAGE=A N=8)
	$(PY) -m controller.fanout --stage $(STAGE) -n $(N)

train: ## REINFORCE training loop             (STAGE=A)
	$(PY) -m controller.train --stage $(STAGE)

necessity: ## M1-M4 ablation study -> results  (STAGE=A)
	$(PY) -m controller.necessity --stage $(STAGE) --repeats 3

api: ## live UI + SSE lifecycle events  -> http://127.0.0.1:8000
	@echo ""
	@echo "  Gravity Gauntlet live UI  ->  http://127.0.0.1:8000"
	@echo "  press LAUNCH GENERATION to spend 8 real Daytona sandboxes"
	@echo ""
	$(VENV)/bin/uvicorn controller.api:app --host 127.0.0.1 --port 8000

sandboxes: ## list live Daytona sandboxes (add ARGS=--delete to reap strays)
	$(PY) -m controller.cleanup $(ARGS)

clean: ## remove caches and scratch run output
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache runs/*
