CONFIG ?= configs/mac_m4pro.yaml
OUTPUT ?= answer.csv

.PHONY: lock sync check warmup indexes history train validate report predict answer test run clean-artifacts

lock:
\tuv lock

sync:
\tuv sync --dev

check:
\tuv run avito-retrieval --config $(CONFIG) check

warmup:
\tuv run avito-retrieval --config $(CONFIG) warmup-model

indexes:
\tuv run avito-retrieval --config $(CONFIG) build-indexes

history:
\tuv run avito-retrieval --config $(CONFIG) build-history

train:
\tuv run avito-retrieval --config $(CONFIG) train-ranker

validate:
\tuv run avito-retrieval --config $(CONFIG) validate

report:
\tuv run avito-retrieval --config $(CONFIG) report

predict:
\tuv run avito-retrieval --config $(CONFIG) predict --output $(OUTPUT)

answer: predict
\tuv run avito-retrieval --config $(CONFIG) validate-answer --answer $(OUTPUT)

test:
\tuv run pytest -q

run:
\tuv sync --dev
\tuv run avito-retrieval --config $(CONFIG) run --output $(OUTPUT)

clean-artifacts:
\trm -rf artifacts
