#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/mac_m4pro.yaml}"
OUTPUT="${2:-answer.csv}"

printf '\n=== Avito Services Retrieval ===\n'
printf 'config: %s\n' "$CONFIG"
printf 'output: %s\n\n' "$OUTPUT"

uv sync --dev
uv run avito-retrieval --config "$CONFIG" run --output "$OUTPUT"
