#!/usr/bin/env bash
# Mirror .github/workflows/ci.yml lint-and-test job (push 전에 실행).
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> ruff format --check src/"
uv run ruff format --check src/

echo "==> ruff check src/"
uv run ruff check src/

echo "==> mypy src/"
uv run mypy src/

echo "==> pytest tests/"
uv run pytest tests/ -v

echo "ci-check: OK"
