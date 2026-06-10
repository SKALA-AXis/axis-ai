.PHONY: ci ci-quick format lint typecheck test pre-commit

# CI 와 동일 (format + lint + mypy + pytest). PR push 전에 실행.
ci: ci-quick test

ci-quick:
	uv run ruff format --check src/
	uv run ruff check src/
	uv run mypy src/

format:
	uv run ruff format src/

lint:
	uv run ruff check src/

typecheck:
	uv run mypy src/

test:
	uv run pytest tests/ -v

pre-commit:
	uv run pre-commit install
	uv run pre-commit run --all-files
