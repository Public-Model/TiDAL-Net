.PHONY: install test lint smoke clean
install:
	pip install -e .[dev]
test:
	pytest -q
lint:
	ruff check .
smoke:
	bash scripts/reproduce_smoke.sh
clean:
	rm -rf outputs/* data/processed/* data/graphs/* .pytest_cache .ruff_cache
	touch outputs/.gitkeep data/processed/.gitkeep data/graphs/.gitkeep
