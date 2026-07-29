.PHONY: install run debug lint lint-strict test clean

src:= src
linter:= flake8
type_checker:= mypy
tester:= tests

install:
	uv sync

run:
	uv run python -m $(src)

debug:
	uv run python -m pdb -m $(src)

test:
	uv run pytest -q

lint:
	uv run $(linter) $(src) $(tester)
	uv run $(type_checker) $(src)

lint-strict:
	uv run $(linter) $(src) $(tester)
	uv run $(type_checker) --strict $(src)

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
