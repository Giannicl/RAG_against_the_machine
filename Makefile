.PHONY: install run debug lint lint-strict test clean index search search_dataset answer evaluate answer_dataset

src:= src
linter:= flake8
type_checker:= mypy
tester:= tests
max_chunk_size?=
k?=
dataset_path?=
save_directory?=

install:
	uv sync

run-index:
	uv run python -m $(src) index $(if $(max_chunk_size), --max_chunk_size $(max_chunk_size))

run-search:
	uv run python -m $(src) search --query $(query) $(if $(k), --k $(k))

run-search-dataset:
	uv run python -m $(src) search_dataset --dataset_path $(if $(dataset_path), --dataset_path $(dataset_path)) $(if $(k), --k $(k)) $(if $(save_directory), --k $(save_directory))

run-answer:
	uv run python -m $(src) answer

run-evaluate:
	uv run python -m $(src) evaluate

run-answer-dataset:
	uv run python -m $(src) answer_dataset

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
