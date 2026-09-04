"""Command-line entry point for the RAG pipeline."""

import sys
import uuid
from pathlib import Path

import fire

from .answer import answer_generator, answers_generator
from .chunking import file_composer
from .evaluation import evaluator
from .index import indexer
from .io import _load_dataset, _load_student_search_results, _save_object
from .errors import InputError
from .llm import LLM
from .models import StudentSearchResults, StudentSearchResultsAndAnswer
from .search import (
    _extract_questions_and_ids,
    _minimal_search_results_builder,
    searcher,
)


_INGEST_PATH = "data/raw/"
_PROCESSED_DATA = "data/processed/"
_UNANSWERED_QUESTIONS_PATH = (
    "data/datasets/UnansweredQuestions/dataset_docs_public.json"
)
_GROUND_TRUTH = (
    "data/datasets/AnsweredQuestions/dataset_docs_public.json"
)
_SEARCH_RESULTS_DIRECTORY = "data/output/search_results/UnansweredQuestions/"
_ANSWER_RESULTS_DIRECTORY = (
    "data/output/search_results_and_answer/UnansweredQuestions/"
)
_STUDENT_SEARCH_RESULTS_PATH = (
    "data/output/search_results/UnansweredQuestions/dataset_docs_public.json"
)
_MAX_CHUNK_SIZE = 2000
_K = 10


def _positive_integer_validator(value: int, name: str) -> int:
    """Check that a numeric command-line argument is a positive int.

    Fire passes arguments through unconverted, so a non-numeric value
    arrives here as a string and a decimal one as a float.

    Args:
        value: The argument as Fire parsed it.
        name: Argument name, used in the error message.

    Returns:
        The validated value.

    Raises:
        InputError: The value is not an integer, or is not positive.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(f"{name} must be an integer, got {value!r}")
    if value <= 0:
        raise InputError(f"{name} must be positive, got {value}")
    return value


def _text_validator(value: str, name: str) -> str:
    """Check that a textual command-line argument holds real text.

    Fire converts what looks like a literal, so a numeric or bracketed
    argument arrives here as an int, a list or a dict rather than a str.

    Args:
        value: The argument as Fire parsed it.
        name: Argument name, used in the error message.

    Returns:
        The validated value.

    Raises:
        InputError: The value is not a string, or is blank.
    """
    if not isinstance(value, str):
        raise InputError(f"{name} must be text, got {value!r}")
    if not value.strip():
        raise InputError(f"{name} must not be empty")
    return value


class Parser:
    """Fire command surface: one method per pipeline stage."""

    def index(
        self,
        max_chunk_size: int = _MAX_CHUNK_SIZE,
        ingest_path: str = _INGEST_PATH,
        build_path: str = _PROCESSED_DATA,
    ) -> None:
        """Chunk the corpus and persist a BM25 index over it.

        Args:
            max_chunk_size: Maximum characters per chunk.
            ingest_path: Directory holding the raw corpus.
            build_path: Directory to write the index to.

        Raises:
            InputError: The corpus is empty or the index cannot be
                written.
        """
        max_chunk_size = _positive_integer_validator(
            max_chunk_size, "max_chunk_size"
        )
        if max_chunk_size > _MAX_CHUNK_SIZE:
            print(
                f"Warning: max_chunk_size capped to {_MAX_CHUNK_SIZE}; "
                f"longer sources are rejected by the grader",
                file=sys.stderr,
            )
            max_chunk_size = _MAX_CHUNK_SIZE
        corpus = file_composer(ingest_path, max_chunk_size)
        indexer(corpus, build_path)

    def search(
        self, query: str, k: int = _K, build_path: str = _PROCESSED_DATA
    ) -> None:
        """Print the top-k source locations for a single query.

        The result is printed as a StudentSearchResults document, the
        same schema search_dataset writes, so a single query can be fed
        to the same tooling.

        Args:
            query: Question text to search for.
            k: Number of sources to retrieve.
            build_path: Directory holding the persisted index.

        Raises:
            InputError: k is not positive, the query is empty, or the
                index cannot be loaded.
        """
        k = _positive_integer_validator(k, "k")
        query = _text_validator(query, "query")
        retrieved_sources = searcher([query], build_path, k)
        minimal_search_results = _minimal_search_results_builder(
            [query], [str(uuid.uuid4())], retrieved_sources
        )
        print(
            StudentSearchResults(
                search_results=minimal_search_results, k=k
            ).model_dump_json(indent=2)
        )

    def search_dataset(
        self,
        dataset_path: str = _UNANSWERED_QUESTIONS_PATH,
        k: int = _K,
        save_directory: str = _SEARCH_RESULTS_DIRECTORY,
        build_path: str = _PROCESSED_DATA,
    ) -> None:
        """Search a whole dataset and save a StudentSearchResults file.

        Args:
            dataset_path: Path to the question dataset.
            k: Number of sources to retrieve per question.
            save_directory: Directory to write the results into.
            build_path: Directory holding the persisted index.

        Raises:
            InputError: k is not positive, or the dataset is unusable.
        """
        k = _positive_integer_validator(k, "k")
        dataset = _load_dataset(dataset_path)
        if not dataset.rag_questions:
            raise InputError(f"no questions in {dataset_path}")
        questions, question_ids = _extract_questions_and_ids(
            dataset.rag_questions
        )
        retrieved_sources = searcher(questions, build_path, k)
        minimal_search_results = _minimal_search_results_builder(
            questions, question_ids, retrieved_sources
        )
        output_path = _save_object(
            StudentSearchResults(
                search_results=minimal_search_results, k=k
            ),
            save_directory,
            Path(dataset_path).name,
        )
        print(f"Saved student_search_results to {output_path}")

    def answer(
        self, query: str, k: int = _K, build_path: str = _PROCESSED_DATA
    ) -> None:
        """Answer a single question grounded in retrieved sources.

        The answer is printed as prose first, for a reader, then as a
        StudentSearchResultsAndAnswer document, the same schema
        answer_dataset writes.

        Args:
            query: The question to answer.
            k: Number of sources to retrieve for context.
            build_path: Directory holding the persisted index.

        Raises:
            InputError: k is not positive, the query is empty, or the
                index or model cannot be loaded.
        """
        k = _positive_integer_validator(k, "k")
        query = _text_validator(query, "query")
        answer = answer_generator(query, build_path, k)
        print(answer.answer)
        print()
        print(
            StudentSearchResultsAndAnswer(
                search_results=[answer], k=k
            ).model_dump_json(indent=2)
        )

    def evaluate(
        self,
        student_search_results_path: str = _STUDENT_SEARCH_RESULTS_PATH,
        dataset_path: str = _GROUND_TRUTH,
    ) -> None:
        """Report recall of saved search results against ground truth.

        Args:
            student_search_results_path: Path to a StudentSearchResults
                file produced by search_dataset.
            dataset_path: Path to the ground-truth AnsweredQuestions
                dataset.

        Raises:
            InputError: A file cannot be read, or holds no usable
                questions.
        """
        student_search_results = _load_student_search_results(
            student_search_results_path
        )
        if not student_search_results.search_results:
            raise InputError(
                f"no search results in {student_search_results_path}"
            )
        dataset = _load_dataset(dataset_path)
        print(evaluator(student_search_results, dataset))

    def answer_dataset(
        self,
        student_search_results_path: str = _STUDENT_SEARCH_RESULTS_PATH,
        save_directory: str = _ANSWER_RESULTS_DIRECTORY,
    ) -> None:
        """Answer every saved search result and save the answers.

        Args:
            student_search_results_path: Path to a StudentSearchResults
                file produced by search_dataset.
            save_directory: Directory to write the answered results
                into, under the input's base filename.

        Raises:
            InputError: The input cannot be read, holds no search
                results, or the model or output path is unusable.
        """
        student_search_results = _load_student_search_results(
            student_search_results_path
        )
        minimal_search_results = student_search_results.search_results
        if not minimal_search_results:
            raise InputError(
                f"no search results in {student_search_results_path}"
            )
        print(f"Loaded {len(minimal_search_results)} questions")
        model = LLM()
        minimal_answers = answers_generator(minimal_search_results, model)
        output_path = _save_object(
            StudentSearchResultsAndAnswer(
                search_results=minimal_answers,
                k=student_search_results.k,
            ),
            save_directory,
            Path(student_search_results_path).name,
        )
        print(f"Saved student_search_results_and_answer to {output_path}")


def main() -> int:
    """Run the CLI and translate user errors into an exit status.

    Returns:
        0 on success, 1 when an error was reported, 130 when
        interrupted.
    """
    try:
        fire.Fire(Parser)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except InputError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(
            f"Error: {type(error).__name__}: {error}", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
