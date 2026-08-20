import fire
import sys

from src.io import _read_document, _JSON_parser, _object_to_json
from src.models import StudentSearchResults
from .splitter import file_composer
from .rag_tools import indexer, searcher, _minimal_search_results_builder
from tqdm import tqdm
from pathlib import Path


_INGEST_PATH = "data/raw/"
_BUILD_PATH = "data/processed/"
_UNANSWERED_QUESTIONS_PATH = (
    "data/datasets/UnansweredQuestions/dataset_docs_public.json"
)
_GROUND_TRUTH = "data/datasets/AnsweredQuestions/dataset_docs_public.json"
_SEARCH_RESULTS_DIRECTORY = "data/output/search_results/UnansweredQuestions/"
_ANSWER_RESULTS_DIRECTORY = "data/output/search_results_and_answer/UnansweredQuestions/"
_STUDENT_SEARCH_RESULTS_PATH = (
    "data/output/search_results/UnansweredQuestions/dataset_docs_public.json"
)
_MAX_CHUNK_SIZE = 2000
_K = 10


class Parser:

    def index(
        self,
        max_chunk_size: int = _MAX_CHUNK_SIZE,
        ingest_path: str = _INGEST_PATH,
        build_path: str = _BUILD_PATH,
    ) -> None:
        corpus = file_composer(ingest_path, max_chunk_size)
        indexer(corpus, build_path)

    def search(self, query: str, build_path: str = _BUILD_PATH, k: int = _K) -> None:
        retrieved_sources = searcher([query], build_path, k)[0]
        for minimal_source in retrieved_sources:
            print(
                f"{minimal_source.file_path} [{minimal_source.first_character_index}:{minimal_source.last_character_index}]"
            )

    def search_dataset(
        self,
        dataset_path: str = _UNANSWERED_QUESTIONS_PATH,
        k: int = _K,
        build_path=_BUILD_PATH,
        save_directory: str = _SEARCH_RESULTS_DIRECTORY,
    ) -> None:
        queries = _JSON_parser(dataset_path)["rag_questions"]
        query_questions = [query["question"] for query in queries]
        query_ids = [query["question_id"] for query in queries]
        retrieved_sources = searcher(query_questions, build_path, k)
        minimal_search_results = _minimal_search_results_builder(
            query_ids, query_questions, retrieved_sources
        )
        student_search_result_model = StudentSearchResults(
            search_results=minimal_search_results, k=k
        )
        _object_to_json(
            student_search_result_model, save_directory, Path(dataset_path).name
        )

    def answer(self, query: str, k=_K) -> None:
        retrieved_sources = self.search(query, _BUILD_PATH, k)

        answer = answer_generator(query, retrieved_sources, k)
        print(answer)

    def evaluate(
        self,
        student_search_results_path: str = _STUDENT_SEARCH_RESULTS_PATH,
        dataset_path: str = _GROUND_TRUTH,
    ) -> None:
        print(f"{student_search_results_path}, {dataset_path}")

    def answer_dataset(
        self,
        student_search_results_path: str = _STUDENT_SEARCH_RESULTS_PATH,
        save_directory: str = _ANSWER_RESULTS_DIRECTORY,
    ) -> None:
        print(f"{student_search_results_path}, {save_directory}")


def main() -> int:
    fire.Fire(Parser)

    return 0


if __name__ == "__main__":
    sys.exit(main())
