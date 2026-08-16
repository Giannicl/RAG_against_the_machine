import fire
import sys
from .splitter import file_composer
from .rag_tools import indexer
from tqdm import tqdm
import time


_INGEST_PATH = "data/raw/vllm-0.10.1"


_INGEST_PATH = "data/raw/"
_BUILD_PATH = "data/processed/"
_DATASET_PATH = "data/datasets/UnansweredQuestions/dataset_docs_public.json"
_SAVE_DIRECTORY = "data/output/search_results/UnansweredQuestions/"
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

    def search(self, query: str, k: int = _K) -> None:
        print(f"{query}, {k}")

    def search_dataset(
        self,
        dataset_path: str = _DATASET_PATH,
        k: int = _K,
        save_directory: str = _SAVE_DIRECTORY,
    ) -> None:
        print(f"{dataset_path}, {k}, {save_directory}")

    def answer(self, query: str, k=_K) -> None:
        print(f"{query}, {k}")

    def evaluate(
        self,
        student_search_results_path: str = _STUDENT_SEARCH_RESULTS_PATH,
        dataset_path: str = _DATASET_PATH,
    ) -> None:
        print(f"{student_search_results_path}, {dataset_path}")

    def answer_dataset(
        self,
        student_search_results_path: str = _STUDENT_SEARCH_RESULTS_PATH,
        save_directory: str = _SAVE_DIRECTORY,
    ) -> None:
        print(f"{student_search_results_path}, {save_directory}")


def main() -> int:
    fire.Fire(Parser)

    return 0


if __name__ == "__main__":
    sys.exit(main())
