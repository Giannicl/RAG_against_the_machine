import fire
import sys


INGEST_PATH = "data/raw/"
BUILD_PATH = "data/processed/"
DATASET_PATH = "data/datasets/UnansweredQuestions/dataset_docs_public.json"
SAVE_DIRECTORY = "data/output/search_results/UnansweredQuestions/"
STUDENT_SEARCH_RESULTS_PATH = (
    "data/output/search_results/UnansweredQuestions/dataset_docs_public.json"
)
MAX_CHUNK_SIZE = 2000
K = 10


class Parser:

    def index(self, max_chunk_size: int = MAX_CHUNK_SIZE) -> None:
        print(f"{max_chunk_size}")

    def search(self, query: str, k: int = K) -> None:
        print(f"{query}, {k}")

    def search_dataset(
        self,
        dataset_path: str = DATASET_PATH,
        k: int = K,
        save_directory: str = SAVE_DIRECTORY,
    ) -> None:
        print(f"{dataset_path}, {k}, {save_directory}")

    def answer(self, query: str, k=K) -> None:
        print(f"{query}, {k}")

    def evaluate(
        self,
        student_search_results_path: str = STUDENT_SEARCH_RESULTS_PATH,
        dataset_path: str = DATASET_PATH,
    ) -> None:
        print(f"{student_search_results_path}, {dataset_path}")

    def answer_dataset(
        self,
        student_search_results_path: str = STUDENT_SEARCH_RESULTS_PATH,
        save_directory: str = SAVE_DIRECTORY,
    ) -> None:
        print(f"{student_search_results_path}, {save_directory}")


def main() -> int:
    fire.Fire(Parser)

    return 0


if __name__ == "__main__":
    sys.exit(main())
