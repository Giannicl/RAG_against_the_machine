import fire
import sys


DEFAULT_INGEST_PATH = "data/raw/"
DEFAULT_BUILD_PATH = "data/processed/"
DEFAULT_SAVE_DIRECTORY = ""
DEFAULT_STUDENT_SEARCH_RESULTS_PATH = ""
K = 10


class Parser:

    def index(
        self, ingest: str = DEFAULT_INGEST_PATH, build: str = DEFAULT_BUILD_PATH
    ) -> None:
        print(f"{ingest} and {build}")

    def search(self, query: str, k: int = K) -> None:
        print(f"{query}, {k}")


def main() -> int:
    fire.Fire(Parser)

    return 0


if __name__ == "__main__":
    sys.exit(main())
