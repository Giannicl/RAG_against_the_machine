import sys
import argparse
from typing import List, Optional


DEFAULT_DATASET_PATH = "" 
DEFAULT_SAVE_DIRECTORY = ""
DEFAULT_STUDEN_SEARCH_RESULTS_PATH = ""
K = 10


def args_parser(argv: Optional[List] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="This programme is a retrieval pipeline "
        "that augments the knowledge of an LLM",
    )

    parser.add_argument("")


def main(argv: Optional[List] = None) -> int:

    return 0


if "__name__" = __main__:
    sys.exit(main())
