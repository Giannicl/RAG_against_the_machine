import bm25s
from itertools import chain
from typing import List, Tuple
from langchain_core.documents import Document
from .models import MinimalSource
from pathlib import Path
from tqdm import tqdm


def _flatten(corpus: List[List[Document]]) -> List[Document]:
    return list(chain.from_iterable(corpus))


def indexer(
    corpus: List[List[Document]], save_directory: str, show_progress: bool = True
) -> None:
    corpus = _flatten(corpus)
    text = [
        chunks.page_content
        for chunks in tqdm(
            corpus,
            desc="Chunking",
            bar_format="{l_bar}{bar}{r_bar}",
            disable=not show_progress,
        )
    ]

    if show_progress:
        tracked_text = tqdm(text, desc="Tokenizing", bar_format="{l_bar}{bar}{r_bar}")
    else:
        tracked_text = text

    corpus_tokens = bm25s.tokenize(
        tracked_text,
        stopwords="en",
    )
    retriever = bm25s.BM25()

    dir_path = Path(save_directory)
    dir_path.mkdir(parents=True, exist_ok=True)

    retriever.index(corpus_tokens)
    retriever.save(save_directory)
    print(
        "Ingestion complete! Indexed {.....} chunks under {path/of/data}"
    )  # insert chunk amount and data path


def searcher(
    query: str,
    top_k: int,
) -> MinimalSource: ...
