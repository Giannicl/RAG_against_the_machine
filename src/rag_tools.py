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
    corpus: List[List[Document]], build_path: str, show_progress: bool = True
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

    dir_path = Path(build_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    retriever.index(corpus_tokens)

    sources = [
        {
            "file_path": doc.metadata["file_path"],
            "first_character_index": doc.metadata["start_index"],
            "last_character_index": doc.metadata["start_index"] + len(doc.page_content),
        }
        for doc in corpus
    ]

    retriever.save(build_path, corpus=sources)
    print(f"Ingestion complete! Indexed {len(text)} chunks under {build_path}")


def searcher(
    query: str,
    build_path: str,
    k: int,
) -> List[MinimalSource]:

    retriever = bm25s.BM25.load(build_path, load_corpus=True)
    query_tokens = bm25s.tokenize(query, stopwords="en")
    results, score = retriever.retrieve(query_tokens, k=k)

    return [
        MinimalSource(**results[0, i])
        for i in range(results.shape[1])
        if score[0, i] > 0
    ]
