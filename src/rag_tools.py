import bm25s
from itertools import chain
from typing import List, Tuple
from langchain_core.documents import Document
from .models import MinimalSearchResults, MinimalSource
from pathlib import Path
from tqdm import tqdm


def _flatten(corpus: List[List[Document]]) -> List[Document]:
    return list(chain.from_iterable(corpus))


def _minimal_search_results_builder(
    question_ids: str, questions: str, retrieved_sources: List[MinimalSource]
) -> List[MinimalSearchResults]:
    metadata = []
    for i in range(len(question_ids)):
        metadata.append(
            {
                "question_id": question_ids[i],
                "question": questions[i],
                "retrieved_sources": retrieved_sources,
            }
        )
        return [MinimalSearchResults(**metadata[i]) for i in range(len(metadata))]


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
    queries: str | List[str],
    build_path: str,
    k: int,
) -> List[MinimalSource]:
    single = isinstance(queries, str)
    if single:
        queries = [queries]
    retriever = bm25s.BM25.load(build_path, load_corpus=True)
    query_tokens = bm25s.tokenize(queries, stopwords="en")
    results, scores = retriever.retrieve(query_tokens, k=k)

    return [
        MinimalSource(**results[i, j])
        for i in range(results.shape[0])
        for j in range(results.shape[1])
        if scores[i, j] > 0
    ]
