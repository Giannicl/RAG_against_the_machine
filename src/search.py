from .models import MinimalSource, MinimalSearchResults
from typing import List, Tuple, Dict
import bm25s


def _extract_questions_and_ids(queries: Dict[str, str]) -> Tuple[str]:
    return (
        [query["question"] for query in queries],
        [query["question_id"] for query in queries],
    )


def _minimal_search_results_builder(
    questions: List[str],
    question_ids: List[str],
    retrieved_sources: List[MinimalSource],
) -> List[MinimalSearchResults]:
    return [
        MinimalSearchResults(
            question_id=question_id,
            question=question,
            retrieved_sources=retrieved_sources,
        )
        for question_id, question, retrieved_sources in zip(
            question_ids, questions, retrieved_sources
        )
    ]


def searcher(
    queries: List[str],
    build_path: str,
    k: int,
) -> List[List[MinimalSource]]:
    retriever = bm25s.BM25.load(build_path, load_corpus=True)
    query_tokens = bm25s.tokenize(queries, stopwords="en")
    results, scores = retriever.retrieve(query_tokens, k=k)

    return [
        [
            MinimalSource(**results[i, j])
            for j in range(results.shape[1])
            if scores[i, j] > 0
        ]
        for i in range(results.shape[0])
    ]
