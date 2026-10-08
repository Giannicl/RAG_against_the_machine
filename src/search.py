"""Retrieval: query the persisted BM25 index for source locations."""

from typing import List, Sequence, Tuple

import bm25s

from .errors import InputError
from .index import _code_tokenizer
from .models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    UnansweredQuestion,
)


def _extract_questions_and_ids(
    questions: Sequence[AnsweredQuestion | UnansweredQuestion],
) -> Tuple[List[str], List[str]]:
    """Split dataset questions into parallel text and id lists.

    Args:
        questions: Validated questions read from a RagDataset.

    Returns:
        The question texts and their ids, in matching order.
    """
    return (
        [question.question for question in questions],
        [question.question_id for question in questions],
    )


def _minimal_search_results_builder(
    questions: List[str],
    question_ids: List[str],
    retrieved_sources: Sequence[List[MinimalSource]],
) -> List[MinimalSearchResults]:
    """Pair every question with the sources retrieved for it.

    Args:
        questions: Question texts.
        question_ids: Ids matching `questions` positionally.
        retrieved_sources: Sources retrieved per question.

    Returns:
        One MinimalSearchResults per question, in the same order.
    """
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
    """Retrieve the top-k source locations for every query.

    Args:
        queries: Question texts to search for.
        build_path: Directory holding the persisted index.
        k: Number of sources to retrieve per query.

    Returns:
        One ranked list per query, of at most k sources. Only chunks
        sharing at least one term with the query are returned, so a
        query holding no indexable or known token yields an empty list.

    Raises:
        InputError: The index cannot be loaded from `build_path`.
    """
    try:
        retriever = bm25s.BM25.load(build_path, load_corpus=True)
    except (OSError, ValueError) as error:
        raise InputError(
            f"cannot load index from {build_path}: {error}"
        ) from error
    query_tokens = [_code_tokenizer(query) for query in queries]
    matchable = [
        index for index, tokens in enumerate(query_tokens) if tokens
    ]
    retrieved: List[List[MinimalSource]] = [[] for _ in queries]
    if not matchable:
        return retrieved
    available_k = min(k, len(retriever.corpus))
    results, scores = retriever.retrieve(
        [query_tokens[index] for index in matchable],
        k=available_k,
        show_progress=len(matchable) > 1,
        leave_progress=True,
    )
    # bm25s always fills the top-k; a zero score means no shared term,
    # so those slots would be arbitrary chunks presented as evidence.
    for row, index in enumerate(matchable):
        retrieved[index] = [
            MinimalSource(**results[row, column])
            for column in range(results.shape[1])
            if scores[row, column] > 0
        ]
    return retrieved
