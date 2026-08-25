"""Answer generation: turn retrieved sources into grounded answers."""

import uuid
from functools import lru_cache
from typing import List

from .io import _read_document
from .llm import LLM
from .models import MinimalAnswer, MinimalSearchResults, MinimalSource
from .search import _minimal_search_results_builder, searcher

_PROMPT_TEMPLATE = """You are answering questions about the vLLM codebase.
Use only the context below. If it does not contain the answer, say so.

Context:
{context}

Question: {question}
Answer:"""

_NO_CONTEXT = "(no context retrieved)"


@lru_cache(maxsize=256)
def _cached_document(file_path: str) -> str:
    """Read a document, memoised so repeated sources hit disk once.

    Args:
        file_path: Path to the document, as stored in the index.

    Returns:
        The full text of the document.
    """
    return _read_document(file_path)


def _extract_sources(minimal_sources: List[MinimalSource]) -> List[str]:
    """Read back the text span each retrieved source points at.

    Args:
        minimal_sources: Sources retrieved for a single question.

    Returns:
        The chunk text of every source that could be read; unreadable
        files are skipped rather than raising.
    """
    texts: List[str] = []
    for minimal_source in minimal_sources:
        try:
            document = _cached_document(minimal_source.file_path)
        except OSError as error:
            print(f"skipping {minimal_source.file_path}: {error}")
            continue
        texts.append(
            document[
                minimal_source.first_character_index:
                minimal_source.last_character_index
            ]
        )
    return texts


def _build_prompts(
    minimal_search_results: List[MinimalSearchResults],
) -> List[str]:
    """Build one grounded prompt per search result.

    Args:
        minimal_search_results: Questions with their retrieved sources.

    Returns:
        One prompt string per search result, in the same order.
    """
    prompts: List[str] = []
    for minimal_search_result in minimal_search_results:
        sources = _extract_sources(minimal_search_result.retrieved_sources)
        context = "\n\n".join(
            f"[{rank}] {text}" for rank, text in enumerate(sources, start=1)
        )
        prompts.append(
            _PROMPT_TEMPLATE.format(
                context=context or _NO_CONTEXT,
                question=minimal_search_result.question,
            )
        )
    return prompts


def answers_generator(
    minimal_search_results: List[MinimalSearchResults],
    model: LLM | None = None,
) -> List[MinimalAnswer]:
    """Generate an answer for every search result, in one batch.

    Args:
        minimal_search_results: Questions with their retrieved sources.
        model: Loaded LLM to reuse; a new one is built when omitted.

    Returns:
        One MinimalAnswer per input search result, same order.
    """
    if not minimal_search_results:
        return []
    prompts = _build_prompts(minimal_search_results)
    model = model if model is not None else LLM()
    answers = model.generate(prompts)
    return [
        MinimalAnswer(
            **minimal_search_result.model_dump(),
            answer=answer,
        )
        for minimal_search_result, answer in zip(
            minimal_search_results, answers
        )
    ]


def answer_generator(
    query: str, build_path: str, k: int
) -> MinimalAnswer | None:
    """Retrieve context for a single query and answer it.

    Args:
        query: The question to answer.
        build_path: Directory holding the persisted index.
        k: Number of sources to retrieve.

    Returns:
        The answered question, or None when the query is empty.
    """
    if not query.strip() or k <= 0:
        return None
    retrieved_sources = searcher([query], build_path, k)
    minimal_search_results = _minimal_search_results_builder(
        [query], [str(uuid.uuid4())], retrieved_sources
    )
    answers = answers_generator(minimal_search_results)
    return answers[0] if answers else None
