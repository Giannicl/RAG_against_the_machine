from .io import _read_document
from .rag_tools import searcher
from .models import MinimalAnswer, MinimalSource
from .llm import LLM
from typing import List


def _format_sources(retrieved_sources: List[MinimalSource]) -> List[str]:
    return [
        _read_document(minimal_source.file_path)[
            minimal_source.first_character_index : minimal_source.last_character_index
        ]
        for minimal_source in retrieved_sources
    ]


def _build_prompts(queries: List[str], context: List[str], k: int) -> List[str]:
    prompt = []
    context = []

    for query in queries:
        prompt.append(" ".join())

    return prompt


# TODO: query <-> sources binding is fragile here.
# searcher() filters `scores > 0`, so each query's inner list is 0..k long,
# NOT always k. Parallel lists (queries, retrieved_sources, prompts, outputs)
# are held together only by position, and there is no question_id in sight.
# Fix: take List[MinimalSearchResults] instead -- it already carries
# question_id + question + that query's sources as one object -- and return
# MinimalAnswer (a MinimalSearchResults subclass) so the id travels with the
# answer into the output JSON. Only zip left is results <-> LLM outputs.
def answer_generator(
    queries: List[str], retrieved_sources: List[MinimalSource], k: int
) -> List[MinimalAnswer] | None:
    sources = _format_sources(retrieved_sources)
    prompts = _build_prompts(queries, k)

    model = LLM()

    model.generate([query])
