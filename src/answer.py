from .io import _read_document
from .models import MinimalAnswer, MinimalSource, MinimalSearchResults
from .search import searcher, _minimal_search_results_builder
from .llm import LLM
from typing import List, Tuple


def _extract_sources(minimal_sources: List[MinimalSource]) -> Tuple[str]:
    sources = _read_documents(minimal_sources.file_path)[
        minimal_sources.first_character_index : minimal_sources.last_character_index
    ]

    return


def _format_sources(
    minimal_search_results: MinimalSearchResults,
) -> Tuple[str, List[str]]:
    sources = []


def _build_prompts(queries: List[str], context: List[List[MinimalSource]]) -> List[str]:

    return prompt


def answer_generator(query: str, build_path: str, k: int) -> List[MinimalAnswer] | None:
    retrieved_sources = searcher([query], build_path, k)
    minimal_search_results = _minimal_search_results_builder(
        ["1"], [query], retrieved_sources
    )
    formated_sources = _format_sources(minimal_search_results)
    prompt = _build_prompts([query], retrieved_sources)

    print(question_answer)


#    prompts = _build_prompts(queries, k)

#    model = LLM()

#   model.generate([query])
