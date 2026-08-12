import bm25s
from itertools import chain
from typing import List
from langchain_core.documents import Document
from .models import MinimalSource


def _flatten(corpus: List[List[Document]]) -> List[Document]:
    return list(chain.from_iterable(corpus))


def indexer(corpus: List[List[Document]], save_directory: str) -> None:
    corpus = _flatten(corpus)
    text = [chunks.page_content for chunks in corpus]
    corpus_tokens = bm25s.tokenize(text, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save(save_directory)
