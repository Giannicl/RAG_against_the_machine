"""Ingestion: persist chunked documents into a BM25 index on disk."""

import re
from itertools import chain
from pathlib import Path
from typing import Iterable, List

import bm25s
import Stemmer
from bm25s.stopwords import STOPWORDS_EN
from langchain_core.documents import Document
from tqdm import tqdm

from .errors import InputError

_BM25_K1 = 1.5
_BM25_B = 0.75
_MIN_TOKEN_LENGTH = 2
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9_]+")
_SUBWORD_PATTERN = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|[0-9]+")
_QUESTION_STOPWORDS = (
    "what", "how", "which", "where", "when", "why", "who",
    "do", "does", "did", "can", "could", "would", "should",
    "you", "your",
)
_STOPWORDS = frozenset(STOPWORDS_EN) | frozenset(_QUESTION_STOPWORDS)
_STEMMER = Stemmer.Stemmer("english")


def _code_tokenizer(text: str) -> List[str]:
    """Tokenize text into stemmed identifiers and identifier subwords.

    Compound identifiers (snake_case, CamelCase, digit-suffixed) emit
    both the whole identifier and every subword, so a query naming
    either form matches. Must be used identically at index and query
    time.

    Args:
        text: Raw chunk or query text.

    Returns:
        Lowercased, stopword-filtered, stemmed tokens.
    """
    tokens: List[str] = []
    for identifier in _IDENTIFIER_PATTERN.findall(text):
        subwords = _SUBWORD_PATTERN.findall(identifier)
        candidates = (
            [identifier] + subwords if len(subwords) > 1 else [identifier]
        )
        for candidate in candidates:
            lowered = candidate.lower()
            if (
                len(lowered) >= _MIN_TOKEN_LENGTH
                and lowered not in _STOPWORDS
            ):
                tokens.append(lowered)
    stemmed: List[str] = _STEMMER.stemWords(tokens)
    return stemmed


def _chunk_text_composer(document: Document) -> str:
    """Prefix a chunk's text with its file path for indexing.

    The path names the file's topic (e.g. docs/cli/serve.md), which
    queries often mention while a short chunk's body may not. Only the
    indexed text changes; the stored source spans are untouched.

    Args:
        document: Chunk carrying file_path metadata.

    Returns:
        The chunk text with its file path prepended.
    """
    return f"{document.metadata['file_path']}\n{document.page_content}"


def _flatten(corpus: List[List[Document]]) -> List[Document]:
    """Merge per-file chunk lists into one flat chunk list.

    Args:
        corpus: One chunk list per ingested file.

    Returns:
        Every chunk, in corpus order.
    """
    return list(chain.from_iterable(corpus))


def indexer(
    corpus: List[List[Document]], build_path: str, show_progress: bool = True
) -> None:
    """Build a BM25 index over chunked documents and persist it.

    Args:
        corpus: One chunk list per ingested file.
        build_path: Directory to write the index to; created when missing.
        show_progress: Whether to display tqdm progress bars.

    Raises:
        InputError: The corpus holds no chunks, or `build_path` cannot
            be written.
    """
    documents = _flatten(corpus)
    if not documents:
        raise InputError(
            "nothing to index: no indexable chunks in the corpus"
        )
    text = [
        _chunk_text_composer(chunk)
        for chunk in tqdm(
            documents,
            desc="Composing",
            bar_format="{l_bar}{bar}{r_bar}",
            disable=not show_progress,
        )
    ]

    tracked_text: Iterable[str]
    if show_progress:
        tracked_text = tqdm(
            text, desc="Tokenizing", bar_format="{l_bar}{bar}{r_bar}"
        )
    else:
        tracked_text = text

    corpus_tokens = [
        _code_tokenizer(chunk_text) for chunk_text in tracked_text
    ]
    retriever = bm25s.BM25(k1=_BM25_K1, b=_BM25_B)
    retriever.index(corpus_tokens)

    sources = [
        {
            "file_path": document.metadata["file_path"],
            "first_character_index": document.metadata["start_index"],
            "last_character_index": (
                document.metadata["start_index"]
                + len(document.page_content)
            ),
        }
        for document in documents
    ]

    try:
        Path(build_path).mkdir(parents=True, exist_ok=True)
        retriever.save(build_path, corpus=sources)
    except OSError as error:
        raise InputError(f"cannot write {build_path}: {error}") from error
    print(
        f"Ingestion complete! "
        f"Indexed {len(text)} chunks under {build_path}"
    )
