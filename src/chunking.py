"""Chunking: split raw corpus files into indexable documents."""

import sys
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from .errors import InputError
from .io import _file_path_extractor, _read_document

_CHUNK_OVERLAP = 250
_INDEXABLE_SUFFIXES = (".txt", ".py", ".md")


def _chunk_overlap(max_chunk_size: int) -> int:
    """Scale the overlap so small chunk sizes stay usable.

    The splitter requires an overlap smaller than the chunk size, so a
    fixed overlap would reject the small chunk sizes the subject allows.

    Args:
        max_chunk_size: Maximum characters per chunk.

    Returns:
        The overlap to give the splitter.
    """
    return min(_CHUNK_OVERLAP, max_chunk_size // 4)


def _text_splitter(
    text: str, max_chunk_size: int, file_path: str
) -> List[Document]:
    """Split prose (.md / .txt) text along markdown boundaries.

    Markdown-aware splitting starts chunks at section headers, so
    chunks line up with the sections questions cite.

    Args:
        text: Full text of the document.
        max_chunk_size: Maximum characters per chunk.
        file_path: Ingested path stored in each chunk's metadata.

    Returns:
        The chunks, each carrying file_path and start_index metadata.
    """
    return RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=max_chunk_size,
        chunk_overlap=_chunk_overlap(max_chunk_size),
        add_start_index=True,
        strip_whitespace=False,
    ).create_documents([text], metadatas=[{"file_path": file_path}])


def _python_splitter(
    text: str, max_chunk_size: int, file_path: str
) -> List[Document]:
    """Split Python source into chunks along language boundaries.

    Args:
        text: Full text of the Python file.
        max_chunk_size: Maximum characters per chunk.
        file_path: Ingested path stored in each chunk's metadata.

    Returns:
        The chunks, each carrying file_path and start_index metadata.
    """
    return RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=max_chunk_size,
        chunk_overlap=_chunk_overlap(max_chunk_size),
        add_start_index=True,
        strip_whitespace=False,
    ).create_documents([text], metadatas=[{"file_path": file_path}])


def _drop_blank_chunks(documents: List[Document]) -> List[Document]:
    """Discard chunks whose text is only whitespace.

    Splitting keeps whitespace so the stored offsets stay exact, which
    can leave blank chunks; indexed, they only carry their file path and
    can take a top-k slot away from a real match.

    Args:
        documents: Chunks produced for one file.

    Returns:
        The chunks holding at least one non-whitespace character.
    """
    return [
        document
        for document in documents
        if document.page_content.strip()
    ]


def file_composer(dir_path: str, max_chunk_size: int) -> List[List[Document]]:
    """Chunk every indexable file under a directory.

    Args:
        dir_path: Directory holding the raw corpus.
        max_chunk_size: Maximum characters per chunk.

    Returns:
        One chunk list per indexable file found.

    Raises:
        InputError: `dir_path` is not a readable directory, or
            `max_chunk_size` is not positive.
    """
    if not Path(dir_path).is_dir():
        raise InputError(f"cannot read directory {dir_path}")
    if max_chunk_size <= 0:
        raise InputError(
            f"max_chunk_size must be positive, got {max_chunk_size}"
        )
    corpus = _file_path_extractor(dir_path)
    chunks: List[List[Document]] = []
    for file_path in corpus:
        suffix = file_path.suffix.lower()
        if suffix not in _INDEXABLE_SUFFIXES:
            continue
        try:
            text = _read_document(str(file_path))
        except OSError as error:
            print(f"skipping {file_path}: {error}", file=sys.stderr)
            continue
        if suffix == ".py":
            split = _python_splitter(text, max_chunk_size, str(file_path))
        else:
            split = _text_splitter(text, max_chunk_size, str(file_path))
        chunks.append(_drop_blank_chunks(split))
    return chunks
