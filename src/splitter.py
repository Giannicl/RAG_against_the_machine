from __future__ import annotations
from typing import List
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document

_CHUNK_SIZE = 512
_CHUNK_OVERLAP = 20


def _read_document(file_path: str) -> str:
    with open(file_path) as file:
        document = file.read()
    return document


def _file_path_extractor(file_path: str) -> List[str]:
    return [file._raw_paths for file in Path(file_path).rglob("*") if file.is_file()]


def _text_splitter(text: str) -> List[Document]:
    return RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    ).create_documents([text])


def _python_splitter(text: str) -> List[Document]:
    return RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP
    ).create_documents([text])


def file_composer(path: str) -> List[List[Document]]:
    corpus = _file_path_extractor(path)
    chunks = []
    for file_path in corpus:
        file_path = str(file_path).strip("''[]")
        text = _read_document(file_path)
        file_path = file_path.lower()
        if file_path.endswith(".py"):
            chunks.append(_python_splitter(text))
        elif file_path.endswith(".txt") or file_path.endswith(".md"):
            chunks.append(_text_splitter(text))
        else:
            continue
    return chunks


file_composer("./data/raw/")
