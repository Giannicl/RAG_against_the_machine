from __future__ import annotations
from typing import List
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document

<<<<<<< HEAD
=======
_CHUNK_SIZE = 512
>>>>>>> 558e14558e7dba959412ded5150bd48d87abd0f5
_CHUNK_OVERLAP = 20


def _read_document(file_path: str) -> str:
<<<<<<< HEAD
    with open(file_path, encoding="utf8", errors="ignore") as file:
=======
    with open(file_path) as file:
>>>>>>> 558e14558e7dba959412ded5150bd48d87abd0f5
        document = file.read()
    return document


<<<<<<< HEAD
def _file_path_extractor(dir_path: str) -> List[Path]:
    return [file_path for file_path in Path(dir_path).rglob("*") if file_path.is_file()]


def _text_splitter(text: str, max_chunk_size: int, file_path: str) -> List[Document]:
    return RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=_CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
        add_start_index=True,
        strip_whitespace=False,
    ).create_documents([text], metadatas=[{"file_path": file_path}])


def _python_splitter(text: str, max_chunk_size: int, file_path: str) -> List[Document]:
    return RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=max_chunk_size,
        chunk_overlap=_CHUNK_OVERLAP,
        add_start_index=True,
        strip_whitespace=False,
    ).create_documents([text], metadatas=[{"file_path": file_path}])


def file_composer(dir_path: str, max_chunk_size: int) -> List[List[Document]]:
    corpus = _file_path_extractor(dir_path)
    chunks: List[List[Document]] = []
    for file_path in corpus:
        suffix = file_path.suffix.lower()
        if suffix not in (".txt", ".py", ".md"):
            continue
        try:
            text = _read_document(str(file_path))
        except OSError as e:
            print(f"skipping {file_path}: {e}")
            continue
        if suffix == ".py":
            chunks.append(_python_splitter(text, max_chunk_size, str(file_path)))
        else:
            chunks.append(_text_splitter(text, max_chunk_size, str(file_path)))
    return chunks


if __name__ == "__main__":
    print(file_composer("./data/raw/vllm-0.10.1", 512))
=======
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


def file_composer(path: str) -> List[Document]:
    corpus = _file_path_extractor(path)
    for file_path in corpus:
        text = _read_document(file_path)
        file_path = file_path.lower()
        if file_path.endswith(".py"):
            return _python_splitter(text)
        elif file_path.endswith(".txt") or file_path.endswith(".md"):
            return _text_splitter(text)
        else:
            continue
>>>>>>> 558e14558e7dba959412ded5150bd48d87abd0f5
