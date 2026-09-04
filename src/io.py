"""File system and JSON helpers shared by every stage."""

import json
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, ValidationError

from .errors import InputError
from .models import RagDataset, StudentSearchResults


def _read_document(file_path: str) -> str:
    """Read a text document from disk.

    Args:
        file_path: Path to the document.

    Returns:
        The full text of the document.
    """
    with open(file_path, encoding="utf8", errors="ignore") as file:
        document = file.read()
    return document


def _file_path_extractor(dir_path: str) -> List[Path]:
    """List every file under a directory, recursively.

    Args:
        dir_path: Directory to walk.

    Returns:
        One path per regular file found.
    """
    return [
        file_path
        for file_path in Path(dir_path).rglob("*")
        if file_path.is_file()
    ]


def _JSON_parser(document: str) -> Dict[str, Any]:
    """Read a JSON file.

    Args:
        document: Path to the JSON file.

    Returns:
        The decoded JSON object.

    Raises:
        InputError: The file is missing, unreadable or not valid JSON.
    """
    try:
        with open(document, "r", encoding="utf8") as file:
            data: Dict[str, Any] = json.load(file)
    except OSError as error:
        raise InputError(f"cannot read {document}: {error}") from error
    except json.JSONDecodeError as error:
        raise InputError(f"invalid JSON in {document}: {error}") from error
    return data


def _load_dataset(dataset_path: str) -> RagDataset:
    """Read a question dataset and validate it against RagDataset.

    Args:
        dataset_path: Path to a RagDataset JSON file.

    Returns:
        The validated dataset.

    Raises:
        InputError: The file cannot be read, or does not match RagDataset.
    """
    try:
        return RagDataset.model_validate(_JSON_parser(dataset_path))
    except ValidationError as error:
        raise InputError(
            f"invalid dataset {dataset_path}: {error}"
        ) from error


def _load_student_search_results(
    student_search_results_path: str,
) -> StudentSearchResults:
    """Read saved search results and validate as StudentSearchResults.

    Args:
        student_search_results_path: Path to a StudentSearchResults
            JSON file.

    Returns:
        The validated search results.

    Raises:
        InputError: The file cannot be read, or does not match
            StudentSearchResults.
    """
    try:
        return StudentSearchResults.model_validate(
            _JSON_parser(student_search_results_path)
        )
    except ValidationError as error:
        raise InputError(
            f"invalid student search results "
            f"{student_search_results_path}: {error}"
        ) from error


def _save_object(model: BaseModel, dir_path: str, file_name: str) -> str:
    """Write a pydantic model to disk as indented JSON.

    Args:
        model: The model to serialise.
        dir_path: Directory to write into; created when missing.
        file_name: Name of the output file.

    Returns:
        The path the model was written to.

    Raises:
        InputError: The output directory or file cannot be written.
    """
    try:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        output_path = str(path / file_name)
        with open(output_path, "w", encoding="utf8") as file:
            file.write(model.model_dump_json(indent=2))
    except OSError as error:
        raise InputError(f"cannot write {dir_path}: {error}") from error
    return output_path
