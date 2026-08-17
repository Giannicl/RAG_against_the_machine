from pathlib import Path
from typing import List, Dict
import json


def _read_document(file_path: str) -> str:
    with open(file_path, encoding="utf8", errors="ignore") as file:
        document = file.read()
    return document


def _file_path_extractor(dir_path: str) -> List[Path]:
    return [file_path for file_path in Path(dir_path).rglob("*") if file_path.is_file()]


def _JSON_parser(document: str) -> Dict:
    with open(document, "r") as file:
        data = json.load(file)
    return data
