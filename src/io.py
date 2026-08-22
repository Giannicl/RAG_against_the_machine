from pathlib import Path
from typing import List, Dict
import json
from pydantic import BaseModel


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


def _save_object(model: BaseModel, dir_path: str, file_name: str) -> str:
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    output_path = f"{dir_path}/{file_name}"
    with open(output_path, "w", encoding="utf8") as file:
        file.write(model.model_dump_json(indent=2))
    return output_path
