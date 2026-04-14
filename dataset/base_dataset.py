from __future__ import annotations

import json

from abc import ABC, abstractmethod

from pathlib import Path

from typing import Any


class BaseVQADataset(ABC):
    """Common interface for VQA-style datasets."""

    def __init__(
        self,
        root: str,
        split: str,
        image_key: str,
        question_key: str,
        answer_key: str,
        choice_key: str | None = None,
    ) -> None:

        self.root = Path(root)

        self.split = split

        self.image_key = image_key

        self.question_key = question_key

        self.answer_key = answer_key

        self.choice_key = choice_key

        self.samples = self._load_split()

    @abstractmethod
    def _load_split(self) -> list[dict[str, Any]]:

        raise NotImplementedError

    def __len__(self) -> int:

        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:

        row = self.samples[idx]

        return {
            "id":
            row.get("id", idx),
            # "image": row[self.image_key],
            "image":
            (str(self.root /
                 row[self.image_key]) if self.image_key in row else None),
            "question":
            row[self.question_key],
            "answer":
            row[self.answer_key],
            "choices":
            row.get(self.choice_key, None),
        }

    @staticmethod
    def normalize_answer(text: str) -> str:

        return " ".join(str(text).strip().lower().split())

    def _read_json_or_jsonl(self, file_path: Path) -> list[dict[str, Any]]:

        if file_path.suffix == ".json":

            return json.loads(file_path.read_text(encoding="utf-8"))

        if file_path.suffix == ".jsonl":

            lines = file_path.read_text(encoding="utf-8").splitlines()

            return [json.loads(line) for line in lines if line.strip()]

        raise ValueError(f"Unsupported file format: {file_path}")
