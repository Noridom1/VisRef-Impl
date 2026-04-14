from __future__ import annotations

from pathlib import Path
from typing import Any

from .base_dataset import BaseVQADataset


class MathVisionDataset(BaseVQADataset):
    """MathVision loader expecting split.{json|jsonl} under dataset root."""

    def _load_split(self) -> list[dict[str, Any]]:
        json_path = Path(self.root) / f"{self.split}.json"
        jsonl_path = Path(self.root) / f"{self.split}.jsonl"
        if json_path.exists():
            return self._read_json_or_jsonl(json_path)
        if jsonl_path.exists():
            return self._read_json_or_jsonl(jsonl_path)
        raise FileNotFoundError(
            f"Could not find {json_path.name} or {jsonl_path.name} in {self.root}"
        )
