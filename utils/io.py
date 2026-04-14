from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    content = "\n".join(json.dumps(r) for r in rows)
    Path(path).write_text(content + "\n", encoding="utf-8")


def read_yaml(path: str) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
