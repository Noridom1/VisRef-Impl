from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from statistics import mean
from typing import Any


ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
ANSWER_PREFIX_RE = re.compile(
    r"(?:final\s+answer|answer)\s*[:\-]\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)
OPTION_RE = re.compile(r"^\(?([a-z])\)?(?:[\.\):\s].*)?$", re.IGNORECASE)
NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(_coerce_text(item) for item in value)
    return str(value)


def extract_answer_text(text: Any) -> str:
    raw = _coerce_text(text).strip()
    if not raw:
        return ""

    tagged_matches = ANSWER_TAG_RE.findall(raw)
    if tagged_matches:
        return tagged_matches[-1].strip()

    lowered = raw.lower()
    if "<answer>" in lowered:
        return raw[lowered.rfind("<answer>") + len("<answer>"):].strip()

    prefix_matches = ANSWER_PREFIX_RE.findall(raw)
    if prefix_matches:
        return prefix_matches[-1].strip()

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    return raw


def _strip_wrapping(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.strip("`")
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'":
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _normalize_numeric(text: str) -> str:
    candidate = text.replace(",", "").strip()
    suffix = ""
    if candidate.endswith("%"):
        suffix = "%"
        candidate = candidate[:-1].strip()
    if not NUMERIC_RE.fullmatch(candidate):
        return text
    try:
        value = Decimal(candidate)
    except InvalidOperation:
        return text

    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized in {"-0", "+0", ""}:
        normalized = "0"
    return normalized + suffix


def normalize_answer(text: Any, dataset_name: str | None = None) -> str:
    raw = _coerce_text(text)
    if dataset_name and dataset_name.lower() == "mathvista":
        raw = extract_answer_text(raw)

    cleaned = _strip_wrapping(raw)
    cleaned = " ".join(cleaned.strip().split())
    cleaned = cleaned.strip(".,;:!?")
    cleaned = cleaned.lower()

    option_match = OPTION_RE.fullmatch(cleaned)
    if option_match:
        return option_match.group(1).lower()

    return _normalize_numeric(cleaned)


def exact_match(pred: Any, gold: Any, dataset_name: str | None = None) -> bool:
    return normalize_answer(pred, dataset_name) == normalize_answer(
        gold, dataset_name)


def compare_answers(pred: Any, gold: Any, dataset_name: str | None = None) -> bool:
    if isinstance(gold, (list, tuple, set)):
        return any(compare_answers(pred, item, dataset_name) for item in gold)
    normalized_pred = normalize_answer(pred, dataset_name)
    normalized_gold = normalize_answer(gold, dataset_name)
    if normalized_pred == normalized_gold:
        return True

    try:
        pred_val = float(normalized_pred.rstrip("%"))
        gold_val = float(normalized_gold.rstrip("%"))
    except ValueError:
        return False

    return math.isclose(pred_val, gold_val, rel_tol=1e-6, abs_tol=1e-6)


def compute_accuracy(records: list[dict]) -> float:
    if not records:
        return 0.0
    hits = [1.0 if bool(r.get("is_correct")) else 0.0 for r in records]
    return float(mean(hits))


def compute_latency_stats(records: list[dict]) -> dict[str, float]:
    if not records:
        return {"avg_latency_sec": 0.0}
    latencies = [float(r["latency_sec"]) for r in records]
    return {"avg_latency_sec": float(mean(latencies))}


def compute_compute_stats(records: list[dict]) -> dict[str, float]:
    if not records:
        return {"avg_steps": 0.0, "avg_selected_tokens": 0.0}
    steps = [float(r["steps_used"]) for r in records]
    selected_counts = []
    for record in records:
        counts = record.get("selected_token_counts") or []
        if counts:
            selected_counts.append(float(mean(counts)))
        else:
            selected_counts.append(0.0)
    return {
        "avg_steps": float(mean(steps)),
        "avg_selected_tokens": float(mean(selected_counts)),
    }
