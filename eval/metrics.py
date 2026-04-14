from __future__ import annotations

from statistics import mean


def normalize_answer(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def compute_accuracy(records: list[dict]) -> float:
    if not records:
        return 0.0
    hits = [
        1.0 if exact_match(r["final_answer"], r["gold_answer"]) else 0.0
        for r in records
    ]
    return float(mean(hits))


def compute_latency_stats(records: list[dict]) -> dict[str, float]:
    if not records:
        return {"avg_latency_sec": 0.0}
    latencies = [float(r["latency_sec"]) for r in records]
    return {"avg_latency_sec": float(mean(latencies))}


def compute_compute_stats(records: list[dict]) -> dict[str, float]:
    if not records:
        return {"avg_steps": 0.0}
    steps = [float(r["steps_used"]) for r in records]
    return {"avg_steps": float(mean(steps))}
