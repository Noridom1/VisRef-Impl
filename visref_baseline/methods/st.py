from __future__ import annotations

import time
from typing import Any

from visref_baseline.engine.stopping import predictive_entropy, should_stop


def run_st(sample: dict[str, Any], adapter, cfg: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    prompt_cfg = cfg["prompt"]
    vis_cfg = cfg["visref"]

    state = adapter.start_reasoning(sample["question"], sample["image"], prompt_cfg)
    entropy_trace: list[float] = []

    for step in range(1, vis_cfg["max_steps"] + 1):
        _, state = adapter.generate_reasoning_step(state)
        probs = adapter.get_answer_distribution(state)
        entropy = predictive_entropy(probs)
        entropy_trace.append(entropy)
        if should_stop(entropy, vis_cfg["entropy_threshold"], step, vis_cfg["max_steps"]):
            break

    pred = adapter.generate_final_answer(state)
    latency = time.perf_counter() - start

    return {
        "sample_id": sample["id"],
        "mode": "st",
        "final_answer": pred,
        "gold_answer": sample["answer"],
        "steps_used": len(entropy_trace),
        "entropy_trace": entropy_trace,
        "selected_token_counts": [],
        "latency_sec": latency,
    }
