from __future__ import annotations

import time
from typing import Any

from engine.stopping import predictive_entropy, should_stop


def run_st(sample: dict[str, Any], model_wrapper,
           cfg: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    prompt_cfg = cfg["prompt"]
    vis_cfg = cfg["visref"]
    choices = sample.get("choices")

    state = model_wrapper.start_reasoning(
        sample["question"],
        sample["image"],
        choices,
        prompt_cfg,
    )
    state["generation_cfg"] = dict(cfg.get("generation", {}))
    entropy_trace: list[float] = []

    for step in range(1, vis_cfg["max_steps"] + 1):
        _, state = model_wrapper.generate_reasoning_step(state)
        probs = model_wrapper.estimate_answer_distribution(state, choices)
        entropy = predictive_entropy(probs)
        entropy_trace.append(entropy)
        if should_stop(entropy, vis_cfg["entropy_threshold"], step,
                       vis_cfg["max_steps"]):
            break

    pred = model_wrapper.generate_final_answer(state, choices)
    latency = time.perf_counter() - start

    return {
        "sample_id": sample["id"],
        "mode": "st",
        "final_answer": pred,
        "raw_final_answer": state.get("raw_final_answer", pred),
        "gold_answer": sample["answer"],
        "steps_used": len(entropy_trace),
        "entropy_trace": entropy_trace,
        "selected_token_counts": [],
        "reasoning_steps": list(state.get("reasoning_steps", [])),
        "latency_sec": latency,
    }
