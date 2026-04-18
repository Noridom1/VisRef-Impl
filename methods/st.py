from __future__ import annotations

import logging
import time
from typing import Any

from engine.stopping import predictive_entropy, should_stop


logger = logging.getLogger(__name__)


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
        latest_step = ""
        reasoning_steps = state.get("reasoning_steps", [])
        if reasoning_steps:
            latest_step = str(reasoning_steps[-1]).strip()
        logger.info(
            "[ST] sample_id=%s step=%d entropy=%.6f reasoning=%s",
            sample.get("id"),
            step,
            entropy,
            latest_step,
        )
        if should_stop(entropy, vis_cfg["entropy_threshold"], step,
                       vis_cfg["max_steps"]):
            break

    pred = model_wrapper.generate_final_answer(state, choices)
    logger.info("[ST] sample_id=%s final_answer=%s gold_answer=%s steps_used=%d latency_sec=%.2f",
                sample.get("id"), pred, sample["answer"], step, time.perf_counter() - start)
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
