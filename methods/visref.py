from __future__ import annotations

import time
from typing import Any

from engine.stopping import predictive_entropy, should_stop
from methods.dpp_selector import build_Mk, build_kernel, greedy_logdet_select


def run_visref(sample: dict[str, Any], model_wrapper,
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
    visual_tokens = state["visual_tokens"]
    visual_features = state["visual_features"]

    entropy_trace: list[float] = []
    selected_counts: list[int] = []

    extra_visual_tokens = None
    for step in range(1, vis_cfg["max_steps"] + 1):
        _, state = model_wrapper.generate_reasoning_step(
            state, extra_visual_tokens=extra_visual_tokens)

        z_k = model_wrapper.extract_reasoning_embeddings(state)
        M_k = build_Mk(z_k)
        L_k = build_kernel(visual_tokens, M_k)

        m = int(max(1, vis_cfg["token_budget_ratio"] * visual_tokens.shape[0]))
        selected_idx = greedy_logdet_select(L_k, m)
        extra_visual_tokens = visual_features[selected_idx]
        selected_counts.append(len(selected_idx))

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
        "mode": "visref",
        "final_answer": pred,
        "raw_final_answer": state.get("raw_final_answer", pred),
        "gold_answer": sample["answer"],
        "steps_used": len(entropy_trace),
        "entropy_trace": entropy_trace,
        "selected_token_counts": selected_counts,
        "reasoning_steps": list(state.get("reasoning_steps", [])),
        "latency_sec": latency,
    }
