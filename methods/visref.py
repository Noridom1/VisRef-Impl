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

    state = model_wrapper.start_reasoning(sample["question"], sample["image"],
                                          prompt_cfg)
    visual_tokens = model_wrapper.encode_image(sample["image"])

    entropy_trace: list[float] = []
    selected_counts: list[int] = []

    extra_visual_tokens = None
    for step in range(1, vis_cfg["max_steps"] + 1):
        _, state = model_wrapper.generate_reasoning_step(
            state, extra_visual_tokens=extra_visual_tokens)

        z_k = model_wrapper.get_reasoning_text_embeddings(state)
        M_k = build_Mk(z_k)
        L_k = build_kernel(visual_tokens, M_k)

        m = int(max(1, vis_cfg["token_budget_ratio"] * visual_tokens.shape[0]))
        selected_idx = greedy_logdet_select(L_k, m)
        extra_visual_tokens = visual_tokens[selected_idx]
        selected_counts.append(len(selected_idx))

        probs = model_wrapper.get_answer_distribution(state)
        entropy = predictive_entropy(probs)
        entropy_trace.append(entropy)
        if should_stop(entropy, vis_cfg["entropy_threshold"], step,
                       vis_cfg["max_steps"]):
            break

    pred = model_wrapper.generate_final_answer(state)
    latency = time.perf_counter() - start

    return {
        "sample_id": sample["id"],
        "mode": "visref",
        "final_answer": pred,
        "gold_answer": sample["answer"],
        "steps_used": len(entropy_trace),
        "entropy_trace": entropy_trace,
        "selected_token_counts": selected_counts,
        "latency_sec": latency,
    }
