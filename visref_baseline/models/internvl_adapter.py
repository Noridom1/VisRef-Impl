from __future__ import annotations

from typing import Any

import numpy as np

from .base_adapter import BaseMLRMAdapter


class InternVLAdapter(BaseMLRMAdapter):
    """Starter adapter.

    Replace placeholder logic with actual InternVL model loading and generation APIs.
    """

    def __init__(self, model_cfg: dict[str, Any]) -> None:
        self.model_cfg = model_cfg

    def encode_image(self, image: Any) -> np.ndarray:
        # Placeholder visual tokens with shape [N, d].
        rng = np.random.default_rng(0)
        return rng.standard_normal((64, 128)).astype(np.float32)

    def start_reasoning(self, question: str, image: Any, prompt_cfg: dict[str, Any]) -> dict[str, Any]:
        return {
            "question": question,
            "image": image,
            "prompt_cfg": prompt_cfg,
            "history": [],
        }

    def generate_reasoning_step(
        self,
        state: dict[str, Any],
        extra_visual_tokens: Any | None = None,
        reflection_instruction: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        k = len(state["history"]) + 1
        prefix = "Reflect" if reflection_instruction else "Think"
        extra = " with visual refocus" if extra_visual_tokens is not None else ""
        step_text = f"{prefix} step {k}{extra}."
        state["history"].append(step_text)
        return step_text, state

    def get_reasoning_text_embeddings(self, state: dict[str, Any]) -> np.ndarray:
        # Placeholder z_k embeddings [T_k, d].
        rng = np.random.default_rng(len(state["history"]))
        return rng.standard_normal((32, 128)).astype(np.float32)

    def get_answer_distribution(self, state: dict[str, Any]) -> np.ndarray:
        # Placeholder 4-way answer distribution.
        probs = np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float32)
        return probs / probs.sum()

    def generate_final_answer(self, state: dict[str, Any]) -> str:
        return "placeholder_answer"
