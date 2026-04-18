from __future__ import annotations

from abc import ABC
from typing import Any


class BaseModelWrapper(ABC):
    """Unified model wrapper interface for ST/TSR/VisRef runners."""

    def start_reasoning(
        self,
        question: str,
        image: Any,
        choices: list[str] | None,
        prompt_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    def generate_reasoning_step(
        self,
        state: dict[str, Any],
        extra_visual_tokens: Any | None = None,
        reflection_instruction: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError

    def extract_reasoning_embeddings(
        self,
        state: dict[str, Any],
    ) -> Any:
        raise NotImplementedError

    def estimate_answer_distribution(
        self,
        state: dict[str, Any],
        choices: list[str] | None = None,
    ) -> Any:
        raise NotImplementedError

    def generate_final_answer(
        self,
        state: dict[str, Any],
        choices: list[str] | None = None,
    ) -> str:
        raise NotImplementedError

    def generate_full_answer(
        self,
        question: str,
        image: Any,
        choices: list[str] | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int | None = None,
    ) -> str:
        """Convenience method to run the full reasoning process and get the final answer."""
        raise NotImplementedError
