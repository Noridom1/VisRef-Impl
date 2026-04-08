from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseMLRMAdapter(ABC):
    """Unified interface for ST/TSR/VisRef runners."""

    @abstractmethod
    def encode_image(self, image: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def start_reasoning(self, question: str, image: Any, prompt_cfg: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def generate_reasoning_step(
        self,
        state: dict[str, Any],
        extra_visual_tokens: Any | None = None,
        reflection_instruction: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_reasoning_text_embeddings(self, state: dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_answer_distribution(self, state: dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def generate_final_answer(self, state: dict[str, Any]) -> str:
        raise NotImplementedError
