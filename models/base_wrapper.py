from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModelWrapper(ABC):
    """Unified model wrapper interface for ST/TSR/VisRef runners."""

    @abstractmethod
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
