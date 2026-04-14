from __future__ import annotations

import io
import os
from typing import Any
import numpy as np
from PIL import Image

from .base_wrapper import BaseModelWrapper
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


class Qwen(BaseModelWrapper):
    """Starter Qwen wrapper.
    Replace placeholder logic with actual Qwen-VL model loading and generation APIs.

    """

    def __init__(self, model_cfg: dict[str, Any]) -> None:

        self.model_cfg = model_cfg
        self.device = model_cfg.get("device", "cpu")

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Thinking",
            dtype=self.model_cfg.get("dtype", "auto"),
            device_map=self.model_cfg.get("device_map", "auto"),
            attn_implementation=self.model_cfg.get("attn_implementation",
                                                   "eager"),
        )

        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-8B-Thinking")

    def prepare_inputs(self, question: str, image: Any,
                       choices: list[str] | None) -> dict[str, Any]:
        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        elif isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                pass
            elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
                if arr.shape[-1] == 1:
                    arr = arr.squeeze(-1)
            else:
                raise ValueError(
                    "NumPy image must have shape [H, W] or [H, W, C] with C in {1, 3, 4}."
                )

            if np.issubdtype(arr.dtype, np.floating):
                if arr.size and float(arr.max()) <= 1.0:
                    arr = arr * 255.0
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            elif arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

            pil_image = Image.fromarray(arr).convert("RGB")
        elif isinstance(image, (str, os.PathLike)):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, (bytes, bytearray)):
            pil_image = Image.open(io.BytesIO(image)).convert("RGB")
        else:
            raise TypeError(
                "Unsupported image type. Expected PIL.Image.Image, numpy.ndarray, path-like, or bytes."
            )

        if choices is not None:
            question += " Choices: " + ", ".join(choices)

        messages = [{
            "role":
            "user",
            "content": [
                {
                    "type": "image",
                    "image": pil_image
                },
                {
                    "type": "text",
                    "text": question
                },
            ],
        }]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs.to(self.model.device)

    def encode_image(self, image: Any) -> np.ndarray:

        # Placeholder visual tokens with shape [N, d].

        rng = np.random.default_rng(0)

        return rng.standard_normal((64, 128)).astype(np.float32)

    def start_reasoning(self, question: str, image: Any,
                        prompt_cfg: dict[str, Any]) -> dict[str, Any]:

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

    def get_reasoning_text_embeddings(self, state: dict[str,
                                                        Any]) -> np.ndarray:

        # Placeholder z_k embeddings [T_k, d].

        rng = np.random.default_rng(len(state["history"]))

        return rng.standard_normal((32, 128)).astype(np.float32)

    def get_answer_distribution(self, state: dict[str, Any]) -> np.ndarray:

        # Placeholder 4-way answer distribution.

        probs = np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float32)

        return probs / probs.sum()

    def generate_final_answer(self, state: dict[str, Any]) -> str:

        return "placeholder_answer"

    def generate_full_answer(
        self,
        question: str,
        image: Any,
        choices: list[str] | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_k: int | None = None,
    ) -> str:

        inputs = self.prepare_inputs(question, image, choices)

        # Inference: Generation of the output
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]
