from __future__ import annotations

from typing import Any

import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModel
import numpy as np

from .base_wrapper import BaseModelWrapper


class InternVL(BaseModelWrapper):
    """Starter InternVL wrapper.

    Replace placeholder logic with actual InternVL model loading and generation APIs.
    """

    def __init__(self, model_cfg: dict[str, Any]) -> None:
        self.model_cfg = model_cfg
        self.device = model_cfg["device"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["hf_repo_or_local_path"], trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            model_cfg["hf_repo_or_local_path"],
            torch_dtype=model_cfg["dtype"],
            trust_remote_code=True,
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            model_cfg["hf_repo_or_local_path"], trust_remote_code=True
        )

    def prepare_inputs(self, text: str, image: Any) -> dict[str, Any]:
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        return inputs

    def encode_image(self, image: Any) -> np.ndarray:
        # Placeholder visual tokens with shape [N, d].
        rng = np.random.default_rng(0)
        return rng.standard_normal((64, 128)).astype(np.float32)

    def start_reasoning(
        self, question: str, image: Any, prompt_cfg: dict[str, Any]
    ) -> dict[str, Any]:
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

    def generate_full_answer(
        self,
        question: str,
        image: Any,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int | None = None,
    ) -> str:
        
        # 1. Prepare inputs
        inputs = self.prepare_inputs(question, image)
        self.model.eval()

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        generated = input_ids

        # 2. Autoregressive generation loop
        for _ in range(max_new_tokens):

            outputs = self.model(
                input_ids=generated, attention_mask=attention_mask, use_cache=True
            )

            logits = outputs.logits[:, -1, :]

            # 3. Sampling strategy
            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)

                if top_k is not None:
                    topk_probs, topk_indices = torch.topk(probs, top_k)
                    probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
                    next_token = topk_indices.gather(
                        -1, torch.multinomial(probs, num_samples=1)
                    )
                else:
                    next_token = torch.multinomial(probs, num_samples=1)

            # 4. Append token
            generated = torch.cat([generated, next_token], dim=-1)

            # 5. Stop if EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        # 6. Decode output
        output_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)

        return output_text
