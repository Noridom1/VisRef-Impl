from __future__ import annotations


import io

import os

from typing import Any, Callable

import numpy as np

from PIL import Image

import torch


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
            model_cfg["hf_repo_or_local_path"],
            # dtype=self.model_cfg.get("dtype", "auto"),
            dtype=torch.bfloat16,
            device_map=self.model_cfg.get("device_map", "auto"),
            attn_implementation=self.model_cfg.get("attn_implementation", "eager"),
            # vision_config={"torch_dtype": torch.bfloat16}
        )

        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Thinking")

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

        messages: list[dict[str, Any]] = []

        system_prompt = self.model_cfg.get("system_prompt")
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": str(system_prompt)
                }],
            })

        messages.append({
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
        })

        print("Prepared messages for processor:", messages)

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        return inputs.to(self.model.device)

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
        print("Prepared inputs keys:", list(inputs.keys()))

        # Inference: Generation of the output

        do_sample = temperature > 0.0

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            repetition_penalty=1.1,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0]

    def get_next_token_logits(
        self,
        question: str,
        image: Any,
        choices: list[str] | None = None,
    ) -> torch.Tensor:
        """Run one forward pass and return next-token logits with shape [B, vocab]."""

        inputs = self.prepare_inputs(question, image, choices)

        self.model.eval()

        with torch.inference_mode():

            outputs = self.model(**inputs, use_cache=False)

        return outputs.logits[:, -1, :]

    def generate_per_token(
        self,
        question: str,
        image: Any,
        choices: list[str] | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int | None = None,
        token_hook: Callable[[dict[str, Any]], torch.Tensor | None] | None = None,
    ) -> str:
        """Autoregressive decoding via explicit forward calls at every token step."""

        model_inputs = self.prepare_inputs(question, image, choices)

        tokenizer = self.processor.tokenizer

        eos_from_cfg = getattr(self.model.generation_config, "eos_token_id", None)

        eos_raw = eos_from_cfg if eos_from_cfg is not None else tokenizer.eos_token_id

        if eos_raw is None:

            stop_token_ids: set[int] = set()

        elif isinstance(eos_raw, int):

            stop_token_ids = {int(eos_raw)}

        else:

            stop_token_ids = {int(x) for x in eos_raw}

        def _apply_hook_and_sample(
            logits: torch.Tensor,
            step: int,
            generated_ids: torch.Tensor,
        ) -> torch.Tensor:

            step_logits = logits

            if token_hook is not None:

                maybe_logits = token_hook(
                    {
                        "step": step,
                        "input_ids": generated_ids,
                        "logits": step_logits,
                        "tokenizer": tokenizer,
                    }
                )

                if maybe_logits is not None:

                    if not isinstance(maybe_logits, torch.Tensor):

                        raise TypeError("token_hook must return None or a torch.Tensor")

                    if maybe_logits.shape != step_logits.shape:

                        raise ValueError(
                            f"token_hook returned shape {tuple(maybe_logits.shape)}, expected {tuple(step_logits.shape)}"
                        )

                    step_logits = maybe_logits

            if temperature <= 0.0:

                return torch.argmax(step_logits, dim=-1, keepdim=True)

            step_logits = step_logits / temperature

            if top_k is not None and top_k > 0:

                k = min(top_k, step_logits.shape[-1])

                top_vals, top_idx = torch.topk(step_logits, k=k, dim=-1)

                probs = torch.softmax(top_vals, dim=-1)

                sampled = torch.multinomial(probs, num_samples=1)

                return top_idx.gather(-1, sampled)

            probs = torch.softmax(step_logits, dim=-1)

            return torch.multinomial(probs, num_samples=1)

        def _is_stop(next_token: torch.Tensor) -> bool:

            if not stop_token_ids:

                return False

            next_ids = next_token.squeeze(-1)

            stop_mask = torch.zeros_like(next_ids, dtype=torch.bool)

            for stop_id in stop_token_ids:

                stop_mask |= next_ids == stop_id

            return bool(torch.all(stop_mask).item())

        def _decode_with_cache() -> str:

            generated = model_inputs["input_ids"]

            attention_mask = model_inputs.get("attention_mask")

            mm_token_type_ids = model_inputs.get("mm_token_type_ids")

            pixel_values = model_inputs.get("pixel_values")

            pixel_values_videos = model_inputs.get("pixel_values_videos")

            image_grid_thw = model_inputs.get("image_grid_thw")

            video_grid_thw = model_inputs.get("video_grid_thw")

            generated_new_tokens: list[torch.Tensor] = []

            past_key_values = None

            step_input_ids = generated

            self.model.eval()

            with torch.inference_mode():

                for step in range(max_new_tokens):

                    forward_inputs = self.model.prepare_inputs_for_generation(
                        step_input_ids,
                        past_key_values=past_key_values,
                        attention_mask=attention_mask,
                        use_cache=True,
                        pixel_values=pixel_values,
                        pixel_values_videos=pixel_values_videos,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        is_first_iteration=(past_key_values is None),
                        mm_token_type_ids=mm_token_type_ids,
                    )

                    outputs = self.model(**forward_inputs)

                    logits = outputs.logits[:, -1, :]

                    past_key_values = outputs.past_key_values

                    next_token = _apply_hook_and_sample(logits, step, generated)

                    generated = torch.cat([generated, next_token], dim=-1)

                    generated_new_tokens.append(next_token)

                    step_input_ids = next_token

                    if attention_mask is not None:

                        ones = torch.ones_like(next_token, dtype=attention_mask.dtype)

                        attention_mask = torch.cat([attention_mask, ones], dim=-1)

                    if mm_token_type_ids is not None:

                        text_type = torch.zeros_like(
                            next_token, dtype=mm_token_type_ids.dtype
                        )

                        mm_token_type_ids = torch.cat(
                            [mm_token_type_ids, text_type], dim=-1
                        )

                    if _is_stop(next_token):

                        break

            if not generated_new_tokens:

                return ""

            new_token_ids = torch.cat(generated_new_tokens, dim=-1)

            return tokenizer.decode(
                new_token_ids[0],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

        def _decode_without_cache() -> str:

            generated = model_inputs["input_ids"]

            attention_mask = model_inputs.get("attention_mask")

            mm_token_type_ids = model_inputs.get("mm_token_type_ids")

            generated_new_tokens: list[torch.Tensor] = []

            self.model.eval()

            with torch.inference_mode():

                for step in range(max_new_tokens):

                    forward_inputs: dict[str, Any] = dict(model_inputs)

                    forward_inputs["input_ids"] = generated

                    if attention_mask is not None:

                        forward_inputs["attention_mask"] = attention_mask

                    if mm_token_type_ids is not None:

                        forward_inputs["mm_token_type_ids"] = mm_token_type_ids

                    forward_inputs["use_cache"] = False

                    outputs = self.model(**forward_inputs)

                    logits = outputs.logits[:, -1, :]

                    next_token = _apply_hook_and_sample(logits, step, generated)

                    generated = torch.cat([generated, next_token], dim=-1)

                    generated_new_tokens.append(next_token)

                    if attention_mask is not None:

                        ones = torch.ones_like(next_token, dtype=attention_mask.dtype)

                        attention_mask = torch.cat([attention_mask, ones], dim=-1)

                    if mm_token_type_ids is not None:

                        text_type = torch.zeros_like(
                            next_token, dtype=mm_token_type_ids.dtype
                        )

                        mm_token_type_ids = torch.cat(
                            [mm_token_type_ids, text_type], dim=-1
                        )

                    if _is_stop(next_token):

                        break

            if not generated_new_tokens:

                return ""

            new_token_ids = torch.cat(generated_new_tokens, dim=-1)

            return tokenizer.decode(
                new_token_ids[0],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

        try:
            
            print("[generate_per_token] Attempting to decode with FlashAttention cache...")
            return _decode_with_cache()

        except RuntimeError as e:

            msg = str(e)

            if (
                "v must have shape (batch_size, seqlen_k, num_heads_k, head_size)"
                not in msg
            ):

                raise

            print(
                "[generate_per_token] FlashAttention cache decode failed; retrying with no-cache full-sequence decode.",
                e,
            )

            return _decode_without_cache()
