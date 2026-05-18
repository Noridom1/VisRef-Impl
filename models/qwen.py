from __future__ import annotations
import io
import os
import re
from typing import Any, Callable

import numpy as np

from PIL import Image

import torch

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


from .base_wrapper import BaseModelWrapper

from eval.metrics import extract_answer_text

REASONING_TAG_RE = re.compile(
    r"<reasoning_step>\s*(.*?)\s*</reasoning_step>", re.IGNORECASE | re.DOTALL
)


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

    def prepare_inputs(
        self,
        question: str,
        image: Any,
        choices: list[str] | None,
        extra_visual_tokens: Any | None = None,
    ) -> dict[str, Any]:
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
        aux_image = self._tokens_to_aux_image(extra_visual_tokens)

        system_prompt = self.model_cfg.get("system_prompt")
        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": str(system_prompt)}],
                }
            )

        messages.append(
            {
                "role": "user",
                "content": [{"type": "image", "image": pil_image}],
            }
        )
        if aux_image is not None:
            messages[-1]["content"].append({"type": "image", "image": aux_image})
        messages[-1]["content"].append({"type": "text", "text": question})

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
        extra_visual_tokens: Any | None = None,
    ) -> str:

        inputs = self.prepare_inputs(question, image, choices, extra_visual_tokens)
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

    # --- Minimal stateful API to match InternVL semantics -----------------
    def _normalize_question(self, question: str) -> str:
        normalized = re.sub(r"<image\d*>", "", str(question))
        normalized = normalized.replace("<image>", "")
        return normalized.strip()

    def _normalize_choices(self, choices: Any) -> list[str] | None:
        if choices is None:
            return None
        if isinstance(choices, dict):
            return [f"{key}. {value}" for key, value in choices.items()]
        if isinstance(choices, (list, tuple)):
            return [str(choice).strip() for choice in choices if str(choice).strip()]
        text = str(choices).strip()
        return [text] if text else None

    def _default_prompt_cfg(self) -> dict[str, Any]:
        return {
            "system": self.model_cfg.get("system_prompt", ""),
            "think_instruction": "Think step by step before giving the final answer.",
            "reflection_instruction": "Wait. Think more carefully using visual evidence.",
        }

    def _build_reasoning_prompt(
        self, state: dict[str, Any], reflection_instruction: str | None = None
    ) -> str:
        prompt_cfg = state.get("prompt_cfg") or {}
        pieces = []
        system_prompt = prompt_cfg.get("system")
        if system_prompt:
            pieces.append(f"System: {system_prompt}")
        pieces.append("User:")
        pieces.append("<image>")
        pieces.append(f"Question: {state['question']}")
        choices_block = ""
        if state.get("choices"):
            choices_block = "Choices:\n" + "\n".join(f"- {c}" for c in state["choices"])
        if choices_block:
            pieces.append(choices_block)
        if state.get("reasoning_steps"):
            pieces.append("Reasoning so far:")
            for idx, step in enumerate(state["reasoning_steps"], start=1):
                pieces.append(f"{idx}. {step}")
        instruction = reflection_instruction or prompt_cfg.get(
            "think_instruction", "Think step by step before giving the final answer."
        )
        pieces.append(f"Instruction: {instruction}")
        pieces.append(
            "Write exactly one new reasoning step and wrap it in <reasoning_step>...</reasoning_step>. Do not give the final answer."
        )
        pieces.append("Assistant:")
        return "\n".join(pieces)

    def _build_answer_prompt(self, state: dict[str, Any]) -> str:
        prompt_cfg = state.get("prompt_cfg") or {}
        pieces = []
        system_prompt = prompt_cfg.get("system")
        if system_prompt:
            pieces.append(f"System: {system_prompt}")
        pieces.append("User:")
        pieces.append("<image>")
        pieces.append(f"Question: {state['question']}")
        if state.get("reasoning_steps"):
            pieces.append("Reasoning:")
            for idx, step in enumerate(state["reasoning_steps"], start=1):
                pieces.append(f"{idx}. {step}")
        if state.get("choices"):
            pieces.append(
                "Select the best option from the provided choices and copy it exactly."
            )
        else:
            pieces.append("Provide the shortest correct final answer.")
        pieces.append("Return the final answer as <answer>...</answer>.")
        pieces.append("Assistant:")
        return "\n".join(pieces)

    def _resolve_generation_cfg(self, state: dict[str, Any]) -> dict[str, Any]:
        generation_cfg = dict(state.get("generation_cfg", {}))
        generation_cfg.setdefault("max_new_tokens", 128)
        generation_cfg.setdefault("temperature", 0.0)
        generation_cfg.setdefault("top_k", None)
        generation_cfg.setdefault(
            "reasoning_step_tokens",
            max(32, min(128, int(generation_cfg["max_new_tokens"]) // 4)),
        )
        generation_cfg.setdefault(
            "answer_max_new_tokens", min(64, int(generation_cfg["max_new_tokens"]))
        )
        return generation_cfg

    def _tokens_to_aux_image(self, extra_visual_tokens: Any | None) -> Image.Image | None:
        if extra_visual_tokens is None:
            return None

        if torch.is_tensor(extra_visual_tokens):
            token_array = extra_visual_tokens.detach().float().cpu().numpy()
        else:
            token_array = np.asarray(extra_visual_tokens)

        if token_array.ndim != 2 or token_array.size == 0:
            return None

        num_tokens, token_dim = token_array.shape
        side = int(np.ceil(np.sqrt(num_tokens)))
        rgb = token_array[:, :3]
        if token_dim < 3:
            rgb = np.pad(rgb, ((0, 0), (0, 3 - token_dim)), mode="constant")

        rgb = rgb.astype(np.float32)
        minimum = rgb.min(axis=0, keepdims=True)
        maximum = rgb.max(axis=0, keepdims=True)
        scale = np.where((maximum - minimum) > 1e-6, maximum - minimum, 1.0)
        normalized = ((rgb - minimum) / scale * 255.0).clip(0, 255).astype(np.uint8)

        canvas = np.zeros((side * side, 3), dtype=np.uint8)
        canvas[:num_tokens] = normalized
        return Image.fromarray(canvas.reshape(side, side, 3), mode="RGB")

    def _load_pil_image(self, image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
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

            return Image.fromarray(arr).convert("RGB")
        if isinstance(image, (str, os.PathLike)):
            return Image.open(image).convert("RGB")
        if isinstance(image, (bytes, bytearray)):
            return Image.open(io.BytesIO(image)).convert("RGB")
        raise TypeError(
            "Unsupported image type. Expected PIL.Image.Image, numpy.ndarray, path-like, or bytes."
        )

    def _extract_visual_tokens(self, image: Any) -> np.ndarray:
        pil_image = self._load_pil_image(image)
        resized = pil_image.resize((224, 224))
        image_array = np.asarray(resized, dtype=np.float32) / 255.0
        patch_size = 16
        hidden_size = int(getattr(getattr(self.model, "config", None), "hidden_size", None) or 768)
        tokens: list[np.ndarray] = []

        for top in range(0, image_array.shape[0], patch_size):
            for left in range(0, image_array.shape[1], patch_size):
                patch = image_array[top : top + patch_size, left : left + patch_size, :]
                if patch.size == 0:
                    continue
                patch_mean = patch.mean(axis=(0, 1))
                patch_std = patch.std(axis=(0, 1))
                base_vector = np.concatenate(
                    [patch_mean, patch_std, np.array([top / 224.0, left / 224.0], dtype=np.float32)]
                ).astype(np.float32)
                repeats = int(np.ceil(hidden_size / base_vector.shape[0]))
                token = np.tile(base_vector, repeats)[:hidden_size].astype(np.float32)
                tokens.append(token)

        if not tokens:
            return np.zeros((1, hidden_size), dtype=np.float32)
        return np.stack(tokens, axis=0)

    def start_reasoning(
        self,
        question: str,
        image: Any,
        choices: list[str] | None,
        prompt_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        visual_tokens = self._extract_visual_tokens(image)
        return {
            "question": self._normalize_question(question),
            "image": image,
            "choices": self._normalize_choices(choices),
            "prompt_cfg": dict(prompt_cfg),
            "visual_features": visual_tokens,
            "visual_tokens": visual_tokens,
            "current_visual_features": visual_tokens,
            "current_visual_tokens": None,
            "reasoning_steps": [],
            "last_reasoning_embeddings": None,
            "raw_reasoning_steps": [],
            "raw_final_answer": "",
            "generation_cfg": {},
        }

    def generate_reasoning_step(
        self,
        state: dict[str, Any],
        extra_visual_tokens: Any | None = None,
        reflection_instruction: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        generation_cfg = self._resolve_generation_cfg(state)
        prompt = self._build_reasoning_prompt(state, reflection_instruction)
        if extra_visual_tokens is not None:
            state["current_visual_tokens"] = extra_visual_tokens
            state["current_visual_features"] = extra_visual_tokens
        raw_text = self.generate_per_token(
            prompt,
            state["image"],
            state.get("choices"),
            extra_visual_tokens=extra_visual_tokens,
            max_new_tokens=int(generation_cfg["reasoning_step_tokens"]),
            temperature=float(generation_cfg["temperature"]),
            top_k=generation_cfg.get("top_k"),
        )
        match = REASONING_TAG_RE.findall(raw_text)
        if match:
            step_text = match[-1].strip()
        else:
            lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
            step_text = lines[-1] if lines else raw_text.strip()

        # store minimal embeddings placeholder (zeros)
        hidden_size = (
            getattr(getattr(self.model, "config", None), "hidden_size", None) or 768
        )
        state["last_reasoning_embeddings"] = np.zeros(
            (1, int(hidden_size)), dtype=np.float32
        )
        state["raw_reasoning_steps"].append(raw_text)
        state["reasoning_steps"].append(step_text)
        return step_text, state

    def extract_reasoning_embeddings(self, state: dict[str, Any]) -> np.ndarray:
        emb = state.get("last_reasoning_embeddings")
        if emb is None:
            hidden_size = (
                getattr(getattr(self.model, "config", None), "hidden_size", None) or 768
            )
            return np.zeros((1, int(hidden_size)), dtype=np.float32)
        return np.asarray(emb, dtype=np.float32)

    def estimate_answer_distribution(
        self, state: dict[str, Any], choices: list[str] | None = None
    ) -> np.ndarray:
        resolved_choices = self._normalize_choices(choices) or state.get("choices")
        prompt = self._build_answer_prompt(state)
        gen_cfg = self._resolve_generation_cfg(state)
        extra_visual_tokens = state.get("current_visual_tokens")
        if resolved_choices:
            # conservative heuristic: generate final answer and compare
            pred = self.generate_full_answer(
                prompt,
                state["image"],
                None,
                extra_visual_tokens=extra_visual_tokens,
                max_new_tokens=int(gen_cfg["answer_max_new_tokens"]),
                temperature=float(gen_cfg["temperature"]),
            )
            probs = np.full(
                (len(resolved_choices),), 1.0 / len(resolved_choices), dtype=np.float32
            )
            for i, c in enumerate(resolved_choices):
                if str(pred).strip() == str(c).strip():
                    probs = np.full(
                        (len(resolved_choices),),
                        0.05 / max(1, len(resolved_choices) - 1),
                        dtype=np.float32,
                    )
                    probs[i] = 0.95
                    break
            return probs
        # no choices: return top-k next-token probs
        logits = self.get_next_token_logits(
            prompt,
            state["image"],
            state.get("choices"),
            extra_visual_tokens=extra_visual_tokens,
        )
        k = min(8, logits.shape[-1])
        top_vals, _ = torch.topk(logits, k=k, dim=-1)
        probs = (
            torch.softmax(top_vals.float(), dim=-1)[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        return probs

    def generate_final_answer(
        self, state: dict[str, Any], choices: list[str] | None = None
    ) -> str:
        resolved_choices = self._normalize_choices(choices) or state.get("choices")
        if resolved_choices is not None:
            state["choices"] = resolved_choices
        gen_cfg = self._resolve_generation_cfg(state)
        prompt = self._build_answer_prompt(state)
        extra_visual_tokens = state.get("current_visual_tokens")
        raw = self.generate_per_token(
            prompt,
            state["image"],
            state.get("choices"),
            extra_visual_tokens=extra_visual_tokens,
            max_new_tokens=int(gen_cfg["answer_max_new_tokens"]),
            temperature=float(gen_cfg["temperature"]),
            top_k=gen_cfg.get("top_k"),
        )
        state["raw_final_answer"] = raw
        final = extract_answer_text(raw)
        return final or raw.strip()

    def get_next_token_logits(
        self,
        question: str,
        image: Any,
        choices: list[str] | None = None,
        extra_visual_tokens: Any | None = None,
    ) -> torch.Tensor:
        """Run one forward pass and return next-token logits with shape [B, vocab]."""

        inputs = self.prepare_inputs(question, image, choices, extra_visual_tokens)

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
        extra_visual_tokens: Any | None = None,
    ) -> str:
        """Autoregressive decoding via explicit forward calls at every token step."""

        model_inputs = self.prepare_inputs(question, image, choices, extra_visual_tokens)

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

            print(
                "[generate_per_token] Attempting to decode with FlashAttention cache..."
            )
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
