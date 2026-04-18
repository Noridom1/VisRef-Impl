from __future__ import annotations

import re
from typing import Any
import logging

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

from eval.metrics import extract_answer_text

from .base_wrapper import BaseModelWrapper


REASONING_TAG_RE = re.compile(
    r"<reasoning_step>\s*(.*?)\s*</reasoning_step>",
    re.IGNORECASE | re.DOTALL,
)

logger = logging.getLogger(__name__)


class InternVL(BaseModelWrapper):
    """Stateful InternVL wrapper for ST/TSR/VisRef style decoding."""

    def __init__(self, model_cfg: dict[str, Any]) -> None:
        self.model_cfg = model_cfg
        self.device = torch.device(model_cfg.get("device", "cuda"))
        self.model_dtype = self._resolve_dtype(model_cfg.get("dtype", "float16"))

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_cfg["hf_repo_or_local_path"],
                trust_remote_code=True,
                use_fast=False,
            )
        except Exception as exc:
            repo = model_cfg.get("hf_repo_or_local_path", "<unknown>")
            raise RuntimeError(
                "Failed to initialize tokenizer for "
                f"{repo}. This usually means one of: "
                "(1) missing sentencepiece/protobuf, "
                "(2) corrupted HuggingFace cache for tokenizer files, or "
                "(3) incompatible transformers version for this checkpoint. "
                "Install sentencepiece + protobuf and retry. If it still fails, "
                "clear the cached model directory and re-download. "
                f"Original error: {exc}"
            ) from exc

        device_map = model_cfg.get("device_map")
        resolved_device_map = device_map
        model_load_kwargs = {
            "torch_dtype": self.model_dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "device_map": device_map,
        }

        try:
            self.model = AutoModel.from_pretrained(
                model_cfg["hf_repo_or_local_path"],
                **model_load_kwargs,
            )
        except (AttributeError, ValueError, RuntimeError) as exc:
            message = str(exc).lower()
            fallback_needed = (
                "all_tied_weights_keys" in message
                or "requires `accelerate`" in message
                or "meta tensor" in message
                or "tensor.item() cannot be called on meta tensors" in message
            )
            if not fallback_needed:
                raise

            if device_map not in (None, "", "none"):
                logger.warning(
                    "Model loading with device_map=%s failed (%s). "
                    "Retrying with device_map=None.",
                    device_map,
                    exc,
                )
                model_load_kwargs["device_map"] = None
                resolved_device_map = None

            try:
                self.model = AutoModel.from_pretrained(
                    model_cfg["hf_repo_or_local_path"],
                    **model_load_kwargs,
                )
            except (AttributeError, ValueError, RuntimeError) as exc_second:
                second_message = str(exc_second).lower()
                if (
                    "all_tied_weights_keys" not in second_message
                    and "meta tensor" not in second_message
                    and "tensor.item() cannot be called on meta tensors" not in second_message
                ):
                    raise
                logger.warning(
                    "Model loading still failed with all_tied_weights_keys bug (%s). "
                    "Retrying with low_cpu_mem_usage=False and device_map=None.",
                    exc_second,
                )
                model_load_kwargs["low_cpu_mem_usage"] = False
                model_load_kwargs["device_map"] = None
                resolved_device_map = None
                self.model = AutoModel.from_pretrained(
                    model_cfg["hf_repo_or_local_path"],
                    **model_load_kwargs,
                )

        if resolved_device_map in (None, "", "none"):
            self.model = self.model.to(self.device)
        self.model.eval()

    def _resolve_dtype(self, dtype_name: str | None) -> torch.dtype:
        normalized = str(dtype_name or "float16").lower()
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return mapping.get(normalized, torch.float16)

    @property
    def language_model(self):
        return getattr(self.model, "language_model", self.model)

    @property
    def text_device(self) -> torch.device:
        return self._get_input_embeddings().weight.device

    def _get_input_embeddings(self):
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings()
        return self.language_model.get_input_embeddings()

    def _get_hidden_size(self) -> int:
        for config_obj in [
            getattr(self.language_model, "config", None),
            getattr(self.model, "config", None),
        ]:
            if config_obj is None:
                continue
            if hasattr(config_obj, "hidden_size"):
                return int(config_obj.hidden_size)
            text_config = getattr(config_obj, "text_config", None)
            if text_config is not None and hasattr(text_config, "hidden_size"):
                return int(text_config.hidden_size)
        raise RuntimeError("Could not resolve the InternVL language-model hidden size.")

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

    def _load_image(self, image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)!r}")

    def _prepare_pixel_values(self, image: Any) -> torch.Tensor:
        pil_image = self._load_image(image)
        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        pixel_values = transform(pil_image).unsqueeze(0)
        return pixel_values.to(self.text_device, dtype=self.model_dtype)

    def _ensure_img_context_token_id(self) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        unk_id = self.tokenizer.unk_token_id
        if token_id is None or token_id < 0:
            raise RuntimeError("Tokenizer does not provide <IMG_CONTEXT> token id.")
        if unk_id is not None and token_id == unk_id:
            raise RuntimeError(
                "<IMG_CONTEXT> resolves to unk_token_id; tokenizer/model pair is incompatible."
            )
        setattr(self.model, "img_context_token_id", int(token_id))
        return int(token_id)

    def _flatten_visual_features(self, visual_features: torch.Tensor) -> torch.Tensor:
        if visual_features.ndim == 3:
            if visual_features.shape[0] != 1:
                raise RuntimeError(
                    f"Expected a single image batch, got shape {tuple(visual_features.shape)}"
                )
            visual_features = visual_features[0]
        if visual_features.ndim != 2:
            raise RuntimeError(
                f"Expected visual features rank 2 after flattening, got shape {tuple(visual_features.shape)}"
            )
        return visual_features

    def _extract_visual_features(self, image: Any) -> torch.Tensor:
        pixel_values = self._prepare_pixel_values(image)
        with torch.inference_mode():
            visual_features = self.model.extract_feature(pixel_values)
        visual_features = self._flatten_visual_features(visual_features)
        return visual_features.to(self.text_device)

    def _format_choices_block(self, choices: list[str] | None) -> str:
        if not choices:
            return ""
        return "Choices:\n" + "\n".join(f"- {choice}" for choice in choices)

    def _build_reasoning_prompt(
        self,
        state: dict[str, Any],
        reflection_instruction: str | None = None,
    ) -> str:
        prompt_cfg = state["prompt_cfg"]
        pieces = []
        system_prompt = prompt_cfg.get("system")
        if system_prompt:
            pieces.append(f"System: {system_prompt}")
        pieces.append("User:")
        pieces.append("<image>")
        pieces.append(f"Question: {state['question']}")
        choices_block = self._format_choices_block(state["choices"])
        if choices_block:
            pieces.append(choices_block)
        if state["reasoning_steps"]:
            pieces.append("Reasoning so far:")
            for idx, step in enumerate(state["reasoning_steps"], start=1):
                pieces.append(f"{idx}. {step}")
        instruction = reflection_instruction or prompt_cfg.get(
            "think_instruction",
            "Think step by step before giving the final answer.",
        )
        pieces.append(f"Instruction: {instruction}")
        pieces.append(
            "Write exactly one new reasoning step and wrap it in "
            "<reasoning_step>...</reasoning_step>. Do not give the final answer."
        )
        pieces.append("Assistant:")
        return "\n".join(pieces)

    def _build_answer_prompt(self, state: dict[str, Any]) -> str:
        prompt_cfg = state["prompt_cfg"]
        pieces = []
        system_prompt = prompt_cfg.get("system")
        if system_prompt:
            pieces.append(f"System: {system_prompt}")
        pieces.append("User:")
        pieces.append("<image>")
        pieces.append(f"Question: {state['question']}")
        choices_block = self._format_choices_block(state["choices"])
        if choices_block:
            pieces.append(choices_block)
        if state["reasoning_steps"]:
            pieces.append("Reasoning:")
            for idx, step in enumerate(state["reasoning_steps"], start=1):
                pieces.append(f"{idx}. {step}")
        if state["choices"]:
            pieces.append(
                "Select the best option from the provided choices and copy it exactly."
            )
        else:
            pieces.append("Provide the shortest correct final answer.")
        pieces.append("Return the final answer as <answer>...</answer>.")
        pieces.append("Assistant:")
        return "\n".join(pieces)

    def _build_prompt_with_image_tokens(
        self,
        prompt: str,
        visual_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_context_token_id = self._ensure_img_context_token_id()
        visual_features = self._flatten_visual_features(visual_features).to(
            self.text_device,
            dtype=self.model_dtype,
        )
        num_visual_tokens = int(visual_features.shape[0])
        image_token_block = "<img>" + ("<IMG_CONTEXT>" * num_visual_tokens) + "</img>"
        prompt_text = prompt if "<image>" in prompt else f"<image>\n{prompt}"
        prompt_text = prompt_text.replace("<image>", image_token_block, 1)

        tokenized = self.tokenizer(prompt_text, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(self.text_device)
        attention_mask = tokenized.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.text_device)
        else:
            attention_mask = attention_mask.to(self.text_device)

        input_embeds = self._get_input_embeddings()(input_ids).clone()
        selected = input_ids.reshape(-1) == img_context_token_id
        if int(selected.sum().item()) != num_visual_tokens:
            raise RuntimeError(
                "Prompt/image token mismatch: "
                f"found {int(selected.sum().item())} image tokens, expected {num_visual_tokens}."
            )
        flat_embeds = input_embeds.reshape(-1, input_embeds.shape[-1])
        flat_embeds[selected] = visual_features.reshape(-1, flat_embeds.shape[-1]).to(
            flat_embeds.dtype)
        input_embeds = flat_embeds.reshape_as(input_embeds)
        return input_ids, attention_mask, input_embeds

    def _forward_sequence(
        self,
        input_ids: torch.Tensor,
        visual_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.text_device)
        input_embeds = self._get_input_embeddings()(input_ids.to(self.text_device)).clone()
        img_context_token_id = self._ensure_img_context_token_id()
        selected = input_ids.reshape(-1).to(self.text_device) == img_context_token_id
        flat_embeds = input_embeds.reshape(-1, input_embeds.shape[-1])
        flat_embeds[selected] = self._flatten_visual_features(visual_features).to(
            self.text_device,
            dtype=flat_embeds.dtype,
        )
        input_embeds = flat_embeds.reshape_as(input_embeds)
        return self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask.to(self.text_device),
            use_cache=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int | None,
    ) -> torch.Tensor:
        if temperature <= 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        step_logits = logits / temperature
        if top_k is not None and top_k > 0:
            k = min(top_k, step_logits.shape[-1])
            top_vals, top_idx = torch.topk(step_logits, k=k, dim=-1)
            probs = torch.softmax(top_vals, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)
            return top_idx.gather(-1, sampled)

        probs = torch.softmax(step_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _trim_on_stop_strings(self, text: str, stop_strings: list[str]) -> str:
        trimmed = text
        for stop_string in stop_strings:
            if stop_string in trimmed:
                trimmed = trimmed.split(stop_string, 1)[0]
        return trimmed

    def _decode_new_tokens(
        self,
        prompt: str,
        visual_features: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        stop_strings: list[str],
    ) -> tuple[str, torch.Tensor, torch.Tensor]:
        prefix_ids, attention_mask, prefix_embeds = self._build_prompt_with_image_tokens(
            prompt,
            visual_features,
        )
        generated_tokens: list[torch.Tensor] = []
        past_key_values = None
        step_input_ids: torch.Tensor | None = None
        step_input_embeds: torch.Tensor | None = prefix_embeds
        running_attention = attention_mask
        stop_token_ids = {
            token_id
            for token_id in [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            ]
            if token_id is not None and token_id >= 0
        }

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                kwargs: dict[str, Any] = {
                    "attention_mask": running_attention,
                    "use_cache": True,
                    "return_dict": True,
                }
                if past_key_values is None:
                    kwargs["inputs_embeds"] = step_input_embeds
                else:
                    kwargs["input_ids"] = step_input_ids
                    kwargs["past_key_values"] = past_key_values

                outputs = self.language_model(**kwargs)
                logits = outputs.logits[:, -1, :]
                next_token = self._sample_next_token(logits, temperature, top_k)
                generated_tokens.append(next_token)
                past_key_values = outputs.past_key_values
                step_input_ids = next_token
                step_input_embeds = None

                ones = torch.ones_like(next_token, dtype=running_attention.dtype)
                running_attention = torch.cat([running_attention, ones], dim=-1)

                decoded = self.tokenizer.decode(
                    torch.cat(generated_tokens, dim=-1)[0],
                    skip_special_tokens=False,
                )
                if any(stop_string in decoded for stop_string in stop_strings):
                    break
                if stop_token_ids and int(next_token.item()) in stop_token_ids:
                    break

        if generated_tokens:
            generated_ids = torch.cat(generated_tokens, dim=-1)
            decoded = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        else:
            generated_ids = torch.empty((1, 0), dtype=prefix_ids.dtype, device=self.text_device)
            decoded = ""
        return self._trim_on_stop_strings(decoded, stop_strings), prefix_ids, generated_ids

    def _extract_generated_hidden_states(
        self,
        prefix_ids: torch.Tensor,
        generated_ids: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> torch.Tensor:
        if generated_ids.numel() == 0:
            hidden_size = self._get_hidden_size()
            return torch.zeros((1, hidden_size), dtype=torch.float32)

        full_input_ids = torch.cat([prefix_ids, generated_ids.to(prefix_ids.device)], dim=-1)
        full_attention = torch.ones_like(full_input_ids, device=self.text_device)
        outputs = self._forward_sequence(
            full_input_ids.to(self.text_device),
            visual_features,
            attention_mask=full_attention,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1][:, prefix_ids.shape[-1]:, :]
        return hidden_states.squeeze(0).detach().to(torch.float32).cpu()

    def _extract_reasoning_step_text(self, raw_text: str) -> str:
        matches = REASONING_TAG_RE.findall(raw_text)
        if matches:
            return matches[-1].strip()
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if lines:
            return lines[-1]
        return raw_text.strip()

    def _next_token_probs(
        self,
        prompt: str,
        visual_features: torch.Tensor,
        top_k: int = 8,
    ) -> np.ndarray:
        input_ids, attention_mask, input_embeds = self._build_prompt_with_image_tokens(
            prompt,
            visual_features,
        )
        with torch.inference_mode():
            outputs = self.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        logits = outputs.logits[:, -1, :]
        k = min(top_k, logits.shape[-1])
        top_values, _ = torch.topk(logits, k=k, dim=-1)
        probs = torch.softmax(top_values, dim=-1)[0].detach().cpu().numpy()
        return probs.astype(np.float32)

    def _score_choice_candidates(
        self,
        prompt: str,
        visual_features: torch.Tensor,
        choices: list[str],
    ) -> np.ndarray:
        prefix_ids, _, _ = self._build_prompt_with_image_tokens(prompt, visual_features)
        prefix_len = prefix_ids.shape[-1]
        scores: list[float] = []

        for choice in choices:
            candidate = f"<answer>{choice}</answer>"
            candidate_ids = self.tokenizer(
                candidate,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].to(self.text_device)
            full_input_ids = torch.cat([prefix_ids, candidate_ids], dim=-1)
            full_attention = torch.ones_like(full_input_ids, device=self.text_device)
            outputs = self._forward_sequence(
                full_input_ids,
                visual_features,
                attention_mask=full_attention,
                output_hidden_states=False,
            )
            logits = outputs.logits[:, prefix_len - 1:-1, :]
            target_ids = full_input_ids[:, prefix_len:]
            log_probs = torch.log_softmax(logits, dim=-1)
            token_scores = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            scores.append(float(token_scores.sum().item()))

        score_tensor = torch.tensor(scores, dtype=torch.float32)
        probs = torch.softmax(score_tensor, dim=0).cpu().numpy()
        return probs.astype(np.float32)

    def _default_prompt_cfg(self) -> dict[str, Any]:
        return {
            "system": self.model_cfg.get("system_prompt", ""),
            "think_instruction": "Think step by step before giving the final answer.",
            "reflection_instruction": "Wait. Think more carefully using visual evidence.",
        }

    def _resolve_generation_cfg(self, state: dict[str, Any]) -> dict[str, Any]:
        generation_cfg = dict(state.get("generation_cfg", {}))
        if "max_new_tokens" not in generation_cfg:
            generation_cfg["max_new_tokens"] = 128
        if "temperature" not in generation_cfg:
            generation_cfg["temperature"] = 0.0
        if "top_k" not in generation_cfg:
            generation_cfg["top_k"] = None
        if "reasoning_step_tokens" not in generation_cfg:
            generation_cfg["reasoning_step_tokens"] = max(
                32,
                min(128, int(generation_cfg["max_new_tokens"]) // 4),
            )
        if "answer_max_new_tokens" not in generation_cfg:
            generation_cfg["answer_max_new_tokens"] = min(
                64,
                int(generation_cfg["max_new_tokens"]),
            )
        return generation_cfg

    def start_reasoning(
        self,
        question: str,
        image: Any,
        choices: list[str] | None,
        prompt_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        visual_features = self._extract_visual_features(image)
        return {
            "question": self._normalize_question(question),
            "image": image,
            "choices": self._normalize_choices(choices),
            "prompt_cfg": dict(prompt_cfg),
            "visual_features": visual_features,
            "visual_tokens": visual_features.detach().cpu().numpy().astype(np.float32),
            "current_visual_features": visual_features,
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
        visual_features = extra_visual_tokens
        if visual_features is None:
            visual_features = state["visual_features"]
        visual_features = self._flatten_visual_features(
            torch.as_tensor(visual_features, device=self.text_device)
        ).to(self.model_dtype)

        prompt = self._build_reasoning_prompt(state, reflection_instruction)
        raw_text, prefix_ids, generated_ids = self._decode_new_tokens(
            prompt=prompt,
            visual_features=visual_features,
            max_new_tokens=int(generation_cfg["reasoning_step_tokens"]),
            temperature=float(generation_cfg["temperature"]),
            top_k=generation_cfg.get("top_k"),
            stop_strings=["</reasoning_step>"],
        )
        step_text = self._extract_reasoning_step_text(raw_text)
        reasoning_embeddings = self._extract_generated_hidden_states(
            prefix_ids,
            generated_ids,
            visual_features,
        )

        state["current_visual_features"] = visual_features
        state["raw_reasoning_steps"].append(raw_text)
        state["reasoning_steps"].append(step_text)
        state["last_reasoning_embeddings"] = reasoning_embeddings.numpy()
        return step_text, state

    def extract_reasoning_embeddings(self, state: dict[str, Any]) -> np.ndarray:
        embeddings = state.get("last_reasoning_embeddings")
        if embeddings is None:
            hidden_size = self._get_hidden_size()
            return np.zeros((1, hidden_size), dtype=np.float32)
        return np.asarray(embeddings, dtype=np.float32)

    def estimate_answer_distribution(
        self,
        state: dict[str, Any],
        choices: list[str] | None = None,
    ) -> np.ndarray:
        resolved_choices = self._normalize_choices(choices) or state["choices"]
        prompt = self._build_answer_prompt(state)
        visual_features = state.get("current_visual_features", state["visual_features"])

        if resolved_choices:
            return self._score_choice_candidates(prompt, visual_features, resolved_choices)
        return self._next_token_probs(prompt, visual_features)

    def generate_final_answer(
        self,
        state: dict[str, Any],
        choices: list[str] | None = None,
    ) -> str:
        resolved_choices = self._normalize_choices(choices) or state["choices"]
        if resolved_choices is not None:
            state["choices"] = resolved_choices

        generation_cfg = self._resolve_generation_cfg(state)
        prompt = self._build_answer_prompt(state)
        visual_features = state.get("current_visual_features", state["visual_features"])
        raw_text, _, _ = self._decode_new_tokens(
            prompt=prompt,
            visual_features=visual_features,
            max_new_tokens=int(generation_cfg["answer_max_new_tokens"]),
            temperature=float(generation_cfg["temperature"]),
            top_k=generation_cfg.get("top_k"),
            stop_strings=["</answer>"],
        )
        state["raw_final_answer"] = raw_text
        final_answer = extract_answer_text(raw_text)
        return final_answer or raw_text.strip()

    def generate_full_answer(
        self,
        question: str,
        image: Any,
        choices: list[str] | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int | None = None,
    ) -> str:
        state = self.start_reasoning(
            question=question,
            image=image,
            choices=choices,
            prompt_cfg=self._default_prompt_cfg(),
        )
        state["generation_cfg"] = {
            "max_new_tokens": max_new_tokens,
            "answer_max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
        }
        return self.generate_final_answer(state, choices)
