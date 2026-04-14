from __future__ import annotations

from collections.abc import Callable
import re
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModel, LogitsProcessor, LogitsProcessorList
import numpy as np
from PIL import Image
from torchvision import transforms


from .base_wrapper import BaseModelWrapper


class InternVL(BaseModelWrapper):
    """Starter InternVL wrapper.

    Replace placeholder logic with actual InternVL model loading and generation APIs.
    """

    def __init__(self, model_cfg: dict[str, Any]) -> None:
        self.model_cfg = model_cfg
        self.device = model_cfg["device"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["hf_repo_or_local_path"],
            trust_remote_code=True,
            use_fast=False,
            force_download=True,
        )

        self.model = AutoModel.from_pretrained(
            model_cfg["hf_repo_or_local_path"],
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            device_map=None
        ).to("cuda")


    def prepare_inputs(self, text: str, image: Any, choices: list[str] | None = None) -> tuple[str, torch.Tensor]:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # InternVL image transform (simplified)
        transform = transforms.Compose([
            transforms.Resize((448, 448)),  # typical for InternVL
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        pixel_values = transform(image).unsqueeze(0).to(self.device).half()

        if choices is not None:
            text += " Choices: " + ", ".join(choices)

        # Normalize dataset placeholders such as <image1>, <image2>, ... to InternVL's <image>.
        normalized_text = re.sub(r"<image\d+>", "<image>", text)

        # Keep at most one image marker for this single-image wrapper.
        if "<image>" in normalized_text:
            parts = normalized_text.split("<image>")
            normalized_text = parts[0] + "<image>" + "".join(parts[1:])

        # InternVL expects an explicit <image> marker in the user question.
        if "<image>" not in normalized_text:
            prompt = f"<image>\n{normalized_text}"
        else:
            prompt = normalized_text


        return prompt, pixel_values

    def _build_single_forward_inputs(
        self,
        prompt: str,
        pixel_values: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        img_context_token_id = self._ensure_img_context_token_id()

        num_image_token = self._get_expected_image_token_count(pixel_values)
        if num_image_token <= 0:
            raise RuntimeError("Cannot build InternVL forward inputs: invalid image token count.")

        image_token_block = "<img>" + ("<IMG_CONTEXT>" * num_image_token) + "</img>"
        prompt_for_forward = prompt.replace("<image>", image_token_block, 1)
        tokenized = self.tokenizer(prompt_for_forward, return_tensors="pt").to(self.device)

        found_image_tokens = int((tokenized["input_ids"] == img_context_token_id).sum().item())
        expected_image_tokens = int(num_image_token * pixel_values.shape[0])
        if found_image_tokens != expected_image_tokens:
            raise RuntimeError(
                "Manual InternVL prompt/image token mismatch: "
                f"found {found_image_tokens} <IMG_CONTEXT> ids, expected {expected_image_tokens}. "
                "Ensure tokenizer contains <IMG_CONTEXT> as a special token and matches the model checkpoint."
            )

        model_inputs: dict[str, torch.Tensor] = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized.get("attention_mask"),
            "pixel_values": pixel_values,
        }

        # Some InternVL builds require image_flags in forward.
        model_inputs["image_flags"] = torch.ones(
            (pixel_values.shape[0], 1),
            dtype=torch.long,
            device=self.device,
        )
        return model_inputs

    def _ensure_img_context_token_id(self) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        unk_id = self.tokenizer.unk_token_id
        if token_id is None or token_id < 0:
            raise RuntimeError("Tokenizer does not provide <IMG_CONTEXT> token id.")
        if unk_id is not None and token_id == unk_id:
            raise RuntimeError(
                "<IMG_CONTEXT> resolves to unk_token_id; tokenizer/model pair is incompatible for manual InternVL forward."
            )

        setattr(self.model, "img_context_token_id", int(token_id))
        return int(token_id)

    def _get_expected_image_token_count(self, pixel_values: torch.Tensor) -> int:
        # Prefer deriving from the actual visual features so token count matches vit_embeds.
        extract_feature = getattr(self.model, "extract_feature", None)
        if callable(extract_feature):
            with torch.inference_mode():
                vit_embeds = extract_feature(pixel_values)
            if not isinstance(vit_embeds, torch.Tensor):
                raise RuntimeError("model.extract_feature did not return a tensor.")

            if vit_embeds.ndim == 3:
                return int(vit_embeds.shape[1])
            if vit_embeds.ndim == 2:
                return int(vit_embeds.shape[0])

            raise RuntimeError(
                f"Unexpected vit_embeds rank={vit_embeds.ndim}, shape={tuple(vit_embeds.shape)}"
            )

        fallback = getattr(self.model, "num_image_token", None)
        if isinstance(fallback, int) and fallback > 0:
            return fallback

        raise RuntimeError(
            "Cannot infer expected image token count; model has neither extract_feature nor valid num_image_token."
        )

    def get_next_token_logits(self, question: str, image: Any, choices: list[str] | None = None) -> torch.Tensor:
        """Run exactly one forward pass and return next-token logits [B, vocab]."""
        prompt, pixel_values = self.prepare_inputs(question, image, choices)
        model_inputs = self._build_single_forward_inputs(prompt, pixel_values)

        self.model.eval()
        with torch.inference_mode():
            try:
                outputs = self.model(**model_inputs)
            except TypeError:
                # Compatibility fallback for variants that do not expose image_flags.
                fallback_inputs = dict(model_inputs)
                fallback_inputs.pop("image_flags", None)
                outputs = self.model(**fallback_inputs)

        return outputs.logits[:, -1, :]

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

    def get_single_token_id(
        self,
        token_text: str = "Wait",
        extra_candidates: list[str] | None = None,
    ) -> int:
        """Resolve a text into one tokenizer id, trying common whitespace variants."""
        candidates: list[str] = [f" {token_text}", token_text, token_text.lower(), token_text.upper()]
        if extra_candidates:
            candidates = list(extra_candidates) + candidates

        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)

            ids = self.tokenizer.encode(candidate, add_special_tokens=False)
            if len(ids) == 1:
                token_id = int(ids[0])
                decoded = self.tokenizer.decode([token_id])
                if decoded.replace(" ", "") == candidate.replace(" ", ""):
                    print(f"[get_single_token_id] candidate={repr(candidate)} -> id={token_id} -> decoded={repr(decoded)}")
                    return token_id
                print(
                    f"[get_single_token_id] candidate={repr(candidate)} -> id={token_id} but decoded={repr(decoded)}; skipped"
                )

        raise ValueError(
            f"Could not map token_text={token_text!r} to a single token id. "
            "Try passing extra_candidates (e.g. variants with leading spaces)."
        )

    def _get_replaceable_stop_token_ids(self) -> set[int]:
        stop_token_ids: set[int] = set()

        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None:
            stop_token_ids.add(int(eos_token_id))

        im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_token_id is not None and im_end_token_id >= 0:
            stop_token_ids.add(int(im_end_token_id))

        return stop_token_ids

    def _build_token_hook_processor(
        self,
        token_hook: Callable[[dict[str, Any]], torch.Tensor | None] | None,
    ) -> LogitsProcessorList | None:
        if token_hook is None:
            return None

        tokenizer = self.tokenizer

        class _TokenHookProcessor(LogitsProcessor):
            def __init__(self) -> None:
                self.step = 0

            def __call__(
                self,
                input_ids: torch.LongTensor,
                scores: torch.FloatTensor,
            ) -> torch.FloatTensor:
                result = token_hook(
                    {
                        "step": self.step,
                        "input_ids": input_ids,
                        "logits": scores,
                        "tokenizer": tokenizer,
                    }
                )
                self.step += 1

                if result is None:
                    return scores

                if not isinstance(result, torch.Tensor):
                    raise TypeError("token_hook must return None or a torch.Tensor")
                if result.shape != scores.shape:
                    raise ValueError(
                        f"token_hook returned shape {tuple(result.shape)}, expected {tuple(scores.shape)}"
                    )

                return result

        return LogitsProcessorList([_TokenHookProcessor()])

    def generate_per_token(
        self,
        question: str,
        image: Any,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int | None = None,
        token_hook: Callable[[dict[str, Any]], torch.Tensor | None] | None = None,
        force_wait_before_max: bool = False,
        wait_token_text: str = "Wait",
        wait_token_candidates: list[str] | None = None,
        replace_only_when_eos_is_argmax: bool = True,
    ) -> str:
        prompt, pixel_values = self.prepare_inputs(question, image)
        self.model.eval()

        model_inputs = self._build_single_forward_inputs(prompt, pixel_values)
        generated = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask")
        image_flags = model_inputs.get("image_flags")

        eos_token_id = self.tokenizer.eos_token_id
        stop_token_ids = self._get_replaceable_stop_token_ids()
        wait_token_id: int | None = None
        if force_wait_before_max:
            if eos_token_id is None:
                raise RuntimeError("Tokenizer has no eos_token_id; cannot replace EOS with Wait.")
            wait_token_id = self.get_single_token_id(
                token_text=wait_token_text,
                extra_candidates=wait_token_candidates,
            )
            print(
                "force_wait_before_max is enabled: will replace stop token ids "
                f"{sorted(stop_token_ids)} with wait_token_id {wait_token_id} until the final step."
            )

        generated_new_tokens: list[torch.Tensor] = []

        with torch.inference_mode():
            for step in range(max_new_tokens):
                forward_inputs: dict[str, Any] = {
                    "input_ids": generated,
                    "attention_mask": attention_mask,
                    "pixel_values": pixel_values,
                    "image_flags": image_flags,
                    "use_cache": False,
                }
                try:
                    outputs = self.model(**forward_inputs)
                except TypeError:
                    fallback_inputs = dict(forward_inputs)
                    fallback_inputs.pop("image_flags", None)
                    outputs = self.model(**fallback_inputs)
                logits = outputs.logits[:, -1, :]

                if token_hook is not None:
                    maybe_logits = token_hook(
                        {
                            "step": step,
                            "input_ids": generated,
                            "logits": logits,
                            "tokenizer": self.tokenizer,
                        }
                    )
                    if maybe_logits is not None:
                        if not isinstance(maybe_logits, torch.Tensor):
                            raise TypeError("token_hook must return None or a torch.Tensor")
                        if maybe_logits.shape != logits.shape:
                            raise ValueError(
                                f"token_hook returned shape {tuple(maybe_logits.shape)}, expected {tuple(logits.shape)}"
                            )
                        logits = maybe_logits

                if temperature <= 0.0:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    step_logits = logits / temperature
                    if top_k is not None and top_k > 0:
                        k = min(top_k, step_logits.shape[-1])
                        top_vals, top_idx = torch.topk(step_logits, k=k, dim=-1)
                        probs = torch.softmax(top_vals, dim=-1)
                        sampled = torch.multinomial(probs, num_samples=1)
                        next_token = top_idx.gather(-1, sampled)
                    else:
                        probs = torch.softmax(step_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)

                if force_wait_before_max and wait_token_id is not None and stop_token_ids:
                    if step < max_new_tokens - 1:
                        stop_mask = torch.zeros_like(next_token.squeeze(-1), dtype=torch.bool)
                        for stop_token_id in stop_token_ids:
                            stop_mask |= next_token.squeeze(-1) == stop_token_id
                        if torch.any(stop_mask):
                            next_token[stop_mask, 0] = wait_token_id

                generated = torch.cat([generated, next_token], dim=-1)
                generated_new_tokens.append(next_token)

                if attention_mask is not None:
                    ones = torch.ones_like(next_token, dtype=attention_mask.dtype)
                    attention_mask = torch.cat([attention_mask, ones], dim=-1)

                if eos_token_id is not None and torch.all(next_token == eos_token_id):
                    print(f"EOS token id {eos_token_id} detected at step {step}")
                    break

        if not generated_new_tokens:
            return ""

        new_token_ids = torch.cat(generated_new_tokens, dim=-1)
        token_list = new_token_ids[0].cpu().tolist()
        print(f"[generate_per_token] Generated {len(token_list)} tokens: {token_list}")
        decoded = self.tokenizer.decode(new_token_ids[0], skip_special_tokens=False)
        print(f"[generate_per_token] Decoded output: {repr(decoded)}")
        return decoded    
        
    def generate_full_answer(
        self,
        question: str,
        image: Any,
        choices: list[str] | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int | None = None,
        token_hook: Callable[[dict[str, Any]], torch.Tensor | None] | None = None,
    ) -> str:

        prompt, pixel_values = self.prepare_inputs(question, image, choices)
        self.model.eval()

        do_sample = temperature > 0.0
        generation_config: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_config["temperature"] = temperature
            if top_k is not None:
                generation_config["top_k"] = top_k

        logits_processor = self._build_token_hook_processor(token_hook)
        if logits_processor is not None:
            generation_config["logits_processor"] = logits_processor

        with torch.inference_mode():
            # Prefer InternVL's chat API because it injects image context tokens correctly.
            if hasattr(self.model, "chat"):
                try:
                    return self.model.chat(
                        self.tokenizer,
                        pixel_values,
                        prompt,
                        generation_config,
                        num_patches_list=[pixel_values.shape[0]],
                    )
                except TypeError:
                    return self.model.chat(
                        self.tokenizer,
                        pixel_values,
                        prompt,
                        generation_config,
                    )

            # Fallback for variants without `chat`.
            tokenized = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            output_ids = self.model.generate(
                **tokenized,
                pixel_values=pixel_values,
                **generation_config,
            )
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
