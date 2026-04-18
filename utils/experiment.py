from __future__ import annotations

from copy import deepcopy
from typing import Any


def load_dataset(dataset_cfg: dict[str, Any]):
    from dataset import MMStarDataset, MathVisionDataset, MathVistaDataset

    name = dataset_cfg["name"].lower()
    common = {
        "root": dataset_cfg["root"],
        "split": dataset_cfg["split"],
        "image_key": dataset_cfg["image_key"],
        "question_key": dataset_cfg["question_key"],
        "answer_key": dataset_cfg["answer_key"],
        "choice_key": dataset_cfg.get("choice_key", None),
    }
    if name == "mathvista":
        return MathVistaDataset(**common)
    if name == "mathvision":
        return MathVisionDataset(**common)
    if name == "mmstar":
        return MMStarDataset(**common)
    raise ValueError(f"Unsupported dataset: {name}")


def load_model_wrapper(model_cfg: dict[str, Any],
                       wrapper_cfg: dict[str, Any] | None = None):
    from models import InternVL, Qwen

    wrapper_name = str((wrapper_cfg or {}).get("class_name", "")).lower()
    model_name = str(model_cfg.get("name", "")).lower()
    selector = wrapper_name or model_name

    if "internvl" in selector:
        return InternVL(model_cfg)
    if "qwen" in selector:
        if Qwen is None:
            raise ImportError(
                "Qwen wrapper is unavailable in the current environment. "
                "Install a transformers version that provides Qwen3-VL classes "
                "or switch model config to InternVL."
            )
        return Qwen(model_cfg)
    raise ValueError(
        "Unsupported model/wrapper name. Add a wrapper in models and map it in load_model_wrapper."
    )


def merge_eval_cfg(
    default_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    mode: str,
    output_dir: str,
) -> dict[str, Any]:
    merged = deepcopy(default_cfg)
    merged["model"] = model_cfg["model"]
    merged["prompt"] = model_cfg["prompt"]
    merged["wrapper"] = model_cfg.get("wrapper", {})
    merged["dataset"] = dataset_cfg["dataset"]
    merged["run"] = {"mode": mode}
    merged.setdefault("logging", {})["output_dir"] = output_dir
    return merged
