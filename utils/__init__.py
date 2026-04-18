"""Utility helpers with lazy imports to avoid heavy runtime dependencies."""

from __future__ import annotations

__all__ = [
    "ensure_parent",
    "read_yaml",
    "write_json",
    "write_jsonl",
    "setup_logging",
    "set_seed",
    "load_dataset",
    "load_model_wrapper",
    "merge_eval_cfg",
]


def __getattr__(name: str):
    if name in {"ensure_parent", "read_yaml", "write_json", "write_jsonl"}:
        from .io import ensure_parent, read_yaml, write_json, write_jsonl

        return {
            "ensure_parent": ensure_parent,
            "read_yaml": read_yaml,
            "write_json": write_json,
            "write_jsonl": write_jsonl,
        }[name]

    if name == "setup_logging":
        from .logging import setup_logging

        return setup_logging

    if name == "set_seed":
        from .seed import set_seed

        return set_seed

    if name in {"load_dataset", "load_model_wrapper", "merge_eval_cfg"}:
        from .experiment import load_dataset, load_model_wrapper, merge_eval_cfg

        return {
            "load_dataset": load_dataset,
            "load_model_wrapper": load_model_wrapper,
            "merge_eval_cfg": merge_eval_cfg,
        }[name]

    raise AttributeError(f"module 'utils' has no attribute {name!r}")
