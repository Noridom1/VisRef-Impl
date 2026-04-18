from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.runner import ExperimentRunner
from methods import run_st, run_tsr, run_visref
from utils.experiment import load_dataset, load_model_wrapper, merge_eval_cfg
from utils.io import read_yaml
from utils.logging import setup_logging

logger = logging.getLogger(__name__)


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.limit is not None:
        cfg.setdefault("run", {})["limit"] = args.limit
    if args.seed is not None:
        cfg["seed"] = args.seed

    visref_cfg = cfg.setdefault("visref", {})
    if args.entropy_threshold is not None:
        visref_cfg["entropy_threshold"] = args.entropy_threshold
    if args.token_budget_ratio is not None:
        visref_cfg["token_budget_ratio"] = args.token_budget_ratio
    if args.max_steps is not None:
        visref_cfg["max_steps"] = args.max_steps

    generation_cfg = cfg.setdefault("generation", {})
    if args.temperature is not None:
        generation_cfg["temperature"] = args.temperature
    if args.top_k is not None:
        generation_cfg["top_k"] = args.top_k
    if args.max_new_tokens is not None:
        generation_cfg["max_new_tokens"] = args.max_new_tokens
    if args.reasoning_step_tokens is not None:
        generation_cfg["reasoning_step_tokens"] = args.reasoning_step_tokens
    if args.answer_max_new_tokens is not None:
        generation_cfg["answer_max_new_tokens"] = args.answer_max_new_tokens
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ST/TSR/VisRef evaluation")
    parser.add_argument("--config", required=True, help="Path to default.yaml")
    parser.add_argument("--model_cfg",
                        required=True,
                        help="Path to model config yaml")
    parser.add_argument("--dataset_cfg",
                        required=True,
                        help="Path to dataset config yaml")
    parser.add_argument("--mode",
                        required=True,
                        choices=["st", "tsr", "visref"])
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--entropy_threshold", type=float, default=None)
    parser.add_argument("--token_budget_ratio", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--reasoning_step_tokens", type=int, default=None)
    parser.add_argument("--answer_max_new_tokens", type=int, default=None)
    args = parser.parse_args()

    default_cfg = read_yaml(args.config)
    model_cfg = read_yaml(args.model_cfg)
    dataset_cfg = read_yaml(args.dataset_cfg)
    cfg = merge_eval_cfg(default_cfg, model_cfg, dataset_cfg, args.mode,
                         args.output_dir)
    cfg = apply_overrides(cfg, args)
    log_path = setup_logging(
        cfg["logging"]["output_dir"],
        "run_eval",
        f"{cfg['dataset']['name']}_{cfg['model']['name']}_{args.mode}",
    )

    logger.info("Starting evaluation")
    logger.info("Config file: %s", args.config)
    logger.info("Model config: %s", args.model_cfg)
    logger.info("Dataset config: %s", args.dataset_cfg)
    logger.info("Mode: %s", args.mode)
    logger.info("Log file: %s", log_path)
    logger.info("Run config: %s", cfg["run"])
    logger.info("Generation config: %s", cfg.get("generation", {}))
    logger.info("VisRef config: %s", cfg.get("visref", {}))

    from utils.seed import set_seed

    set_seed(int(cfg.get("seed", 42)))
    dataset = load_dataset(cfg["dataset"])
    model_wrapper = load_model_wrapper(cfg["model"], cfg.get("wrapper"))
    logger.info("Loaded dataset=%s samples=%d", cfg["dataset"]["name"],
                len(dataset))
    logger.info("Loaded model wrapper=%s", type(model_wrapper).__name__)

    if args.mode == "st":
        method_fn = run_st
    elif args.mode == "tsr":
        method_fn = run_tsr
    else:
        method_fn = run_visref

    runner = ExperimentRunner(cfg["logging"]["output_dir"])
    result = runner.run_split(dataset, method_fn, model_wrapper, cfg)
    run_name = args.run_name or f"{cfg['dataset']['name']}_{cfg['model']['name']}_{args.mode}"
    runner.save(result, run_name=run_name)
    logger.info("Completed evaluation: %s", result["summary"])
    print(result["summary"])


if __name__ == "__main__":
    main()
