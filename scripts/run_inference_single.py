from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from methods import run_st, run_tsr, run_visref
from utils.experiment import load_dataset, load_model_wrapper, merge_eval_cfg
from utils.io import read_yaml, write_json
from utils.logging import setup_logging

logger = logging.getLogger(__name__)


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    visref_cfg = cfg.setdefault("visref", {})
    if args.entropy_threshold is not None:
        visref_cfg["entropy_threshold"] = args.entropy_threshold
    if args.max_steps is not None:
        visref_cfg["max_steps"] = args.max_steps

    generation_cfg = cfg.setdefault("generation", {})
    if args.max_new_tokens is not None:
        generation_cfg["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None:
        generation_cfg["temperature"] = args.temperature
    if args.top_k is not None:
        generation_cfg["top_k"] = args.top_k
    if args.reasoning_step_tokens is not None:
        generation_cfg["reasoning_step_tokens"] = args.reasoning_step_tokens
    if args.answer_max_new_tokens is not None:
        generation_cfg["answer_max_new_tokens"] = args.answer_max_new_tokens
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one example with ST/TSR/VisRef and print reasoning steps"
    )
    parser.add_argument("--config", default="./configs/default.yaml", help="Path to default.yaml")
    parser.add_argument("--model_cfg", default="./configs/qwen3-8b-instruct.yaml", help="Path to model config yaml")
    parser.add_argument("--dataset_cfg", default="./configs/dataset_mathvision.yaml", help="Path to dataset config yaml")
    parser.add_argument("--mode", default="st", choices=["st", "tsr", "visref"])
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--seed", type=int, default=0, help="Seed for deterministic single-sample run")
    parser.add_argument("--index", type=int, default=0, help="Index of the dataset sample")

    parser.add_argument("--max_new_tokens", type=int, default=None, help="Override generation.max_new_tokens")
    parser.add_argument("--temperature", type=float, default=None, help="Override generation.temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Override generation.top_k")
    parser.add_argument("--reasoning_step_tokens", type=int, default=None, help="Override generation.reasoning_step_tokens")
    parser.add_argument("--answer_max_new_tokens", type=int, default=None, help="Override generation.answer_max_new_tokens")
    parser.add_argument("--entropy_threshold", type=float, default=None, help="Override visref.entropy_threshold")
    parser.add_argument("--max_steps", type=int, default=None, help="Override visref.max_steps")
    parser.add_argument(
        "--save_record",
        action="store_true",
        help="Save the single-run record as JSON in output_dir",
    )
    args = parser.parse_args()

    default_cfg = read_yaml(args.config)
    model_cfg = read_yaml(args.model_cfg)
    dataset_cfg = read_yaml(args.dataset_cfg)
    cfg = merge_eval_cfg(default_cfg, model_cfg, dataset_cfg, args.mode, args.output_dir)
    cfg = apply_overrides(cfg, args)

    log_path = setup_logging(
        cfg["logging"]["output_dir"],
        "run_inference_single",
        f"{cfg['dataset']['name']}_{cfg['model']['name']}_{args.mode}",
    )
    logger.info("Starting single-example inference")
    logger.info("Config file: %s", args.config)
    logger.info("Model config: %s", args.model_cfg)
    logger.info("Dataset config: %s", args.dataset_cfg)
    logger.info("Mode: %s", args.mode)
    logger.info("Log file: %s", log_path)
    logger.info("Generation config: %s", cfg.get("generation", {}))
    logger.info("VisRef config: %s", cfg.get("visref", {}))

    from utils.seed import set_seed

    set_seed(args.seed)
    dataset = load_dataset(cfg["dataset"])
    model_wrapper = load_model_wrapper(cfg["model"], cfg.get("wrapper"))
    logger.info("Loaded dataset=%s samples=%d", cfg["dataset"]["name"], len(dataset))
    logger.info("Loaded model wrapper=%s", type(model_wrapper).__name__)

    example = dataset[args.index]
    logger.info("Selected example id=%s", example.get("id"))
    print("Question:", example["question"])
    print("Image:", example["image"])
    print("Choices:", example.get("choices", "N/A"))

    if args.mode == "st":
        method_fn = run_st
    elif args.mode == "tsr":
        method_fn = run_tsr
    else:
        method_fn = run_visref

    record = method_fn(example, model_wrapper, cfg)

    print("Final answer:", record["final_answer"])
    print("Raw final answer:", record.get("raw_final_answer", record["final_answer"]))
    print("Gold answer:", record["gold_answer"])
    print("Steps used:", record.get("steps_used", 0))
    print("Entropy trace:", record.get("entropy_trace", []))
    print("Latency (sec):", round(float(record.get("latency_sec", 0.0)), 4))

    reasoning_steps = list(record.get("reasoning_steps", []))
    print("Reasoning steps:")
    if reasoning_steps:
        for idx, step in enumerate(reasoning_steps, start=1):
            print(f"[{idx}] {step}")
    else:
        print("(no reasoning steps returned)")

    if args.save_record:
        run_name = f"{cfg['dataset']['name']}_{cfg['model']['name']}_{args.mode}"
        sample_id = str(record.get("sample_id", "sample")).replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(
            cfg["logging"]["output_dir"],
            f"{run_name}_{sample_id}_{timestamp}.json",
        )
        write_json(save_path, record)
        print("Saved record:", save_path)


if __name__ == "__main__":
    main()
