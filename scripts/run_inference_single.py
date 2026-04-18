from __future__ import annotations

import os

import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.logging import setup_logging

from utils.io import read_yaml

from utils.experiment import (
    load_dataset,
    load_model_wrapper,
    merge_eval_cfg,
)

import logging

import argparse


logger = logging.getLogger(__name__)


def main():

    parser = argparse.ArgumentParser(
        description="Run VLM on single image-question pair for inference debugging"
    )

    parser.add_argument(
        "--config",
        default="./configs/default.yaml",
        help="Path to default.yaml",
    )

    parser.add_argument(
        "--model_cfg",
        default="./configs/qwen3-8b-instruct.yaml",
        help="Path to model config yaml",
    )

    parser.add_argument(
        "--dataset_cfg",
        default="./configs/dataset_mathvision.yaml",
        help="Path to dataset config yaml",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Maximum number of new tokens to generate (overrides model_cfg)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed to sample a single example from the dataset for inference",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the example to run inference on (default: 0)",
    )

    parser.add_argument("--mode", default="st", choices=["st", "tsr", "visref"])

    parser.add_argument("--output_dir", default="outputs")

    args = parser.parse_args()

    default_cfg = read_yaml(args.config)

    model_cfg = read_yaml(args.model_cfg)

    dataset_cfg = read_yaml(args.dataset_cfg)

    cfg = merge_eval_cfg(
        default_cfg, model_cfg, dataset_cfg, args.mode, args.output_dir
    )

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

    from utils.seed import set_seed

    set_seed(args.seed)

    dataset = load_dataset(cfg["dataset"])

    model_wrapper = load_model_wrapper(cfg["model"], cfg.get("wrapper"))

    logger.info("Loaded dataset=%s samples=%d", cfg["dataset"]["name"], len(dataset))

    logger.info("Loaded model wrapper=%s", type(model_wrapper).__name__)

    # Sample a single example from the dataset for inference

    example = dataset[args.index]  # Take the example at the specified index

    print("Example keys:", list(example.keys()))
    logger.info("Selected example id=%s", example.get("id"))

    print("Selected example question:", example["question"])

    print("Selected example image:", example["image"])

    print(
        "Selected example choices:",
        example.get("choices", "N/A"),
    )

    # Run inference on the single example

    logger.info("Running inference on the selected example")

    # answer = model_wrapper.generate_per_token(
    #     question=example[cfg["dataset"]["question_key"]],
    #     image=example[cfg["dataset"]["image_key"]],
    #     choices=example["choices"] if "choices" in example else None,
    #     max_new_tokens=args.max_new_tokens if args.max_new_tokens is not None else 1024,
    #     temperature=0.7,
    #     top_k=50,
    #     # force_wait_before_max=False,
    #     # wait_token_text="Wait",
    #     # wait_token_candidates=[" Wait", "Wait"],
    # )

    answer = model_wrapper.generate_full_answer(
        question=example["question"],
        image=example["image"],
        choices=example["choices"] if "choices" in example else None,
        max_new_tokens=args.max_new_tokens if args.max_new_tokens is not None else 1024,
        temperature=0.0,
        # top_k=50,
    )

    print("Question:", example["question"])

    print("Answer:", answer)


if __name__ == "__main__":

    main()
