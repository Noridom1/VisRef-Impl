from __future__ import annotations

import argparse
import logging

from visref_baseline.utils.experiment import load_dataset, load_model_wrapper, merge_eval_cfg
from visref_baseline.utils.io import read_yaml
from visref_baseline.utils.logging import setup_logging
from visref_baseline.utils.seed import set_seed


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run VLM on single image-question pair for inference debugging")
    parser.add_argument("--config", required=True, help="Path to default.yaml")
    parser.add_argument("--model_cfg", required=True, help="Path to model config yaml")
    parser.add_argument("--dataset_cfg", required=True, help="Path to dataset config yaml")
    parser.add_argument("--seed", type=int, default=0, help="Seed to sample a single example from the dataset for inference")
    parser.add_argument("--mode", required=True, choices=["st", "tsr", "visref"])
    parser.add_argument("--output_dir", default="outputs")
    
    args = parser.parse_args()
    
    default_cfg = read_yaml(args.config)
    model_cfg = read_yaml(args.model_cfg)
    dataset_cfg = read_yaml(args.dataset_cfg)
    cfg = merge_eval_cfg(default_cfg, model_cfg, dataset_cfg, args.mode, args.output_dir)
    log_path = setup_logging(cfg["logging"]["output_dir"], "run_inference_single", f"{cfg['dataset']['name']}_{cfg['model']['name']}_{args.mode}")

    logger.info("Starting single-example inference")
    logger.info("Config file: %s", args.config)
    logger.info("Model config: %s", args.model_cfg)
    logger.info("Dataset config: %s", args.dataset_cfg)
    logger.info("Mode: %s", args.mode)
    logger.info("Log file: %s", log_path)

    set_seed(int(cfg.get("seed", 42)))
    dataset = load_dataset(cfg["dataset"])
    model_wrapper = load_model_wrapper(cfg["model"], cfg.get("wrapper"))
    logger.info("Loaded dataset=%s samples=%d", cfg["dataset"]["name"], len(dataset))
    logger.info("Loaded model wrapper=%s", type(model_wrapper).__name__)
    
    # Sample a single example from the dataset for inference
    example = dataset[0]  # For simplicity, we just take the first example. You can modify this to sample randomly if needed.
    logger.info("Selected example id=%s", example.get("id"))
    
    # Run inference on the single example
    logger.info("Running inference on the selected example")
    answer = model_wrapper.generate_full_answer(
        question=example[cfg["dataset"]["question_key"]],
        image=example[cfg["dataset"]["image_key"]],
    )
    
    print("Question:", example[cfg["dataset"]["question_key"]])
    print("Answer:", answer)
    
    
    
    