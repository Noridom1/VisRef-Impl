from __future__ import annotations

import argparse
import logging

from engine.runner import ExperimentRunner
from methods import run_st, run_tsr, run_visref
from utils.experiment import load_dataset, load_model_wrapper, merge_eval_cfg
from utils.io import read_yaml
from utils.logging import setup_logging
from utils.seed import set_seed

logger = logging.getLogger(__name__)


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
    args = parser.parse_args()

    default_cfg = read_yaml(args.config)
    model_cfg = read_yaml(args.model_cfg)
    dataset_cfg = read_yaml(args.dataset_cfg)
    cfg = merge_eval_cfg(default_cfg, model_cfg, dataset_cfg, args.mode,
                         args.output_dir)
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
    run_name = f"{cfg['dataset']['name']}_{cfg['model']['name']}_{args.mode}"
    runner.save(result, run_name=run_name)
    logger.info("Completed evaluation: %s", result["summary"])
    print(result["summary"])


if __name__ == "__main__":
    main()
