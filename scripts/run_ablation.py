from __future__ import annotations

import argparse
import itertools
import logging
import subprocess
import sys

from utils.io import read_yaml
from utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VisRef ablations")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_cfg", required=True)
    parser.add_argument("--dataset_cfg", required=True)
    parser.add_argument("--output_dir", default="outputs/ablations")
    args = parser.parse_args()

    base = read_yaml(args.config)
    log_path = setup_logging(args.output_dir, "run_ablation")
    logger.info("Starting ablation sweep")
    logger.info("Config file: %s", args.config)
    logger.info("Model config: %s", args.model_cfg)
    logger.info("Dataset config: %s", args.dataset_cfg)
    logger.info("Log file: %s", log_path)
    entropy_values = [0.15, 0.2, 0.25, 0.3]
    budget_values = [0.2, 0.3, 0.4]
    modes = ["visref"]

    for mode, entropy, budget in itertools.product(modes, entropy_values,
                                                   budget_values):
        cmd = [
            sys.executable,
            "scripts/run_eval.py",
            "--config",
            args.config,
            "--model_cfg",
            args.model_cfg,
            "--dataset_cfg",
            args.dataset_cfg,
            "--mode",
            mode,
            "--output_dir",
            args.output_dir,
        ]
        logger.info("Running mode=%s entropy=%s budget=%s", mode, entropy,
                    budget)
        # TODO: propagate these values through a temporary config override.
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
