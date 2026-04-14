from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fixed-budget parallel scaling experiments")
    parser.add_argument("--chains",
                        type=int,
                        default=4,
                        help="Number of parallel chains")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_cfg", required=True)
    parser.add_argument("--dataset_cfg", required=True)
    parser.add_argument("--output_dir", default="outputs/parallel_budget")
    args = parser.parse_args()

    # Placeholder orchestration: run repeated VisRef chains independently.
    for i in range(args.chains):
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
            "visref",
            "--output_dir",
            f"{args.output_dir}/chain_{i}",
        ]
        subprocess.run(cmd, check=True)

    print("Completed placeholder parallel runs. Add vote aggregation next.")


if __name__ == "__main__":
    main()
