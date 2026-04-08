from __future__ import annotations

import argparse

from visref_baseline.data import MMStarDataset, MathVisionDataset, MathVistaDataset
from visref_baseline.engine.runner import ExperimentRunner
from visref_baseline.methods import run_st, run_tsr, run_visref
from visref_baseline.models import InternVLAdapter
from visref_baseline.utils.io import read_yaml
from visref_baseline.utils.seed import set_seed


def _load_dataset(dataset_cfg: dict):
    name = dataset_cfg["name"].lower()
    common = {
        "root": dataset_cfg["root"],
        "split": dataset_cfg["split"],
        "image_key": dataset_cfg["image_key"],
        "question_key": dataset_cfg["question_key"],
        "answer_key": dataset_cfg["answer_key"],
    }
    if name == "mathvista":
        return MathVistaDataset(**common)
    if name == "mathvision":
        return MathVisionDataset(**common)
    if name == "mmstar":
        return MMStarDataset(**common)
    raise ValueError(f"Unsupported dataset: {name}")


def _merge_cfg(default_cfg: dict, model_cfg: dict, dataset_cfg: dict, mode: str, output_dir: str) -> dict:
    merged = dict(default_cfg)
    merged["model"] = model_cfg["model"]
    merged["prompt"] = model_cfg["prompt"]
    merged["dataset"] = dataset_cfg["dataset"]
    merged["run"] = {"mode": mode}
    merged.setdefault("logging", {})["output_dir"] = output_dir
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ST/TSR/VisRef evaluation")
    parser.add_argument("--config", required=True, help="Path to default.yaml")
    parser.add_argument("--model_cfg", required=True, help="Path to model config yaml")
    parser.add_argument("--dataset_cfg", required=True, help="Path to dataset config yaml")
    parser.add_argument("--mode", required=True, choices=["st", "tsr", "visref"])
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    default_cfg = read_yaml(args.config)
    model_cfg = read_yaml(args.model_cfg)
    dataset_cfg = read_yaml(args.dataset_cfg)
    cfg = _merge_cfg(default_cfg, model_cfg, dataset_cfg, args.mode, args.output_dir)

    set_seed(int(cfg.get("seed", 42)))
    dataset = _load_dataset(cfg["dataset"])
    adapter = InternVLAdapter(cfg["model"])

    if args.mode == "st":
        method_fn = run_st
    elif args.mode == "tsr":
        method_fn = run_tsr
    else:
        method_fn = run_visref

    runner = ExperimentRunner(cfg["logging"]["output_dir"])
    result = runner.run_split(dataset, method_fn, adapter, cfg)
    run_name = f"{cfg['dataset']['name']}_{cfg['model']['name']}_{args.mode}"
    runner.save(result, run_name=run_name)
    print(result["summary"])


if __name__ == "__main__":
    main()
