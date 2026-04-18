from __future__ import annotations

from typing import Any, Callable

from tqdm import tqdm

from eval.metrics import compare_answers, compute_accuracy, compute_compute_stats, compute_latency_stats, normalize_answer
from utils.io import write_json, write_jsonl


class ExperimentRunner:

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir

    def run_split(
        self,
        dataset,
        method_fn: Callable[[dict[str, Any], Any, dict[str, Any]], dict[str,
                                                                        Any]],
        model_wrapper,
        cfg: dict[str, Any],
    ) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        run_cfg = cfg.get("run", {})
        dataset_name = str(cfg.get("dataset", {}).get("name", ""))
        limit = int(run_cfg.get("limit", 0) or 0)
        total = min(len(dataset), limit) if limit > 0 else len(dataset)

        for i in tqdm(range(total), desc="Evaluating"):
            sample = dataset[i]
            rec = method_fn(sample, model_wrapper, cfg)
            rec["normalized_final_answer"] = normalize_answer(
                rec["final_answer"], dataset_name)
            rec["normalized_gold_answer"] = normalize_answer(
                rec["gold_answer"], dataset_name)
            rec["is_correct"] = compare_answers(
                rec["final_answer"],
                rec["gold_answer"],
                dataset_name,
            )
            records.append(rec)

        summary = {
            "accuracy": compute_accuracy(records),
            **compute_latency_stats(records),
            **compute_compute_stats(records),
            "num_samples": len(records),
        }
        return {"summary": summary, "records": records}

    def save(self, result: dict[str, Any], run_name: str) -> None:
        write_jsonl(f"{self.output_dir}/{run_name}_records.jsonl",
                    result["records"])
        write_json(f"{self.output_dir}/{run_name}_summary.json",
                   result["summary"])
