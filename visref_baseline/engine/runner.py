from __future__ import annotations

from typing import Any, Callable

from tqdm import tqdm

from visref_baseline.eval.metrics import compute_accuracy, compute_compute_stats, compute_latency_stats
from visref_baseline.utils.io import write_json, write_jsonl


class ExperimentRunner:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir

    def run_split(
        self,
        dataset,
        method_fn: Callable[[dict[str, Any], Any, dict[str, Any]], dict[str, Any]],
        model_wrapper,
        cfg: dict[str, Any],
    ) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        for i in tqdm(range(len(dataset)), desc="Evaluating"):
            sample = dataset[i]
            rec = method_fn(sample, model_wrapper, cfg)
            rec["is_correct"] = rec["final_answer"].strip().lower() == rec["gold_answer"].strip().lower()
            records.append(rec)

        summary = {
            "accuracy": compute_accuracy(records),
            **compute_latency_stats(records),
            **compute_compute_stats(records),
            "num_samples": len(records),
        }
        return {"summary": summary, "records": records}

    def save(self, result: dict[str, Any], run_name: str) -> None:
        write_jsonl(f"{self.output_dir}/{run_name}_records.jsonl", result["records"])
        write_json(f"{self.output_dir}/{run_name}_summary.json", result["summary"])
