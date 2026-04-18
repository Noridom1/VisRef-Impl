from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.metrics import compare_answers, normalize_answer
from utils.io import read_yaml


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def aggregate_parallel_records(
    all_chain_records: list[list[dict]],
    dataset_name: str,
) -> dict[str, object]:
    aggregated_records: list[dict] = []
    num_samples = len(all_chain_records[0]) if all_chain_records else 0

    for sample_idx in range(num_samples):
        sample_records = [chain_records[sample_idx] for chain_records in all_chain_records]
        vote_counts: dict[str, tuple[int, int, dict]] = {}
        for chain_idx, record in enumerate(sample_records):
            normalized = normalize_answer(record["final_answer"], dataset_name)
            count, first_idx, first_record = vote_counts.get(
                normalized,
                (0, chain_idx, record),
            )
            vote_counts[normalized] = (count + 1, first_idx, first_record)

        winning_answer, (winning_votes, _, winning_record) = max(
            vote_counts.items(),
            key=lambda item: (item[1][0], -item[1][1]),
        )
        aggregated_records.append({
            "sample_id": winning_record["sample_id"],
            "mode": "visref_parallel_vote",
            "final_answer": winning_record["final_answer"],
            "raw_final_answer": winning_record.get("raw_final_answer", winning_record["final_answer"]),
            "gold_answer": winning_record["gold_answer"],
            "normalized_final_answer": winning_answer,
            "normalized_gold_answer": normalize_answer(
                winning_record["gold_answer"], dataset_name),
            "steps_used": winning_record["steps_used"],
            "entropy_trace": winning_record.get("entropy_trace", []),
            "selected_token_counts": winning_record.get("selected_token_counts", []),
            "latency_sec": winning_record.get("latency_sec", 0.0),
            "reasoning_steps": winning_record.get("reasoning_steps", []),
            "vote_count": winning_votes,
            "chain_predictions": [record["final_answer"] for record in sample_records],
            "is_correct": compare_answers(
                winning_record["final_answer"],
                winning_record["gold_answer"],
                dataset_name,
            ),
        })

    accuracies = [record["is_correct"] for record in aggregated_records]
    avg_steps = (
        sum(float(record["steps_used"]) for record in aggregated_records) / len(aggregated_records)
        if aggregated_records else 0.0
    )
    avg_latency = (
        sum(float(record["latency_sec"]) for record in aggregated_records) / len(aggregated_records)
        if aggregated_records else 0.0
    )
    avg_selected_tokens = 0.0
    if aggregated_records:
        avg_selected_tokens = sum(
            (
                sum(record["selected_token_counts"]) / len(record["selected_token_counts"])
                if record["selected_token_counts"] else 0.0
            )
            for record in aggregated_records
        ) / len(aggregated_records)

    return {
        "records": aggregated_records,
        "summary": {
            "accuracy": (
                sum(1.0 for is_correct in accuracies if is_correct) / len(accuracies)
                if accuracies else 0.0
            ),
            "avg_latency_sec": avg_latency,
            "avg_steps": avg_steps,
            "avg_selected_tokens": avg_selected_tokens,
            "num_samples": len(aggregated_records),
            "num_chains": len(all_chain_records),
        },
    }


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    all_chain_records: list[list[dict]] = []
    dataset_cfg = read_yaml(args.dataset_cfg)
    dataset_name = str(dataset_cfg.get("dataset", {}).get("name", "mathvista"))

    for i in range(args.chains):
        chain_output_dir = output_root / f"chain_{i}"
        run_name = f"parallel_budget_visref_chain_{i}"
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
            str(chain_output_dir),
            "--run_name",
            run_name,
            "--seed",
            str(args.seed + i),
            "--temperature",
            str(args.temperature),
            "--top_k",
            str(args.top_k),
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        subprocess.run(cmd, check=True)
        records = read_jsonl(chain_output_dir / f"{run_name}_records.jsonl")
        all_chain_records.append(records)
        if not dataset_name and records:
            dataset_name = "mathvista"

    aggregated = aggregate_parallel_records(all_chain_records, dataset_name)
    write_jsonl(output_root / "parallel_budget_aggregate_records.jsonl", aggregated["records"])
    write_json(output_root / "parallel_budget_aggregate_summary.json", aggregated["summary"])
    print(aggregated["summary"])


if __name__ == "__main__":
    main()
