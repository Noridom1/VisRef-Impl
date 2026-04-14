# VisRef Baseline Scaffold

This repository contains a starter implementation scaffold for reproducing:

VisRef: Visual Refocusing while Thinking Improves Test-Time Scaling in Multi-Modal Large Reasoning Models.

## What is implemented

- Dataset interfaces for MathVista, MathVision, MM-Star
- Unified model adapter API
- ST, TSR, and VisRef method runners
- DPP greedy selector implementation
- Entropy-based stopping
- Experiment runner, metrics, and JSON/JSONL outputs
- CLI scripts for eval, ablations, and parallel-budget experiments

## What you need to fill in

- Real InternVL adapter model loading and generation in `models/internvl.py`
- Dataset files under `dataset/...`
- Optional: answer normalization specific to each benchmark
- Optional: exact entropy extraction from model logits

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare dataset files (for example MathVista):

- `dataset/mathvista/testmini.json` or `dataset/mathvista/testmini.jsonl`

Each item must include at least:

- `image`
- `question`
- `answer`

3. Run baseline modes:

```bash
python scripts/run_eval.py --config configs/default.yaml --model_cfg configs/model_internvl8b.yaml --dataset_cfg configs/dataset_mathvista.yaml --mode st
python scripts/run_eval.py --config configs/default.yaml --model_cfg configs/model_internvl8b.yaml --dataset_cfg configs/dataset_mathvista.yaml --mode tsr
python scripts/run_eval.py --config configs/default.yaml --model_cfg configs/model_internvl8b.yaml --dataset_cfg configs/dataset_mathvista.yaml --mode visref
```

Outputs are written to `outputs/`.
