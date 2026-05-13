<p align="center">
  <h1 align="center">FrontierSmith</h1>
</p>

<h3 align="center">
Synthetic Open-ended Problem Generation
</h3>

<p align="center">
  <a href="https://github.com/FrontierCS/Frontier-CS"><img src="https://img.shields.io/badge/Frontier--CS-Official_Repo-blue?logo=github" alt="Frontier-CS"></a>
  <img src="https://img.shields.io/badge/Synthetic_Problems-10-green" alt="Synthetic Problems">
  <img src="https://img.shields.io/badge/Python-3.11+-yellow?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Docker-24+-2496ED?logo=docker&logoColor=white" alt="Docker">
</p>

---

## Overview

**FrontierSmith** is the synthetic open-ended problem-generation pipeline. This repository contains training code, evaluation code, and **10 synthetic algorithmic problems** used in the paper's parity experiment.

> The orchestrator and LLM-driven test/checker generators are intentionally withheld.

---

## Repository Structure

```
FrontierSmith/
├── README.md
├── requirements.txt
├── setup-env.sh                          # one-shot environment bootstrap
├── verl/                                 # vendored VERL framework (editable install)
├── ALE-Bench/                            # ALE-Bench validator (third-party)
├── Frontier-CS/
│   ├── algorithmic/
│   │   ├── problems/                     # 10 synthetic problems
│   │   │   └── frontiersmith_{1..10}/
│   │   ├── Dockerfile / server.js / judge/ / scripts/
│   │   └── ...
│   ├── src/ pyproject.toml
│   └── README.md
├── harbor/
│   └── adapters/frontier-cs-algorithm/   # Harbor adapter
├── scripts/                              # training / evaluation / data-prep
└── data/
    └── sample_lists/                     # reproducibility manifests
```

---

## Synthetic Problems

10 problems in `Frontier-CS/algorithmic/problems/`. These correspond to problems **306–315** in the [Frontier-CS main repository](https://github.com/FrontierCS/Frontier-CS):

| ID | Frontier-CS ID | Name |
|:---|:---------------|:-----|
| `frontiersmith_1` | 306 | Scorched Bridges Campaign |
| `frontiersmith_2` | 307 | Farmwide Teleport Pad Deployment |
| `frontiersmith_3` | 308 | Metallic Pink Resonator Layout |
| `frontiersmith_4` | 309 | Park Ranger Shift Balancing |
| `frontiersmith_5` | 310 | Prime Resonance Retuning |
| `frontiersmith_6` | 311 | Mobile Relay Layout |
| `frontiersmith_7` | 312 | Archipelago Relay Network Design |
| `frontiersmith_8` | 313 | Resonant Bay Layout |
| `frontiersmith_9` | 314 | Duff's Defensive Lineup |
| `frontiersmith_10` | 315 | Quadratic Witness Packing |

Each directory contains:

```
chk.cc           # custom checker (testlib, prints "Ratio: <float>")
config.yaml      # judge configuration
gen.cpp          # testlib-style test-case generator
statement.txt    # problem statement
testdata/        # *.in / *.ans pairs
```

---

## Environment Setup

```bash
source setup-env.sh             # creates .venv, installs all deps
source setup-env.sh --skip      # activate existing env quickly
```

External services:

```bash
hf auth login          # to download Qwen3.5-9B / 27B weights
wandb login            # optional, for training logs
```

### Tested Versions

| Package      | Version           | Notes                                   |
|:-------------|:------------------|:----------------------------------------|
| Python       | 3.11              | `apt install python3.11 python3.11-dev` |
| torch        | 2.11.0+cu130      | pulled by vllm                          |
| vllm         | 0.20.0            |                                         |
| transformers | 5.7.0             | Qwen3.5 needs >= 5.2.0                  |
| verl         | 0.8.0.dev (local) | editable install from `verl/`           |
| ray          | 2.55.1            |                                         |

---

## Datasets

### Frontier-CS Algorithmic Track (172 problems, public)

Not redistributed. Use the [official release](https://github.com/FrontierCS/Frontier-CS) to populate `Frontier-CS/algorithmic/problems/<numeric_id>/`.

### HardTest (sampled, public)

```bash
python scripts/download_hardtest.py
python scripts/install_hardtest_frontier_packages.py
python scripts/split_hardtest_by_difficulty.py
python scripts/sample_hardtest_problems.py --n 200 --seed 42 \
       -o results/hardtest_hard_sampled_200.json
```

The exact 200-problem manifest is at `data/sample_lists/hardtest_hard_sampled_200.json`.

### Synthetic Problems (10, this repo)

The 30-problem mixed sample list (10 from each of HardTest, Frontier-CS, synthetic) is at `data/sample_lists/harbor_sample_30.jsonl`.

### ALE-Bench (validation only)

```bash
python scripts/prepare_alebench_parquet.py
```

---

## Services

### Frontier-CS Judge

```bash
cd Frontier-CS/algorithmic
docker build -t frontiercs-judge .
./run_judge.sh                  # listens on http://localhost:8082
```

### ALE-Bench Judge (optional)

```bash
cd ALE-Bench
bash scripts/docker_build_202301.sh $(id -u) $(id -g)
```

---

## Data Preparation

```bash
python scripts/prepare_frontiercs_parquet.py             # Frontier-CS 172 → parquet
python scripts/prepare_hardtest_hard_sample_parquet.py    # HardTest 200 → parquet
python scripts/prepare_synthetic_parquet.py               # 10 synthetic → parquet
python scripts/prepare_alebench_parquet.py                # ALE-Bench validation
python scripts/prepare_random_reward_train_parquet.py     # Random-reward
```

---

## Training

All scripts use `python -m verl.trainer.main_ppo` with Hydra overrides.

```bash
# Main 9B GRPO run
bash scripts/run_verl_grpo_frontiercs_qwen35_9b.sh

# Multi-GPU
NGPU=8 TP=2 bash scripts/run_verl_grpo_frontiercs_qwen35_9b.sh
```

| Variable      | Default                         | Description                          |
|:--------------|:--------------------------------|:-------------------------------------|
| `MODEL_PATH`  | `Qwen/Qwen3.5-9B`              | HF model id or local path           |
| `TRAIN_DATA`  | `data/frontiercs/train.parquet` | training parquet                     |
| `VAL_DATA`    | `data/frontiercs/full.parquet`  | validation parquet                   |
| `NGPU`        | `4`                             | GPUs per node                        |
| `TP`          | `1`                             | tensor parallel size for vLLM        |
| `FRESH_START` | `0`                             | set `1` to start from scratch        |

### Experiment Scripts

| Script | Purpose |
|:-------|:--------|
| `run_verl_grpo_frontiercs_qwen35_9b.sh` | 9B on Frontier-CS (172 problems) |
| `run_verl_grpo_frontiercs_qwen35_9b_no_validation.sh` | above, validation disabled |
| `run_verl_grpo_frontiercs_qwen35_9b_alebench.sh` | 9B with ALE-Bench validation |
| `run_verl_grpo_frontiercs_qwen35_9b_hardtest.sh` | 9B on HardTest 200 |
| `run_verl_grpo_frontiercs_qwen35_9b_synthetic.sh` | 9B + synthetic mix |
| `run_verl_grpo_frontiercs_qwen35_9b_nofilter.sh` | ablation (no filtering) |
| `run_verl_grpo_frontiercs_qwen35_9b_randomreward.sh` | random-reward control |

---

## Evaluation

```bash
# Start vLLM server
bash scripts/start_vllm_server.sh

# Base model / checkpoint sweeps
bash scripts/eval_base_model_frontiercs.sh
bash scripts/eval_frontiercs_checkpoints.sh

# Single-shot evaluation
python scripts/eval_frontiercs_via_vllm.py
python scripts/run_qwen_frontiercs.py
python scripts/run_merged_model.py

# VERL inference
bash scripts/run_verl_inference_server.sh
bash scripts/run_verl_inference_from_ckpt.sh
bash scripts/run_verl_inference_from_model.sh

# Convert FSDP shards → HF model
python scripts/merge_fsdp_to_hf.py --ckpt-dir <...> --output-dir <...>
```

### Plotting & Diagnostics

```bash
python scripts/plot_frontiercs_validation.py
python scripts/plot_loss_reward_frontiercs.py
python scripts/plot_training_loss.py
python scripts/parse_frontiercs_val_metrics.py
python scripts/compute_numeric_problem_scores.py
```
