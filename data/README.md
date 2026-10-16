# Data

Only the datasets used by the tutorial are kept here.

| Stage | Files | Rows |
|---|---|---:|
| SFT | `sft/train.jsonl`, `sft/validation.jsonl` | 800 / 200 |
| GRPO | `grpo/train.parquet`, `grpo/validation.parquet` | 1000 / 50 |
| Evaluation | `evaluation/tasks.jsonl` (Final-200 Clean) | 200 |

Adjacent `metadata.json` files record SHA256 checksums and collection
provenance. All SFT, GRPO and evaluation splits are task-disjoint. Generated
trajectories belong under `outputs/`, never under `data/`. First use
`scripts/collect_sft_data.py` to save raw Teacher trajectories, then run
`scripts/clean_sft_data.py` offline to create an audited SFT dataset before
promoting its train/validation files into this directory.
