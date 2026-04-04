# Repository Cleanup Plan

This document outlines a plan to transition this repository into a lightweight, public-facing portfolio version, focusing on clarity for recruiters and removing experimental clutter.

## 1. Keep in public portfolio version
*Essential for demonstrating core technical competence and project results.*

- **`src_v2/`**: The core modeling, training, and evaluation logic (Bi-LSTM, CNN, LinEx, MC Dropout). This is the most important part for recruiters.
- **`README.md`**: The primary entry point and high-level overview.
- **`PORTFOLIO_NOTES.md`**: Excellent guide for interviewers to navigate the codebase.
- **`figures/`**: Visualizations like `group_rmse_comparison.png` that provide immediate impact.
- **`outputs_v2/group_comparison.csv`**: The cleanest summary of experimental results.
- **`outputs_v2/G4_multiseed_logs/`**: High-value summary data for the best-performing configuration.
- **`scripts/`**: Essential utility scripts like `plot_group_comparison.py`.
- **`results/`**: High-level summary documentation.
- **`LICENSE`**: Standard for any public repository.

## 2. Keep but de-emphasize
*Useful for depth, but can be hidden or moved to a separate 'deep-dive' folder/branch to reduce noise.*

- **`CMAPSSData/`**: The raw NASA datasets. While important for reproducibility, they are large and can be downloaded via a script or link in the README instead of bloating the repo.
- **`outputs_v2/G1_...`, `G2_...`, `G3_...`**: Individual seed-level logs and checkpoints. They add weight without adding new high-level insights beyond the summaries.
- **`src_v2/experiments_v2/rounds/`**: The specialized, one-off experiment scripts (e.g., `round8_toy_mechanism_experiment.py`).

## 3. Archive later
*Items to be moved to a long-term storage/archive branch.*

- **`outputs_v2/round8_toy_mechanism/`**: Highly specific, large-scale experimental outputs that are not part of the main project narrative.
- **`src_v2/experiments_v2/docs/`**: Detailed internal experiment documentation and protocol drafts.

## 4. Candidate for removal from the public-facing branch
*Purely experimental or redundant items.*

- **`data/`**: Likely contains redundant or intermediate data processed during experiments.
- **`src/`**: If this contains superseded or legacy versions of the code.
