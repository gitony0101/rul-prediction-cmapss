# Prune Candidates for Public-Facing Branch

This document lists directories and files that are strong candidates for removal from the public-facing version of the repository to reduce clutter and repository size.

## Candidates for Removal

- **`data/`**
    - **Why safe to remove**: Likely contains intermediate or redundant data files that are not part of the core project narrative or essential for understanding the models.
    - **Coverage**: Not covered by `public_results/`, but the core datasets are provided in `CMAPSSData/` (which is being de-emphasized but kept).

- **`src/`**
    - **Why safe to remove**: Appears to contain legacy or superseded code versions that are not used in the current `src_v2/` implementation.
    - **Coverage**: Fully superseded by `src_v2/`.

- **`outputs_v2/` (Sub-directories containing individual seed logs)**
    - **Why safe to remove**: The high-level results are already captured in `public_results/`. The granular seed-level outputs (e.g., `outputs_v2/G1_CNN_BiLSTM_MSE/seed_123`) add significant weight without adding new architectural insights.
    - **Coverage**: High-level summaries and group comparisons are preserved in `public_results/`.

- **`src_v2/experiments_v2/docs/`**
    - **Why safe to ***remove*** from public branch**: These are internal experiment protocols and technical drafts that are useful for development but add noise to a recruiter-facing overview.
    - **Coverage**: The `README.md` and `PORTFOLIO_NOTES.md` provide the necessary high-level technical context.

- **`src_v2/experiments_v2/rounds/` (Specific experiment scripts)**
    - **Why safe to remove**: Contains specialized, one-off experiment scripts (e.g., `round8_toy_mechanism_experiment.py`) that are not part of the core, repeatable experiment pipeline (G1-G4).
    - **Coverage**: The core experiment logic is captured in `src_v2/experiments_v2/run_group_multiseed.py` and the representative `G*_run.py` scripts.

## Note on Retention
No files or directories from the core `src_v2/` architecture, `CMAPSSData/`, or the `public_results/` directory are recommended for removal, as they are essential for project understanding and reproducibility.
