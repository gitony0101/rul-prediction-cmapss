# Portfolio Project Notes: RUL Prediction

This document provides a concise guide to the experimental setup and key results for the Remaining Useful Life (RUL) prediction project.

## Experiment Overview

The experiments (G1 through G4) represent an iterative progression in modeling, loss design, and uncertainty awareness:

- **G1 (CNN-BiLSTM MSE)**: Baseline model using a CNN-BiLSTM architecture with standard Mean Squared Error (MSE) loss.
- **G2 (CNN-BiLSTM MSE + MC Dropout)**: Extends the baseline with Monte Carlo Dropout to provide uncertainty-aware predictions and support risk-sensitive interpretation.
- **G3 (CNN-BiLSTM LinEx)**: Replaces MSE with LinEx loss to model asymmetric error costs, especially when overestimation and underestimation have different maintenance implications.
- **G4 (CNN-BiLSTM LinEx + MC Dropout)**: Combines asymmetric loss and uncertainty estimation in a single configuration for decision-relevant RUL modeling.

## Key Results for Public Presentation

For a public-facing presentation or recruiter overview, the most useful summary files are:

- **`outputs_v2/group_comparison.csv`**: Best high-level comparison across G1 to G4.
- **`outputs_v2/G3_multiseed_logs/G3_multiseed_summary.json`**: Useful for showing the effect of asymmetric loss.
- **`outputs_v2/G4_multiseed_logs/G4_multiseed_summary.json`**: Best summary of the combined LinEx + MC Dropout setting.
- **`outputs_v2/G4_multiseed_logs/G4_multiseed_detail.csv`**: Detailed multi-seed results for showing consistency and variance.

## Files to Highlight to Recruiters

If a recruiter or interviewer wants to inspect the technical core of the project, these are the best files to highlight:

- **`src_v2/experiments_v2/G4_run.py`**: Representative experiment script showing how asymmetric loss and MC Dropout are combined in the training workflow.
- **`src_v2/rul/training/loss.py`**: Custom loss implementation, including LinEx.
- **`src_v2/rul/inference/mcd.py`**: Uncertainty-aware inference logic based on Monte Carlo Dropout.
- **`outputs_v2/group_comparison.csv`**: Cleanest entry point for understanding comparative results.

## Suggested Viewing Order

To understand the project efficiently, view the files in this order:

1. **`README.md`**
2. **`outputs_v2/group_comparison.csv`**
3. **`outputs_v2/G1_multiseed_logs/G1_multiseed_summary.json`**
4. **`outputs_v2/G3_multiseed_logs/G3_multiseed_summary.json`**
5. **`outputs_v2/G4_multiseed_logs/G4_multiseed_summary.json`**
6. **`src_v2/experiments_v2/G4_run.py`**
