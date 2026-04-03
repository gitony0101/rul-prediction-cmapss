# RUL Prediction: Uncertainty-Aware Maintenance Decision Support

Deep learning-based Remaining Useful Life (RUL) prediction integrating asymmetric loss and uncertainty estimation for maintenance-relevant risk analysis.

## Overview

This project studies Remaining Useful Life (RUL) prediction using Bi-LSTM and CNN-BiLSTM models. It examines how asymmetric loss design and predictive uncertainty can support maintenance-oriented decision analysis by reducing harmful overestimation and improving risk awareness.

## Datasets

The project includes experiments on:

- **NASA C-MAPSS**: benchmark turbofan engine degradation datasets
- **Synthetic Degradation Data**: controlled degradation patterns for mechanism-oriented validation and analysis

## Methods

- **Architectures**: Bi-LSTM, CNN-BiLSTM
- **Loss Functions**: MSE, LinEx (asymmetric loss)
- **Uncertainty Estimation**: Monte Carlo Dropout

## Evaluation Metrics

- RMSE
- MAE
- NASA Score
- Overestimation Rate

## Representative Results (5 seeds)

| Group | Setting | RMSE (mean ± std) | MAE (mean ± std) | NASA Score (mean) | Best RMSE |
|------|---------|-------------------|------------------|-------------------|-----------|
| G1 | Baseline (MSE) | 15.93 ± 0.85 | 11.62 ± 0.64 | 509.16 | 14.86 |
| G2 | MSE + MC Dropout | 15.45 ± 0.97 | 11.21 ± 0.71 | 526.91 | 14.57 |
| G3 | LinEx | 15.02 ± 0.82 | 10.62 ± 0.75 | 518.31 | 13.75 |
| G4 | LinEx + MC Dropout | 14.78 ± 0.38 | 10.89 ± 0.28 | 376.89 | 14.17 |

![Group RMSE Comparison](figures/group_rmse_comparison.png)

## Key Results

Across 5-seed experiments, the combined LinEx + MC Dropout setting (G4) achieved the best overall mean RMSE (14.78) and the lowest mean NASA Score (376.89). The LinEx-only setting (G3) achieved the best single-run RMSE (13.75). These results suggest that asymmetric loss and uncertainty estimation can improve decision-relevant RUL modeling beyond a standard MSE baseline.

## Repository Structure

- `src_v2/`: core modeling, training, evaluation, and experiment logic
- `CMAPSSData/`: NASA C-MAPSS benchmark datasets
- `outputs_v2/`: experimental results, logs, and summaries

## How to Run

Run representative experiments from the current repository structure:

```bash
python src_v2/experiments_v2/G1_run.py
python src_v2/experiments_v2/G2_run.py
python src_v2/experiments_v2/G3_run.py
```