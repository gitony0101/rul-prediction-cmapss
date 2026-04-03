# Project Results Summary: RUL Prediction

This project studies how asymmetric loss design and uncertainty estimation affect Remaining Useful Life (RUL) prediction performance and maintenance-relevant evaluation.

## Experimental Groups

- **G1 (Baseline)**: Standard CNN-BiLSTM architecture using MSE loss for baseline performance benchmarking.
- **G2 (Uncertainty-Aware)**: Extends the baseline with Monte Carlo Dropout to provide uncertainty-aware predictions.
- **G3 (Asymmetric Loss)**: Replaces MSE with LinEx loss to model asymmetric error costs.
- **G4 (LinEx + MC Dropout)**: Combines asymmetric loss and uncertainty estimation in a single setting.

## Performance Comparison

| Group | RMSE (Mean) | MAE (Mean) | NASA Score (Mean) | Best Single-Run RMSE |
| :--- | :--- | :--- | :--- | :--- |
| **G1** | 15.93 | 11.62 | 509.16 | 14.86 |
| **G2** | 15.45 | 11.21 | 526.91 | 14.57 |
| **G3** | 15.02 | 10.62 | 518.31 | 13.75 |
| **G4** | **14.78** | 10.89 | **376.89** | 14.17 |

## Key Findings

- **Best Overall Mean Performance in G4**: The G4 configuration achieved the best overall mean RMSE and the lowest mean NASA Score across multiple seeds.
- **Best Single-Run RMSE in G3**: The G3 configuration achieved the strongest best-case RMSE among all groups.
- **Decision-Relevant Modeling Value**: The results suggest that asymmetric loss and uncertainty estimation can improve RUL modeling beyond a standard MSE baseline.
