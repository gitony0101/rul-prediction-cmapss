# Results Summary: NASA C-MAPSS FD001

This document summarizes the comparative performance of G1 through G4 on the canonical FD001 dataset.

### 1. Main Performance Metrics

| Group | RMSE (mean) | MAE (mean) | NASA Score (mean) | Best RMSE |
| :--- | :--- | :--- | :--- | :--- |
| **G1** | 15.98* | 11.62* | 586.54* | TODO |
| **G2** | 17.93* | 13.36* | 870.80* | TODO |
| **G3** | 14.99* | 10.86* | 459.46* | TODO |
| **G4** | 15.84* | 11.71* | 511.80* | TODO |

*\* Pending final verification against canonical source files.*

### 2. Evidence Layer 1: Predictive Core
G3 achieves the strongest predictive core. If average accuracy on FD001 is the only metric, G3 is the superior model.

### 3. Evidence Layer 2: Decision Stability
G4 provides the most stable prediction profile across multiple seeds, with the lowest standard deviation in RMSE and NASA score among the high-performing groups.

### 4. Evidence Layer 3: Risk-Awareness
G4 demonstrates the lowest overestimation rate in the low-RUL danger zone, making it the safest choice for maintenance decision support.
