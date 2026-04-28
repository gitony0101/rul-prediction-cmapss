# Danger Zone Analysis

The "danger zone" is defined as the regime where the true Remaining Useful Life (RUL) is low (e.g., $y_{true} < 20$ cycles). Errors in this regime have disproportionate consequences.

### 1. Signed Error in the Danger Zone
We track the signed error ($e = y_{pred} - y_{true}$) specifically for samples in the danger zone.
- **Positive Error (Overestimation)**: Delays maintenance, increasing failure risk.
- **Negative Error (Underestimation)**: Triggers early maintenance, increasing cost but ensuring safety.

### 2. Overestimation Rate
G4 (LinEx + MCD) achieves the lowest danger-zone overestimation rate among the tested groups. This suggests that the combination of asymmetric loss and stochastic inference successfully pushes the model towards a more conservative, safety-first prediction profile.

### 3. RMSE vs. Danger-Zone Performance
A model can have a good overall RMSE by performing well in the high-RUL regime while still being dangerous in the low-RUL regime. This analysis ensures that such behavior is identified and penalized.
