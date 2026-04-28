# Reproducibility and Canonical Protocol

This project adheres to a strict canonical protocol to ensure all results are auditable and reproducible.

### 1. Canonical Output Policy
Only results passing the **Final-Report Eligibility** check are included in public summaries. A run is eligible if it contains:
- Valid `config.json`
- Completed `test_metrics.json`
- Consistent prediction schema (`y_true`, `y_pred`)
- Known source script and protocol version

### 2. Execution Environment
Results are verified on **Apple Silicon (MPS)** and **NVIDIA CUDA**.
- `device`: recorded in diagnostics.
- `seed`: fixed for all canonical runs.

### 3. Data Splits
We use a fixed unit-based split for training and validation to prevent data leakage. Normalization statistics are fitted **only** on the training set.

### 4. Re-running the Campaign
To reproduce the full multiseed campaign:
```bash
python scripts/03_run_multiseed.py --group G1
python scripts/03_run_multiseed.py --group G2
python scripts/03_run_multiseed.py --group G3
python scripts/03_run_multiseed.py --group G4
python scripts/04_summarize_results.py
```
