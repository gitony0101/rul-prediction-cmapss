# Variability-Aware Risk-Aware RUL Prediction for Engineering Maintenance Decision Support

### A controlled CNN BiLSTM study on NASA C-MAPSS evaluating asymmetric LinEx training, Monte Carlo Dropout, predictive variability, low-RUL overestimation risk, and maintenance decision stability.

This project studies Remaining Useful Life (RUL) prediction as an engineering decision-support problem. It evaluates models beyond average RMSE by comparing prediction variability, low-RUL overestimation, and maintenance-oriented proxy behavior. The central insight is that the model with the best average accuracy is not necessarily the most attractive model for maintenance decisions if its predictions are more variable or more optimistic in the danger zone.

## Project Overview

In industrial maintenance, the cost of a "false negative" (overestimating RUL and delaying maintenance) is often much higher than a "false positive" (underestimating RUL and performing early maintenance). This project implements a risk-aware RUL prediction pipeline that accounts for this asymmetry using the **LinEx asymmetric loss function** and evaluates the stability of predictions using **Monte Carlo Dropout**.

## Why Variability Matters in Engineering

Engineering decisions rely on stability. A model with a slightly higher mean error but much lower variability is often preferred over a highly accurate but erratic model. Wide prediction spreads lead to unstable maintenance intervals, while overestimation in the "danger zone" (low remaining life) directly increases the risk of catastrophic failure.

## Research Question

Under asymmetric failure cost conditions, which RUL prediction setting provides the best balance among average accuracy, predictive variability, low-RUL overestimation risk, and engineering decision stability?

## Experimental Design

The study follows a controlled 2x2 factorial design:

| Group | Loss | Monte Carlo Dropout | Interpretive Role |
| :--- | :--- | :--- | :--- |
| **G1** | MSE | No | Symmetric deterministic baseline |
| **G2** | MSE | Yes | Symmetric model with uncertainty layer |
| **G3** | LinEx | No | Deterministic asymmetric comparison |
| **G4** | LinEx | Yes | Asymmetric model with uncertainty layer |

**Error Convention:** $e = y_{pred} - y_{true}$
*   $e > 0$: RUL Overestimation (Dangerous)
*   $e < 0$: RUL Underestimation (Conservative)

**LinEx Loss:** $L(e; a) = \exp(ae) - ae - 1$
When $a > 0$, overestimation is penalized exponentially more than underestimation.

## Model Architecture

The core model is a **CNN-BiLSTM** network:
1.  **CNN Layer**: Feature extraction from multi-sensor time-series windows.
2.  **BiLSTM Layers**: Capturing bidirectional temporal dependencies in engine degradation.
3.  **Dense Layers**: Final RUL regression.
4.  **Monte Carlo Dropout (Optional)**: Applied during inference to generate predictive distributions and variability proxies.

## Key Findings (Canonical FD001)

On canonical FD001, the deterministic LinEx model (**G3**) achieves the strongest predictive core according to RMSE, MAE, and NASA score. However, from a variability-aware engineering perspective, **G4** (LinEx + MCD) is more interesting because its mean RMSE is only slightly worse than G3 while its variability is substantially lower, its danger-zone overestimation is lower, and its fixed-threshold maintenance proxy cost is the lowest among the four groups.

| Group | RMSE (mean ± std) | MAE (mean ± std) | NASA Score (mean ± std) | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **G1** | 15.93 ± 0.85 | 11.62 ± 0.64 | 509.16 ± 159.18 | Baseline |
| **G2** | 15.45 ± 0.97 | 11.21 ± 0.71 | 526.91 ± 209.91 | High Variance |
| **G3** | 15.02 ± 0.82 | 10.62 ± 0.75 | 518.31 ± 113.24 | Best Accuracy |
| **G4** | 14.78 ± 0.38 | 10.89 ± 0.28 | 376.89 ± 59.92 | Best Stability |

## Engineering Takeaway

From an accuracy-only perspective, G3 would be selected for its superior RMSE and NASA score. However, for maintenance decision support, **G4 is the more robust candidate**. It provides a tighter, more stable prediction profile and minimizes risk where it matters most: the danger zone. This study demonstrates that RUL model selection must evaluate both average performance and predictive variability to ensure safe and stable engineering decisions.

## Repository Structure

```text
risk-aware-rul/
  ├── docs/               # Engineering interpretation and analysis
  ├── src/                # Core implementation (Data, Models, Training)
  ├── scripts/            # Reproducibility and figure generation
  ├── configs/            # Experiment configurations
  ├── results/            # Canonical summaries and figures
  └── tests/              # Integrity and sign-convention tests
```

## How to Reproduce

### Single Group Run (Quick Start)
```bash
python scripts/G1_run.py
python scripts/G2_run.py
python scripts/G3_run.py
python scripts/G4_run.py
```

### Full Multi-Seed Campaign
```bash
# Run 5 seeds for G4
python scripts/03_run_multiseed.py --group G4

# Summarize results
python scripts/04_summarize_results.py --group G4
python scripts/compare_groups.py
```

### Generate Figures
```bash
python scripts/05_build_figures.py
```

## Canonical Output Policy

Only results passing the **Final-Report Eligibility** check are included in public-facing summaries. This ensures every claim is backed by a run with a valid manifest, consistent schema, and known protocol version.

## Limitations

1.  **Asymmetric Loss**: LinEx is used as a controlled objective; specific cost calibration for real-world deployment is out of scope.
2.  **Uncertainty Proxy**: MCD standard deviation is a variability proxy, not a calibrated probability.
3.  **Maintenance Proxy**: The evaluation uses a fixed-threshold proxy, not a dynamic maintenance policy.
4.  **Checkpoint Governance**: Models are currently selected via validation RMSE, which may mismatch the asymmetric training objective.

## LinkedIn and Resume Summary

See [docs/linkedin_resume_summary.md](docs/linkedin_resume_summary.md) for ready-to-use project descriptions.
