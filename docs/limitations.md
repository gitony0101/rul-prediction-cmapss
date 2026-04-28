# Project Limitations

This project is a controlled research study and should be interpreted with the following limitations in mind:

1.  **Asymmetric Loss Concept**: Asymmetric error cost is a well-established concept in statistics and engineering; this project does not claim it as a new idea but applies it as a controlled experimental variable.
2.  **LinEx Calibration**: LinEx-current is used as a controlled risk-aware objective. In a real-world deployment, the parameter $a$ would need careful calibration against actual failure costs and maintenance budgets.
3.  **Uncertainty Proxy**: The standard deviation derived from Monte Carlo Dropout is treated as a variability proxy for model epistemic uncertainty. It is **not** a calibrated probability of failure.
4.  **Maintenance Proxy**: The maintenance evaluation is based on a fixed-threshold proxy cost, not a comprehensive dynamic maintenance optimization policy.
5.  **Evidence Base**: The strongest current evidence is derived from canonical NASA C-MAPSS FD001. While indicative, these patterns may vary across different operating conditions or datasets.
6.  **Checkpoint Governance**: Models are currently trained with asymmetric objectives but selected based on **validation RMSE**. This creates a governance mismatch that may prevent the models from reaching their full "risk-aware" potential.
7.  **Sample Size**: Danger-zone analysis and multi-seed variability metrics are based on a limited number of seeds (typically 5). Larger scale validation is required for higher statistical confidence.
8.  **Execution Platform**: Results may vary slightly across different hardware (e.g., Apple Silicon MPS vs. NVIDIA CUDA) due to floating-point nondeterminism, although seeds are fixed.
