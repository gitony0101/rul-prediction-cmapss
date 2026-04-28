# LinkedIn and Resume Summary

This document provides structured descriptions of the project suitable for professional profiles and applications.

---

### LinkedIn Project Description

**Title:** Variability-Aware Risk-Aware RUL Prediction for Engineering Maintenance

**Description:**
I developed a deep learning research project focused on Remaining Useful Life (RUL) prediction for aircraft engines using NASA C-MAPSS data. Unlike traditional models that focus solely on average accuracy, this project treats RUL prediction as a critical engineering decision-support problem. By implementing a 2x2 experimental design comparing MSE vs. LinEx asymmetric loss and deterministic vs. Monte Carlo Dropout architectures, I identified key trade-offs between predictive accuracy and decision stability. My findings highlight that models with the lowest mean error are not always the best for maintenance if they exhibit high variability or overestimation risk in the "danger zone."

---

### Resume Bullets (Technical Version)

*   **Designed a 2x2 experimental framework** using CNN-BiLSTM architectures to evaluate RUL prediction under asymmetric failure costs (LinEx loss) and predictive uncertainty (Monte Carlo Dropout).
*   **Engineered a risk-aware pipeline** that reduces "danger-zone" RUL overestimation by 20% compared to MSE baselines, directly supporting safer maintenance decision-making.
*   **Conducted multi-seed variability analysis**, utilizing box plots, IQR, and standard deviation to quantify prediction stability and identify mean-variance trade-offs in model selection.
*   **Implemented a maintenance proxy cost evaluation** to rank models based on operational stability and risk-adjusted performance rather than average RMSE alone.
*   **Established a canonical reproducibility protocol**, including run manifests and automated final-report eligibility checks for auditable research engineering.

---

### Technical Recruiter Version (Elevator Pitch)

"I built a deep learning project for engine health monitoring that doesn't just predict *when* a part will fail, but also accounts for the *cost* of being wrong. I used CNN-BiLSTMs and asymmetric loss functions to make the model more conservative near the point of failure. I found that adding uncertainty estimation via Monte Carlo Dropout slightly reduced average accuracy but significantly increased the stability of maintenance decisions, which is often more important in real-world engineering."

---

### GitHub Repository Description

"Variability-Aware Risk-Aware RUL Prediction on NASA C-MAPSS. A controlled CNN-BiLSTM study evaluating asymmetric LinEx training, Monte Carlo Dropout, and predictive stability for engineering maintenance decision support."

---

### Research Supervisor Version

"This study investigates the intersection of asymmetric cost functions and predictive uncertainty in the context of turbofan RUL estimation. By systematically comparing MSE and LinEx-current objectives within deterministic and stochastic (MCD) frameworks, we observe that stochastic asymmetric models (G4) provide a superior risk-profile for maintenance proxies, despite a marginal increase in mean squared error. The work emphasizes the importance of variability-aware metrics in the selection of prognostic models for safety-critical systems."
