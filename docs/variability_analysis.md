# Variability and Stability Analysis

In engineering systems, predictive variability is a primary concern. High variability leads to "jittery" maintenance schedules and reduced trust in automated decision support.

### 1. Multi-Seed Analysis
Every group in this study is evaluated across at least 5 random seeds to quantify the impact of weight initialization and data shuffling on final RUL predictions.

### 2. Variance vs. Accuracy Trade-off
We observe a clear mean-variance trade-off:
- **G3** (Deterministic LinEx) has the lowest mean error but higher variability across seeds.
- **G4** (LinEx + MCD) has a slightly higher mean error but significantly tighter standard deviation.

### 3. Box Plot Interpretation
Box plots (see `docs/figures/`) are used to visualize the IQR and outliers for every metric.
- Tight box plots indicate high decision stability.
- Large boxes or long whiskers indicate that the model's performance is highly sensitive to initialization.

### 4. Impact on Decision Support
A model with low variability allow engineers to set tighter maintenance intervals with higher confidence, reducing unnecessary downtime while maintaining a high safety margin.
