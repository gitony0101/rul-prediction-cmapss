# Maintenance Proxy Interpretation

To evaluate models beyond statistical metrics, we implement a fixed-threshold maintenance proxy.

### 1. The Proxy Logic
We simulate a maintenance policy where a part is replaced as soon as the predicted RUL falls below a fixed threshold (e.g., 15 cycles).

### 2. Cost Components
- **Failure Event (Costly)**: $y_{true} < 0$ at the time of prediction (predicted RUL was too high).
- **Preventive Action (Standard)**: Maintenance performed before failure.
- **Unnecessary Maintenance (Inefficient)**: Maintenance performed much too early (e.g., $y_{true} > 30$).

### 3. Group Rankings
G4 consistently achieves the lowest total proxy cost. Even though G3 is more accurate on average, G4's conservative bias leads to fewer failure events and more stable maintenance intervals.

### 4. Decision Stability
Because G4 has lower variability, the maintenance intervals it triggers are more predictable, which simplifies logistics and spare parts management.
