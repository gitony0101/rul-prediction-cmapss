

# CS 6516 Project · Checkpoint Summary

## 1. Project Status

At this checkpoint, the `experiments_v2` pipeline has been rebuilt and stabilized around a controlled 2×2 design on C-MAPSS FD001.

The four primary groups are:

| Group | Loss | MC Dropout | Meaning |
|---|---|---:|---|
| G1 | MSE | No | deterministic baseline |
| G2 | MSE | Yes | uncertainty added under standard symmetric loss |
| G3 | LinEx | No | risk-sensitive loss without MCD |
| G4 | LinEx | Yes | risk-sensitive loss with MCD |

The current pipeline includes:

- primary multi-seed training
- protocol checking
- Round 3 uncertainty validity analysis
- Round 4 error structure analysis
- Round 5 validation full-trajectory maintenance simulation

All four primary groups passed protocol checking, which confirms that the run directories, metrics files, prediction files, and split records are structurally consistent.

---

## 2. Main 2×2 Results

### 2.1 Overall performance ranking

Across the current multi-seed comparison on FD001, the average test RMSE ranking is:

\[
G4 < G3 < G2 < G1
\]

The current group comparison is:

| Group | num_runs | rmse_mean | rmse_std | mae_mean | nasa_mean | val_rmse_mean |
|---|---:|---:|---:|---:|---:|---:|
| G4 | 5 | 14.7826 | 0.7365 | 10.6275 | 360.6944 | 11.3785 |
| G3 | 5 | 15.0216 | 0.6494 | 10.5824 | 489.4751 | 11.5963 |
| G2 | 5 | 15.4512 | 0.5747 | 10.9471 | 575.3778 | 12.7079 |
| G1 | 5 | 15.9345 | 0.3873 | 11.6208 | 509.8733 | 12.2687 |

### 2.2 Main interpretation

These results support three core conclusions.

First, **G4 is currently the best overall configuration** in terms of average test RMSE.

Second, **LinEx appears to be the primary gain source**. This is most clearly seen from the controlled comparison between G1 and G3. When only the loss is changed from MSE to LinEx, performance improves meaningfully.

Third, **MC Dropout alone provides limited improvement under MSE**, since G2 improves only modestly relative to G1. However, when MCD is combined with LinEx, the resulting G4 configuration achieves the best overall performance.

This supports the current project narrative:

\[
\text{LinEx is the main innovation signal, and MCD acts as an auxiliary enhancement.}
\]

---

## 3. Round 3 · MCD Validity

## 3.1 Goal

Round 3 evaluates whether MC Dropout uncertainty estimates are actually informative.

This round is only applied to the two MCD groups:

- G2
- G4

## 3.2 Summary results

### G4

- `rmse_mean = 14.8679`
- `mae_mean = 10.9469`
- `nasa_mean = 371.6499`
- `pearson_unc_abs_err_mean = 0.2394`
- `spearman_unc_abs_err_mean = 0.3127`
- `ece_abs_err_vs_unc_mean = 4.6509`
- `top_bottom_error_ratio_mean = 2.4867`
- `coverage_1sigma_mean = 0.5620`
- `coverage_2sigma_mean = 0.7920`

### G2

- `rmse_mean = 15.4691`
- `mae_mean = 11.0759`
- `nasa_mean = 580.1129`
- `pearson_unc_abs_err_mean = 0.3547`
- `spearman_unc_abs_err_mean = 0.4261`
- `ece_abs_err_vs_unc_mean = 4.5218`
- `top_bottom_error_ratio_mean = 3.7302`
- `coverage_1sigma_mean = 0.6440`
- `coverage_2sigma_mean = 0.9000`

## 3.3 Interpretation

Round 3 reveals an important separation between **point prediction quality** and **uncertainty quality**.

G4 is better in point prediction, since it has lower RMSE and lower NASA.

However, G2 shows stronger uncertainty-error coupling:

- higher Pearson correlation between uncertainty and absolute error
- higher Spearman correlation
- stronger top-bottom error separation
- coverage closer to the nominal interval intuition

This means that the model with the best predictive accuracy does not necessarily produce the strongest uncertainty-quality signal.

The current interpretation is therefore:

\[
\text{G4 is the stronger predictor, but G2 may be the stronger uncertainty reporter.}
\]

This is a valuable result for the project because it suggests that uncertainty usefulness and prediction accuracy should be treated as related but distinct questions.

---

## 4. Round 4 · Error Structure

## 4.1 Goal

Round 4 studies where each group wins or loses across:

- overall error
- danger zone
- safe zone
- error direction
- severe and moderate deviations

## 4.2 Main summary

### G1

- `rmse_mean = 15.9345`
- `nasa_mean = 509.8733`
- `mean_error_mean = 2.9915`
- `danger_rmse_mean = 4.4529`
- `danger_nasa_mean = 7.3705`
- `safe_rmse_mean = 17.2390`
- `safe_nasa_mean = 502.5029`

### G2

- `rmse_mean = 15.4512`
- `nasa_mean = 575.3778`
- `mean_error_mean = 0.5327`
- `danger_rmse_mean = 3.6503`
- `danger_nasa_mean = 5.6335`
- `safe_rmse_mean = 17.1926`
- `safe_nasa_mean = 521.2675`

### G3

- `rmse_mean = 15.0216`
- `nasa_mean = 489.4751`
- `mean_error_mean = 1.1152`
- `danger_rmse_mean = 3.8264`
- `danger_nasa_mean = 6.1223`
- `safe_rmse_mean = 16.4628`
- `safe_nasa_mean = 512.0266`

### G4

- `rmse_mean = 14.7826`
- `nasa_mean = 360.6944`
- `mean_error_mean = -1.2883`
- `danger_rmse_mean = 3.9893`
- `danger_nasa_mean = 6.3165`
- `safe_rmse_mean = 15.9404`
- `safe_nasa_mean = 370.5663`

## 4.3 Interpretation

Round 4 gives a more nuanced explanation than the overall RMSE ranking.

### 4.3.1 G4 is best overall

G4 has the best overall RMSE and the best overall NASA. This confirms that it is the strongest overall model in the current 2×2 design.

### 4.3.2 G4 is not best everywhere

G4 is not the absolute best group in the danger zone. In fact, G2 has lower danger-zone RMSE and lower danger-zone NASA than G4.

This means the current advantage of G4 cannot be summarized as “best in every region.”

### 4.3.3 G4’s biggest gain appears in the safe zone

The strongest separation occurs in safe-zone NASA:

- G1: 502.5029
- G2: 521.2675
- G3: 512.0266
- G4: 370.5663

This suggests that the largest advantage of G4 comes from global risk control outside the strict danger zone.

### 4.3.4 LinEx changes error bias in a risk-consistent direction

G1 and G3 both still show positive mean error, indicating an overall tendency toward overestimation.

G4, by contrast, shows negative mean error, and its safe-zone mean error is even more negative. This means G4 is more biased toward underestimation.

In the project setting, moderate underestimation is often safer than overestimation. Therefore, this shift is consistent with the intended role of LinEx.

## 4.4 Current Round 4 conclusion

Round 4 supports the following interpretation:

\[
\text{G4’s overall advantage comes less from absolute danger-zone dominance and more from safer global error structure, especially in the safe zone.}
\]

At the same time, the G1 to G3 comparison shows that LinEx alone already improves structure and overall performance, which strengthens the claim that the loss function is the main driver.

---

## 5. Round 5 · Validation Full-Trajectory Maintenance Simulation

## 5.1 Goal

Round 5 translates prediction differences into threshold-based maintenance decisions.

This version uses **validation full trajectories**, reconstructed from `split_units.json`, rather than the official censored test split.

That change is essential because policy simulation requires complete run-to-failure trajectories.

## 5.2 Why this version matters

Earlier threshold-based maintenance analysis on the test split can produce misleading zero-cost or trivial results because the official C-MAPSS test units are censored.

The current Round 5 avoids that issue by:

1. fitting the normalizer on `train_units`
2. reconstructing `val_units`
3. performing `mode="all"` inference on full validation trajectories
4. simulating threshold crossing, preventive action, missed danger, and total cost at the unit level

## 5.3 Current maintenance results

### G1

- `best_threshold = 10.0`
- `best_total_cost_mean = 4.9385`
- `best_failure_event_mean = 0.4300`
- `best_unnecessary_maintenance_mean = 0.4500`
- `best_preventive_action_mean = 0.5700`
- `best_protected_danger_mean = 0.5700`
- `best_early_lead_mean = 1.3700`

### G2

- `best_threshold = 30.0`
- `best_total_cost_mean = 4.2925`
- `best_failure_event_mean = 0.3500`
- `best_unnecessary_maintenance_mean = 0.5500`
- `best_preventive_action_mean = 0.6500`
- `best_protected_danger_mean = 0.6500`
- `best_early_lead_mean = 2.8500`

### G3

- `best_threshold = 10.0`
- `best_total_cost_mean = 5.1163`
- `best_failure_event_mean = 0.4500`
- `best_unnecessary_maintenance_mean = 0.3500`
- `best_preventive_action_mean = 0.5500`
- `best_protected_danger_mean = 0.5500`
- `best_early_lead_mean = 1.3250`

### G4

- `best_threshold = 15.0`
- `best_total_cost_mean = 5.1275`
- `best_failure_event_mean = 0.4500`
- `best_unnecessary_maintenance_mean = 0.4000`
- `best_preventive_action_mean = 0.5500`
- `best_protected_danger_mean = 0.5500`
- `best_early_lead_mean = 1.5500`

## 5.4 Interpretation

Round 5 is now structurally valid, because it uses validation full trajectories.

The resulting thresholds and costs are nontrivial, which indicates that the simulation is behaving like an actual policy layer rather than a static threshold sanity check.

However, maintenance conclusions should still be presented carefully.

The current evidence suggests:

- different groups prefer different operating thresholds
- G2 may look more favorable under the current maintenance cost design
- G4 remains the strongest overall predictor
- the best predictive model does not automatically become the cheapest maintenance policy under every threshold rule

This is actually a meaningful result. It shows that **prediction quality and policy quality are related, but not identical**.

That distinction is useful for the final write-up.

---

## 6. Consolidated Interpretation

At this checkpoint, the most defensible project narrative is the following.

### 6.1 LinEx is the main gain source

This is supported by the controlled G1 versus G3 comparison. Changing the loss from MSE to LinEx yields a clear performance improvement.

### 6.2 G4 is the best overall prediction configuration

G4 currently has the best overall RMSE and the best overall NASA.

### 6.3 MCD alone provides limited improvement in point prediction

G2 does not outperform G3, and its gain over G1 is modest.

### 6.4 MCD still matters analytically

Round 3 shows that uncertainty quality can behave differently from point prediction quality. G2 may provide stronger uncertainty-error coupling than G4.

### 6.5 G4’s advantage is structural

Round 4 shows that G4’s overall win is not just a single-number effect. Its strongest advantage appears in safer global error structure, especially in the safe zone.

### 6.6 Maintenance policy is a separate layer

Round 5 shows that the model with the strongest point prediction does not automatically dominate every policy objective. This reinforces the need to treat decision analysis as its own stage.

---

## 7. What is already stable

The following points are now stable enough to use in a checkpoint report:

1. the 2×2 design is implemented and reproducible
2. all four groups pass protocol checking
3. LinEx is the primary performance-improving factor
4. G4 is currently the strongest overall prediction model
5. Round 3, Round 4, and Round 5 are all running under a coherent protocol
6. Round 5 has been upgraded to a valid validation full-trajectory simulation

---

## 8. What still remains open

The following questions remain open and are good candidates for the next stage:

1. whether the current maintenance ranking remains stable across more complete runs and cost settings
2. whether the findings generalize beyond FD001
3. whether the CNN front-end is truly necessary
4. whether a synthetic toy mechanism can explain the observed LinEx behavior under asymmetric risk

---

## 9. Recommended Next Step

The next most valuable steps are:

### Option A
Write up the current checkpoint and stabilize the documentation.

### Option B
Proceed to a minimal next experiment, preferably one of:

- Round 6 minimal ablation
- Round 7 multi-dataset transfer
- Round 8 toy mechanism

Among these, Round 8 is likely the most aligned with the course emphasis on controlled mechanism validation.

---

## 10. Current One-Sentence Project Summary

The current evidence suggests that **risk-sensitive loss design is the main reason for improved RUL prediction performance, while MC Dropout contributes more as a secondary uncertainty and decision-support component than as the primary source of predictive gain.**
