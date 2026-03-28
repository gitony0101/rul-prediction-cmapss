# Checkpoint Results Bundle

## 1. Main group comparison

| group | num_runs | rmse_mean | rmse_std | mae_mean | nasa_mean | val_rmse_mean | pred_std_mean_mean |
|---|---|---|---|---|---|---|---|
| G4 | 5 | 14.7826 | 0.3776 | 10.8873 | 376.8906 | 13.9960 | 10.7760 |
| G3 | 5 | 15.0216 | 0.8197 | 10.6186 | 518.3084 | 11.5966 | nan |
| G2 | 5 | 15.4512 | 0.9691 | 11.2141 | 526.9068 | 13.2415 | 13.2107 |
| G1 | 5 | 15.9345 | 0.8528 | 11.6208 | 509.1627 | 11.7541 | nan |

## 2. Round 3 summaries

### G2
- rmse_mean: 15.4512
- mae_mean: 11.2141
- nasa_mean: 526.9068
- pearson_unc_abs_err_mean: 0.3447
- spearman_unc_abs_err_mean: 0.4397
- ece_abs_err_vs_unc_mean: 4.4877
- top_bottom_error_ratio_mean: 3.6856
- coverage_1sigma_mean: 0.6360
- coverage_2sigma_mean: 0.8640

### G4
- rmse_mean: 14.7826
- mae_mean: 10.8873
- nasa_mean: 376.8906
- pearson_unc_abs_err_mean: 0.2690
- spearman_unc_abs_err_mean: 0.3243
- ece_abs_err_vs_unc_mean: 4.2085
- top_bottom_error_ratio_mean: 2.7200
- coverage_1sigma_mean: 0.5480
- coverage_2sigma_mean: 0.8000

## 3. Round 4 summaries

### G1
- rmse_mean: 15.9345
- mae_mean: 11.6208
- nasa_mean: 509.1627
- mean_error_mean: -0.0618
- danger_rmse_mean: 4.0930
- danger_nasa_mean: 6.6626
- safe_rmse_mean: 17.2883
- safe_nasa_mean: 502.5001

### G2
- rmse_mean: 15.4512
- mae_mean: 11.2141
- nasa_mean: 526.9068
- mean_error_mean: -0.3182
- danger_rmse_mean: 3.6503
- danger_nasa_mean: 5.6335
- safe_rmse_mean: 16.7807
- safe_nasa_mean: 521.2733

### G3
- rmse_mean: 15.0216
- mae_mean: 10.6186
- nasa_mean: 518.3084
- mean_error_mean: 1.1152
- danger_rmse_mean: 4.0359
- danger_nasa_mean: 6.2798
- safe_rmse_mean: 16.2891
- safe_nasa_mean: 512.0286

### G4
- rmse_mean: 14.7826
- mae_mean: 10.8873
- nasa_mean: 376.8906
- mean_error_mean: -1.2883
- danger_rmse_mean: 3.9893
- danger_nasa_mean: 6.3165
- safe_rmse_mean: 16.0275
- safe_nasa_mean: 370.5741

## 4. Round 5 summaries

### G1
- best_threshold: 10.0000
- best_total_cost_mean: 4.9385
- best_failure_event_mean: 0.4300
- best_unnecessary_maintenance_mean: 0.4500
- best_preventive_action_mean: 0.5700
- best_protected_danger_mean: 0.5700
- best_early_lead_mean: 1.3700

### G2
- best_threshold: 35.0000
- best_total_cost_mean: 4.7000
- best_failure_event_mean: 0.3900
- best_unnecessary_maintenance_mean: 0.5400
- best_preventive_action_mean: 0.6100
- best_protected_danger_mean: 0.6100
- best_early_lead_mean: 3.8000

### G3
- best_threshold: 10.0000
- best_total_cost_mean: 5.4650
- best_failure_event_mean: 0.4900
- best_unnecessary_maintenance_mean: 0.3900
- best_preventive_action_mean: 0.5100
- best_protected_danger_mean: 0.5100
- best_early_lead_mean: 1.1000

### G4
- best_threshold: 40.0000
- best_total_cost_mean: 4.9000
- best_failure_event_mean: 0.4100
- best_unnecessary_maintenance_mean: 0.5400
- best_preventive_action_mean: 0.5900
- best_protected_danger_mean: 0.5900
- best_early_lead_mean: 4.2000
