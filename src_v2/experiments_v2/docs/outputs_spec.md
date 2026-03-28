

# CS 6516 Project · Outputs Specification
## 1. Purpose
This document specifies the required outputs for `experiments_v2` and the meaning of each file.
The specification is intended to ensure:
- consistency across groups
- compatibility with downstream analysis
- clean reproducibility
- reliable reporting in the final write-up
---
## 2. Output root
All artifacts are written under:

outputs_v2/


⸻

3. Primary group output specification

Each main group has a dedicated output directory:
	•	outputs_v2/G1_CNN_BiLSTM_MSE/
	•	outputs_v2/G2_CNN_BiLSTM_MSE_MCD/
	•	outputs_v2/G3_CNN_BiLSTM_LinEx/
	•	outputs_v2/G4_CNN_BiLSTM_LinEx_MCD/

Each run must live in its own subdirectory.

Examples:
	•	seed_42
	•	seed_87
	•	seed_42_a_0.04
	•	seed_123_a_0.04

⸻

4. Required files inside each run directory

Every run directory must contain the following files.

4.1 best_model.pth

Best checkpoint selected by validation criterion.

Meaning:
	•	model weights at best validation epoch

Required for:
	•	inference
	•	Round 5 reconstruction
	•	later policy evaluation

⸻

4.2 last_model.pth

Last checkpoint at the final training epoch.

Meaning:
	•	model weights at the end of training

Use:
	•	diagnostic comparison
	•	debugging
	•	recovery if needed

⸻

4.3 train_history.csv

Epoch-level training log.

Required columns:
	•	epoch
	•	train_loss
	•	val_loss

May also include:
	•	val_rmse_det
	•	val_mae_det
	•	val_nasa_det
	•	linex_a

Purpose:
	•	inspect convergence
	•	inspect overfitting
	•	identify early stopping behavior

⸻

4.4 predictions.csv

Prediction file used by downstream rounds.

Required base columns:
	•	y_true
	•	y_pred

For MCD groups, required additional columns:
	•	pred_mean
	•	pred_std

Notes:
	•	for deterministic groups, y_pred is the point prediction
	•	for MCD groups, pred_mean is the MC mean prediction
	•	y_pred may duplicate pred_mean for compatibility
	•	metadata columns such as unit_id and cycle are allowed and encouraged

Used by:
	•	Round 3
	•	Round 4
	•	compatibility checks
	•	sanity inspection

⸻

4.5 config.json

Full run configuration.

Required keys:
	•	experiment_name
	•	dataset_name
	•	seed
	•	data_dir
	•	output_root
	•	seq_len
	•	rul_cap
	•	validation_ratio
	•	hidden_size
	•	num_lstm_layers
	•	dense_size
	•	cnn_out_channels
	•	cnn_kernel_size
	•	cnn_stride
	•	cnn_pool_size
	•	batch_size
	•	lr
	•	weight_decay
	•	num_epochs
	•	early_stopping_patience
	•	grad_clip

Additional required keys for MCD groups:
	•	dropout_rate
	•	mc_samples_val
	•	mc_samples_test

Additional required key for LinEx groups:
	•	linex_a

Purpose:
	•	full reproducibility
	•	protocol checking
	•	Round 5 reconstruction

⸻

4.6 normalizer.json

Normalization statistics.

Must contain:
	•	feature names
	•	means
	•	standard deviations

Purpose:
	•	audit normalization
	•	ensure consistent data preprocessing
	•	optional reconstruction support

⸻

4.7 split_units.json

Validation split record.

Required keys:
	•	train_units
	•	val_units

Purpose:
	•	reproducible split reconstruction
	•	Round 5 validation full-trajectory simulation

This file is critical.
Without it, validation policy simulation should not be trusted.

⸻

4.8 val_metrics.json

Validation metrics summary.

Required for all groups:
	•	rmse
	•	mae
	•	nasa

Required for MCD groups:
	•	pred_std_mean
	•	pred_std_std
	•	mc_samples

Purpose:
	•	group summary
	•	protocol checking
	•	comparison of deterministic and uncertainty-aware validation behavior

⸻

4.9 test_metrics.json

Test metrics summary.

Required for all groups:
	•	rmse
	•	mae
	•	nasa

Required for MCD groups:
	•	pred_std_mean
	•	pred_std_std
	•	mc_samples

Purpose:
	•	multi-seed group comparison
	•	primary performance reporting

⸻

4.10 best_summary.json

Best checkpoint summary.

Required keys:
	•	best_epoch
	•	best_val_rmse

Purpose:
	•	concise model selection record
	•	report generation
	•	quick inspection of training outcome

⸻

5. Multi-seed summary specification

Each group must have a multi-seed log directory:
	•	outputs_v2/G1_multiseed_logs/
	•	outputs_v2/G2_multiseed_logs/
	•	outputs_v2/G3_multiseed_logs/
	•	outputs_v2/G4_multiseed_logs/

Each must contain:
	•	{group}_multiseed_detail.csv
	•	{group}_multiseed_summary.json
	•	{group}_multiseed_protocol_check.json
	•	{group}_multiseed_manifest.json

⸻

6. Group comparison specification

File:

outputs_v2/group_comparison.csv

This is the primary cross-group comparison table.

Expected columns include:
	•	group
	•	num_runs
	•	rmse_mean
	•	rmse_std
	•	mae_mean
	•	mae_std
	•	nasa_mean
	•	nasa_std
	•	best_rmse
	•	best_seed
	•	best_run_dir
	•	best_linex_a
	•	val_rmse_mean
	•	val_rmse_std
	•	pred_std_mean_mean
	•	pred_std_mean_std

Notes:
	•	best_linex_a is only relevant to LinEx groups
	•	pred_std_mean_mean is only relevant to MCD groups

Purpose:
	•	main result table
	•	checkpoint reporting
	•	final write-up summary

⸻

7. Protocol check specification

Directory:

outputs_v2/protocol_checks/

Required files:
	•	protocol_check_detail.json
	•	protocol_check_summary.csv
	•	protocol_check_manifest.json

Purpose:
	•	verify output completeness
	•	verify metric field consistency
	•	detect missing files before round analysis

Protocol check must pass before analysis results are treated as final.

⸻

8. Round 3 outputs

Round 3 is only for G2 and G4.

Directories:
	•	outputs_v2/G2_round3_mcd_validity/
	•	outputs_v2/G4_round3_mcd_validity/

Required files:
	•	round3_detail.csv
	•	round3_summary.json
	•	round3_manifest.json

round3_detail.csv

One row per run.

Expected fields include:
	•	run_dir
	•	num_samples
	•	rmse
	•	mae
	•	nasa
	•	pred_std_mean
	•	pred_std_std
	•	abs_err_mean
	•	pearson_unc_abs_err
	•	spearman_unc_abs_err
	•	ece_abs_err_vs_unc
	•	top_bottom_error_ratio
	•	coverage_1sigma
	•	coverage_2sigma

round3_summary.json

Aggregated group-level uncertainty validity summary.

Used to interpret:
	•	uncertainty-error coupling
	•	interval coverage quality
	•	relative uncertainty usefulness between G2 and G4

⸻

9. Round 4 outputs

Directory:

outputs_v2/round4_error_structure/

Each group must have its own subdirectory:
	•	G1/
	•	G2/
	•	G3/
	•	G4/

Each group directory must contain:
	•	round4_detail.csv
	•	round4_summary.json
	•	round4_bins_detail.csv
	•	round4_bins_summary.csv

round4_detail.csv

One row per run.

Expected fields include:
	•	overall metrics
	•	mean error
	•	mean absolute error
	•	overestimate rate
	•	underestimate rate
	•	severe/moderate error probabilities
	•	danger-zone metrics
	•	safe-zone metrics

round4_summary.json

Group-level aggregate of run-wise Round 4 results.

This file is the primary Round 4 interpretation input.

round4_bins_detail.csv

Per-run, per-bin error structure data.

round4_bins_summary.csv

Group-level aggregated per-bin statistics.

Purpose:
	•	identify where performance gains occur across the RUL range
	•	distinguish danger-zone benefit from safe-zone benefit

⸻

10. Round 5 outputs

Directory:

outputs_v2/round5_maintenance/

Each group must have:
	•	G1/
	•	G2/
	•	G3/
	•	G4/

Each group directory must contain:
	•	round5_detail.csv
	•	round5_threshold_summary.csv
	•	round5_summary.json

Global file:
	•	round5_manifest.json

⸻

10.1 round5_detail.csv

One row per:
	•	run
	•	threshold
	•	validation unit

This is the most granular policy evaluation output.

Expected fields include:
	•	unit_id
	•	threshold
	•	num_windows
	•	first_cycle
	•	last_cycle
	•	last_true_rul
	•	last_decision_rul
	•	true_cross_exists
	•	pred_cross_exists
	•	true_cross_cycle
	•	pred_cross_cycle
	•	true_cross_rul
	•	pred_cross_rul
	•	pred_cross_true_rul
	•	action
	•	action_cycle
	•	action_true_rul
	•	action_pred_rul
	•	preventive_action
	•	protected_danger
	•	failure_event
	•	unnecessary_maintenance
	•	early_lead
	•	maintenance_cost
	•	failure_cost
	•	total_cost
	•	run_dir

⸻

10.2 round5_threshold_summary.csv

One row per threshold.

Expected aggregate fields include:
	•	threshold
	•	num_runs
	•	num_units
	•	preventive_action_mean
	•	preventive_action_std
	•	protected_danger_mean
	•	protected_danger_std
	•	failure_event_mean
	•	failure_event_std
	•	unnecessary_maintenance_mean
	•	unnecessary_maintenance_std
	•	early_lead_mean
	•	early_lead_std
	•	maintenance_cost_mean
	•	maintenance_cost_std
	•	failure_cost_mean
	•	failure_cost_std
	•	total_cost_mean
	•	total_cost_std

This file is the main threshold scan for maintenance analysis.

⸻

10.3 round5_summary.json

Best-threshold summary for a group.

Required fields:
	•	group
	•	num_thresholds
	•	best_threshold
	•	best_total_cost_mean
	•	best_total_cost_std
	•	best_failure_event_mean
	•	best_failure_event_std
	•	best_unnecessary_maintenance_mean
	•	best_unnecessary_maintenance_std
	•	best_preventive_action_mean
	•	best_preventive_action_std
	•	best_protected_danger_mean
	•	best_protected_danger_std
	•	best_early_lead_mean
	•	best_early_lead_std

Purpose:
	•	concise decision-level comparison across groups

⸻

11. Round 5 validity rule

A Round 5 result is considered valid only if:
	1.	it is generated from validation full trajectories
	2.	split_units.json is used
	3.	train_units are used to fit the normalizer
	4.	val_units are used for full-trajectory policy evaluation
	5.	all groups being compared have the intended number of runs

If these conditions are not met, Round 5 results should be treated as provisional only.

⸻

12. Naming conventions

Naming conventions must be stable.

Examples:
	•	deterministic run:
	•	seed_42
	•	LinEx run:
	•	seed_42_a_0.04

Do not mix naming styles within the same group.

⸻

13. What should be deleted before a clean rerun

Before a full rerun, the following should be deleted or regenerated:
	•	all group output directories that are being rerun
	•	all multi-seed logs
	•	group comparison CSV
	•	protocol check outputs
	•	Round 3 outputs
	•	Round 4 outputs
	•	Round 5 outputs

This prevents stale derived files from being interpreted as current results.

⸻

14. Minimum files needed for final report

At minimum, the final report should be backed by:
	•	four complete primary group directories
	•	four multi-seed summaries
	•	one group_comparison.csv
	•	one passing protocol check
	•	Round 3 summaries for G2 and G4
	•	Round 4 summaries for all groups
	•	Round 5 summaries for all groups using validation full trajectories

⸻

15. Recommended interpretation order

When reading outputs, use this order:
	1.	group_comparison.csv
	2.	round3_summary.json
	3.	round4_summary.json
	4.	round5_summary.json

This keeps the interpretation chain aligned with the project logic:
	•	point prediction first
	•	uncertainty second
	•	error structure third
	•	maintenance policy last




