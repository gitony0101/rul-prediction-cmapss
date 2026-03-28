# CS 6516 Project · Experiment V2 Protocol

## 1. Purpose

This document defines the execution protocol for `src_v2/experiments_v2` and its associated outputs in `outputs_v2/`.

The goal is to ensure that all experiments are:

- reproducible
- comparable
- protocol-consistent
- suitable for checkpoint reporting and final write-up

This protocol applies to:

- four primary groups: `G1`, `G2`, `G3`, `G4`
- multi-seed runs
- post-hoc analysis rounds:
  - Round 3: MCD validity
  - Round 4: error structure
  - Round 5: maintenance simulation

---

## 2. Core Research Design

## 2.1 Main 2×2 design

The main experiment is a controlled 2×2 design:

| Group | Loss | MC Dropout | Meaning |
|---|---|---:|---|
| G1 | MSE | No | baseline deterministic predictor |
| G2 | MSE | Yes | uncertainty added without changing loss |
| G3 | LinEx | No | risk-sensitive loss only |
| G4 | LinEx | Yes | risk-sensitive loss + uncertainty |

This design allows the following controlled comparisons:

- `G1 vs G3`: effect of LinEx without MCD
- `G1 vs G2`: effect of MCD under MSE
- `G3 vs G4`: effect of MCD under LinEx
- `G2 vs G4`: effect of LinEx under MCD

---

## 2.2 Current experimental scope

Current primary dataset:

- `FD001`

Current primary LinEx setting:

- `a = 0.04`

Current primary random seeds:

- `7`
- `21`
- `42`
- `87`
- `123`

---

## 2.3 Main hypothesis

The current working hypothesis is:

1. LinEx provides the main performance gain in asymmetric-risk RUL prediction.
2. MC Dropout does not necessarily improve point prediction by itself.
3. MC Dropout may still be useful for uncertainty interpretation and decision support.
4. G4 is expected to be the strongest overall configuration, but its advantage may come more from safe-zone risk control than from absolute superiority in the danger zone.

---

## 3. File and Directory Structure

## 3.1 Source structure

src_v2/
├── rul/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── inference/
│   ├── evaluation/
│   └── utils.py
│
└── experiments_v2/
    ├── G1_run.py
    ├── G2_run.py
    ├── G3_run.py
    ├── G4_run.py
    ├── run_group_multiseed.py
    ├── summarize_group_multiseed.py
    ├── compare_groups.py
    ├── analysis/
    │   └── protocol_check.py
    ├── rounds/
    │   ├── round3_mcd_validity.py
    │   ├── round4_error_structure.py
    │   └── round5_maintenance.py
    └── docs/
        ├── protocol.md
        ├── outputs_spec.md
        └── experiment_map.md

3.2 Output structure

All generated artifacts are written under:

outputs_v2/


⸻

4. Primary Experiment Execution Protocol

4.1 Required execution order

The standard execution order is:
	1.	run primary groups
	2.	summarize each group
	3.	compare groups
	4.	run protocol check
	5.	run Round 3
	6.	run Round 4
	7.	run Round 5

No analysis round should be treated as final unless the corresponding primary group outputs exist and pass protocol checking.

⸻

4.2 Primary group commands

Run from:

src_v2/experiments_v2/

G1

python run_group_multiseed.py --group G1 --seeds 7 21 42 87 123

G2

python run_group_multiseed.py --group G2 --seeds 7 21 42 87 123 --mc-samples-val 10 --mc-samples-test 20

G3

python run_group_multiseed.py --group G3 --seeds 7 21 42 87 123 --linex-a 0.04

G4

python run_group_multiseed.py --group G4 --seeds 7 21 42 87 123 --linex-a 0.04 --mc-samples-val 10 --mc-samples-test 20


⸻

4.3 Summary and comparison commands

python summarize_group_multiseed.py --group G1
python summarize_group_multiseed.py --group G2
python summarize_group_multiseed.py --group G3
python summarize_group_multiseed.py --group G4
python compare_groups.py


⸻

4.4 Protocol check

Run from:

src_v2/experiments_v2/analysis/

python protocol_check.py

Protocol check must pass before Round 3, Round 4, or Round 5 are interpreted.

⸻

5. Round 3 Protocol

5.1 Goal

Round 3 evaluates whether MC Dropout uncertainty is informative.

It is only run for:
	•	G2
	•	G4

5.2 Input

The inputs are the already-produced predictions.csv files from each run directory.

5.3 Metrics

Round 3 reports:
	•	rmse
	•	mae
	•	nasa
	•	pred_std_mean
	•	pred_std_std
	•	pearson_unc_abs_err
	•	spearman_unc_abs_err
	•	ece_abs_err_vs_unc
	•	top_bottom_error_ratio
	•	coverage_1sigma
	•	coverage_2sigma

5.4 Commands

Run from:

src_v2/experiments_v2/rounds/

python round3_mcd_validity.py --group G4
python round3_mcd_validity.py --group G2

5.5 Interpretation rule

Round 3 should answer:
	•	whether uncertainty correlates with absolute error
	•	whether higher uncertainty identifies harder examples
	•	whether predictive intervals have reasonable empirical coverage

Round 3 is not used to rank groups by overall predictive quality alone. That role belongs to the main group comparison.

⸻

6. Round 4 Protocol

6.1 Goal

Round 4 analyzes error structure and risk distribution.

It is run for:
	•	G1
	•	G2
	•	G3
	•	G4

6.2 Main questions

Round 4 should answer:
	1.	Is the overall advantage of G4 coming from danger-zone performance or safe-zone performance?
	2.	Does LinEx change error direction bias?
	3.	Are overestimation and underestimation rates redistributed in a risk-consistent way?
	4.	Which RUL bins benefit most from LinEx and MCD?

6.3 Metrics

Round 4 reports:
	•	overall RMSE, MAE, NASA
	•	mean error
	•	mean absolute error
	•	overestimate rate
	•	underestimate rate
	•	severe and moderate error probabilities
	•	danger-zone metrics
	•	safe-zone metrics
	•	per-bin metrics over RUL ranges

6.4 Command

python round4_error_structure.py

6.5 Interpretation rule

Round 4 must be interpreted structurally, not only numerically.

A model may be best overall without being best in the danger zone.
That distinction must be made explicit in the report.

⸻

7. Round 5 Protocol

7.1 Goal

Round 5 performs maintenance simulation using validation full trajectories.

This is the only acceptable version of Round 5 for policy interpretation.

7.2 Critical rule

Round 5 must not use the official test split as the policy simulation source.

Reason:
	•	C-MAPSS test units are censored
	•	they do not provide full run-to-failure trajectories
	•	threshold-based maintenance simulation on censored test trajectories can create trivial or misleading zero-cost outcomes

Therefore, Round 5 uses:
	•	split_units.json
	•	train_units for normalization fitting
	•	val_units for full-trajectory policy evaluation

7.3 Inputs

For each run:
	•	config.json
	•	split_units.json
	•	best_model.pth

Then:
	1.	reconstruct the training split
	2.	reconstruct the validation split
	3.	normalize using only train_units
	4.	run mode="all" window inference on val_units
	5.	simulate threshold-triggered maintenance decisions per unit

7.4 Decision logic

For each validation unit and each threshold:
	•	compute true threshold crossing time
	•	compute predicted threshold crossing time
	•	if predicted crossing occurs before or at true crossing:
	•	preventive action
	•	if true crossing occurs first and no preventive action happened:
	•	failure event
	•	if preventive action occurs too early:
	•	early maintenance penalty
	•	compute total cost

7.5 Main outputs

Per threshold:
	•	total cost
	•	failure event rate
	•	preventive action rate
	•	unnecessary maintenance rate
	•	protected danger rate
	•	early lead

7.6 Command

python round5_maintenance.py --thresholds 5 10 15 20 25 30 35 40 --preventive-cost 1.0 --failure-cost 10.0 --early-cost-per-cycle 0.05

Optional faster sanity check:

python round5_maintenance.py --thresholds 5 10 15 20 25 30 35 40 --preventive-cost 1.0 --failure-cost 10.0 --early-cost-per-cycle 0.05 --mc-samples-override 10

7.7 Interpretation rule

Round 5 is a policy-level analysis.

It should be used to answer:
	•	which model minimizes maintenance cost
	•	which model best balances failure avoidance and unnecessary maintenance
	•	whether LinEx changes policy preference compared with MSE
	•	whether MCD helps in operational decision support

⸻

8. Re-run Policy

8.1 When to delete outputs

Delete and rerun if any of the following is true:
	•	protocol check fails
	•	group run counts are inconsistent
	•	Round 5 was previously produced from censored test trajectories
	•	core prediction fields changed
	•	windowing protocol changed
	•	validation split logic changed

8.2 Recommended clean rerun scope

If primary code changed in G2, G3, or G4, rerun those groups and all dependent rounds.
If output protocol changed, rerun all summaries and rounds.

⸻

9. Interpretation Hierarchy

Results should be interpreted in this order:
	1.	Main group comparison
	2.	Round 3 uncertainty validity
	3.	Round 4 error structure
	4.	Round 5 maintenance policy

This avoids overclaiming based on any single round.

⸻

10. Current stable narrative

The current working narrative supported by the pipeline is:
	1.	LinEx is the primary source of performance improvement.
	2.	G4 is the strongest overall model configuration.
	3.	G2 may show stronger uncertainty-error coupling than G4.
	4.	G4’s strongest global advantage may come from safe-zone risk control rather than absolute dominance in the danger zone.
	5.	Maintenance conclusions must be drawn only from validation full-trajectory policy simulation.

⸻

11. Reproducibility Notes

For every retained run, the following must be saved:
	•	full config
	•	seed
	•	train/validation split units
	•	best checkpoint
	•	metrics files
	•	predictions file

No reported result should rely on undocumented manual filtering.

⸻

12. Minimal final reporting checklist

Before using any result in the report, confirm:
	•	primary group outputs exist
	•	multi-seed summaries exist
	•	group comparison exists
	•	protocol check passes
	•	Round 3 exists for G2 and G4
	•	Round 4 exists for G1 to G4
	•	Round 5 uses validation full trajectories
	•	all compared groups in Round 5 have the intended number of runs

---



