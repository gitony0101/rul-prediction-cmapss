# CS 6516 Project · Experiment Map
## 1. Purpose

This document maps the full `experiments_v2` pipeline from:

- main controlled experiments
- to analysis rounds
- to final report interpretation

It is designed to answer one question clearly:

> each script exists to answer which research question?

---

## 2. Big Picture

The experiment system is organized into three layers:

1. **Primary experiments**
2. **Analysis and protocol layer**
3. **Research rounds**

The logic is:

\[
\text{primary runs}
\rightarrow
\text{summary and consistency check}
\rightarrow
\text{uncertainty analysis}
\rightarrow
\text{error structure analysis}
\rightarrow
\text{maintenance policy analysis}
\rightarrow
\text{ablation / generalization / toy mechanism}
\]

---

## 3. Source Tree Map

```text
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
    │   ├── round5_maintenance.py
    │   ├── round6_ablation.py
    │   ├── round7_multidataset.py
    │   └── round8_toy_mechanism.py
    └── docs/
        ├── protocol.md
        ├── outputs_spec.md
        └── experiment_map.md


⸻

4. Primary Experiment Layer

4.1 G1_run.py

Role

Deterministic baseline.

Configuration
	•	loss: MSE
	•	MC Dropout: off

Research question

What is the performance of the plain baseline under the fixed protocol?

Interpretation role

Reference point for all other groups.

⸻

4.2 G2_run.py

Role

Uncertainty-only variant.

Configuration
	•	loss: MSE
	•	MC Dropout: on

Research question

If uncertainty is added without changing the loss, does performance improve?

Interpretation role

Isolates the effect of MCD under standard symmetric loss.

⸻

4.3 G3_run.py

Role

Risk-sensitive deterministic variant.

Configuration
	•	loss: LinEx
	•	MC Dropout: off

Research question

If the loss is changed to reflect asymmetric risk, does prediction improve even without uncertainty sampling?

Interpretation role

Isolates the effect of LinEx alone.

⸻

4.4 G4_run.py

Role

Full proposed configuration.

Configuration
	•	loss: LinEx
	•	MC Dropout: on

Research question

Does combining asymmetric loss and uncertainty sampling yield the strongest overall system?

Interpretation role

Main proposed model.

⸻

4.5 Controlled comparison structure

The four groups form a 2×2 design:

Group	Loss	MCD
G1	MSE	No
G2	MSE	Yes
G3	LinEx	No
G4	LinEx	Yes

This enables the following controlled contrasts:
	•	G1 vs G2: MCD effect under MSE
	•	G1 vs G3: LinEx effect without MCD
	•	G3 vs G4: MCD effect under LinEx
	•	G2 vs G4: LinEx effect under MCD

⸻

5. Multi-seed Execution Layer

5.1 run_group_multiseed.py

Role

Run one group over multiple random seeds.

Input
	•	group name
	•	seed list
	•	optional LinEx or MCD settings

Output

Multiple run directories in outputs_v2/<group_root>/

Research role

Reduces dependence on a single random seed.

Interpretation role

Transforms single-run observations into stable group-level evidence.

⸻

5.2 summarize_group_multiseed.py

Role

Aggregate all runs for a group.

Output
	•	detail CSV
	•	summary JSON
	•	protocol check JSON
	•	manifest JSON

Research role

Provides within-group mean and variance statistics.

Interpretation role

Used to answer whether a group’s apparent advantage is stable.

⸻

5.3 compare_groups.py

Role

Aggregate all group summaries into one comparison table.

Main output

outputs_v2/group_comparison.csv

Research role

Provides the primary cross-group ranking.

Interpretation role

This is the main table for:
	•	RMSE
	•	MAE
	•	NASA
	•	validation RMSE
	•	best run info
	•	uncertainty summary where applicable

⸻

6. Protocol and Consistency Layer

6.1 analysis/protocol_check.py

Role

Check whether outputs are complete and protocol-consistent.

What it verifies
	•	required files exist
	•	required JSON keys exist
	•	required CSV columns exist
	•	run directories are structurally valid

Research role

Prevents accidental interpretation of incomplete or stale outputs.

Interpretation rule

No analysis round should be treated as final unless protocol check passes.

⸻

7. Research Rounds Overview

The rounds are not independent side experiments.
They are explanatory layers built on top of the main 2×2 design.

⸻

8. Round 3

8.1 Script

round3_mcd_validity.py

8.2 Applies to
	•	G2
	•	G4

8.3 Core research question

[
\text{Does MC Dropout produce uncertainty estimates that actually track prediction difficulty?}
]

8.4 Inputs
	•	predictions.csv from MCD groups

8.5 Outputs
	•	uncertainty-error correlation
	•	coverage
	•	calibration-style statistics
	•	top-bottom error separation

8.6 Interpretation role

Round 3 does not answer which group is best overall.
It answers whether the uncertainty signal is meaningful.

Typical conclusions
	•	G4 can be better in point prediction
	•	G2 can still show stronger uncertainty-error coupling

This distinction is important and should not be collapsed into one number.

⸻

9. Round 4

9.1 Script

round4_error_structure.py

9.2 Applies to
	•	G1
	•	G2
	•	G3
	•	G4

9.3 Core research question

[
\text{Where does each model win or lose across the RUL range and error directions?}
]

9.4 Inputs
	•	predictions.csv

9.5 Main outputs
	•	overall metrics
	•	error-direction statistics
	•	danger zone metrics
	•	safe zone metrics
	•	per-bin metrics

9.6 Interpretation role

Round 4 explains why overall rankings happen.

Examples of what it can reveal
	•	a model can be best overall but not best in the danger zone
	•	LinEx can shift bias from overestimation to safer underestimation
	•	gains can come mainly from safe-zone risk reduction

Round 4 is the main bridge between predictive performance and research explanation.

⸻

10. Round 5

10.1 Script

round5_maintenance.py

10.2 Applies to
	•	G1
	•	G2
	•	G3
	•	G4

10.3 Core research question

[
\text{How do prediction differences translate into operational maintenance decisions?}
]

10.4 Critical version note

The valid Round 5 version is:

[
\text{validation full-trajectory maintenance simulation}
]

It must not rely on censored official test trajectories for full policy interpretation.

10.5 Inputs
	•	config.json
	•	split_units.json
	•	best_model.pth

10.6 Simulation logic

For each validation unit and each threshold:
	1.	reconstruct the full validation trajectory
	2.	find true threshold crossing
	3.	find predicted threshold crossing
	4.	determine:
	•	preventive action
	•	failure event
	•	unnecessary maintenance
	•	early lead
	5.	compute total cost

10.7 Interpretation role

Round 5 is the policy layer.

It answers:
	•	which group minimizes cost
	•	which group avoids failures best
	•	which group over-maintains
	•	whether LinEx changes decision preference
	•	whether MCD helps at the policy level

⸻

11. Round 6

11.1 Script

round6_ablation.py

11.2 Intended role

Minimal architectural ablation.

11.3 Core research question

[
\text{Is the CNN front-end actually necessary?}
]

11.4 Recommended design

Keep Round 6 small and controlled.

Recommended comparisons:
	•	BiLSTM + MSE
	•	CNN+BiLSTM + MSE
	•	BiLSTM + LinEx
	•	CNN+BiLSTM + LinEx

11.5 Interpretation role

Helps decide whether CNN should remain central in the final narrative.

This round should not explode into many uncontrolled variants.

⸻

12. Round 7

12.1 Script

round7_multidataset.py

12.2 Intended role

Cross-dataset generalization check.

12.3 Core research question

[
\text{Do the conclusions hold beyond FD001?}
]

12.4 Recommended dataset order
	•	FD001
	•	FD003
	•	FD002
	•	FD004

12.5 Recommended comparison scope

Do not necessarily rerun all four groups.

A minimal useful comparison is:
	•	G1 vs G4

A more focused research comparison is:
	•	G3 vs G4

12.6 Interpretation role

Determines whether the current conclusions are robust across operating conditions.

⸻

13. Round 8

13.1 Script

round8_toy_mechanism.py

13.2 Intended role

Synthetic mechanism validation.

13.3 Core research question

[
\text{Under controlled degradation and asymmetric risk, does LinEx behave as expected?}
]

13.4 Purpose

This is not just another benchmark.
It is the mechanism-explanation round.

13.5 What it should test
	•	controlled degradation signal
	•	controllable noise
	•	asymmetric cost of overestimation
	•	uncertainty sensitivity under harder regimes

13.6 Interpretation role

Supports the project at the conceptual level.

This round is especially important because it aligns with the course emphasis on synthetic validation before noisy real-world claims.

⸻

14. Execution Order Map

The recommended execution order is:

1. G1/G2/G3/G4 primary runs
2. summarize_group_multiseed.py
3. compare_groups.py
4. protocol_check.py
5. round3_mcd_validity.py
6. round4_error_structure.py
7. round5_maintenance.py
8. round6_ablation.py
9. round7_multidataset.py
10. round8_toy_mechanism.py


⸻

15. Interpretation Order Map

The recommended interpretation order is:

main group comparison
→ uncertainty validity
→ error structure
→ maintenance policy
→ architecture ablation
→ dataset generalization
→ toy mechanism

This order matters.
It prevents later rounds from being used to overrule the primary controlled evidence.

⸻

16. What each layer contributes

Layer	Main answer
Primary groups	which model predicts best overall
Round 3	whether uncertainty is informative
Round 4	where and how gains happen
Round 5	whether gains translate into policy benefit
Round 6	whether architecture choices are necessary
Round 7	whether findings generalize across datasets
Round 8	whether the mechanism is conceptually justified


⸻

17. Current project narrative supported by the map

At the current stage, the most coherent project story is:
	1.	Start from the 2×2 controlled design.
	2.	Show that LinEx is the primary gain source.
	3.	Show that G4 is the strongest overall configuration.
	4.	Use Round 3 to show that uncertainty usefulness and point prediction quality are not identical questions.
	5.	Use Round 4 to show that G4’s advantage may come mainly from safer global error structure, especially outside the strict danger zone.
	6.	Use Round 5 to determine whether those predictive differences actually improve maintenance policy.
	7.	Use later rounds only as focused supporting evidence, not as a replacement for the core controlled design.

⸻

18. Minimal report mapping

If the final write-up is kept concise, the mapping can be:
	•	Main Results
	•	group_comparison.csv
	•	Uncertainty Analysis
	•	round3_mcd_validity.py
	•	Error Structure Analysis
	•	round4_error_structure.py
	•	Maintenance Decision Analysis
	•	round5_maintenance.py
	•	Optional Extensions
	•	round6_ablation.py
	•	round7_multidataset.py
	•	round8_toy_mechanism.py

⸻

19. Final practical guidance

When deciding whether a script is essential, ask:

if this file disappeared, which research question would become unanswered?

If the answer is “none,” that script is probably not central.
If the answer is “we could no longer justify a core claim,” that script is part of the main evidence chain.

For the current project, the strongest evidence chain is:

[
\text{G1/G2/G3/G4}
\rightarrow
\text{Round 3}
\rightarrow
\text{Round 4}
\rightarrow
\text{Round 5}
]




