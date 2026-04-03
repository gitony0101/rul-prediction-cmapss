# Public Results Summary

This directory contains a curated selection of experimental results intended for a high-level overview of the project. The goal is to provide clear, impactful evidence of the project's findings without the clutter of individual seed-level logs.

## Contents

- **`group_comparison.csv`**: The primary comparison of all experimental groups (G1 through G4) across key metrics like RMSE and MAE.
- **`G1_multiseed_summary.json`**, **`G2_multiseed_summary.json`**, **`G3_multiseed_summary.json`**, **`G4_multiseed_summary.json`**: Summary statistics for each major experimental configuration, capturing the mean performance across multiple random seeds.
- **`G4_multiseed_detail.csv`**: A detailed breakdown of the best-performing configuration (LinEx + MC Dropout), showing the performance variation across seeds.

## Selection Criteria

Files were selected based on their ability to:
1. Provide an immediate, high-level comparison of the different methodologies.
2. Demonstrate the statistical significance and consistency of the results through multi-seed summaries.
3. Highlight the superior performance of the final proposed configuration (G4).
