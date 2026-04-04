# Public Results Summary

This directory contains curated, employer-facing evidence from the project in two parallel surfaces:

- **Curated multiseed results**: higher-rigor summary artifacts used to compare G1 through G4 across repeated runs.
- **Single-seed local verification summaries**: compact records from successful local end-to-end executions used to verify runnable baseline behavior.

The multiseed artifacts remain the stronger evidence surface for comparative performance. The single-seed artifacts are included for execution verification and repository credibility, not as a replacement for multiseed rigor.

`outputs/` should be treated as the raw generated-output archive for local runs. `public_results/` is the curated evidence surface intended for review.

## Contents

### Multiseed Curated Results

- **`group_comparison.csv`**: Primary comparison of all experimental groups across key metrics.
- **`G1_multiseed_summary.json`**, **`G2_multiseed_summary.json`**, **`G3_multiseed_summary.json`**, **`G4_multiseed_summary.json`**: Summary statistics for each major configuration across multiple random seeds.
- **`G4_multiseed_detail.csv`**: Detailed breakdown of the best-performing configuration across seeds.

### Single-Seed Local Verification Summaries

- **`single_seed_group_comparison.csv`**: Compact single-seed comparison across G1 through G4 from verified local end-to-end runs.
- **`single_seed_group_summary.json`**: Machine-readable per-group summary of the same single-seed verification runs.
- **`single_seed_uncertainty_summary.csv`**: Compact uncertainty summary for the MC Dropout configurations (G2 and G4).

## Interpretation

Use the multiseed artifacts when evaluating comparative performance claims. Use the single-seed artifacts when evaluating local runnability and end-to-end execution integrity.
