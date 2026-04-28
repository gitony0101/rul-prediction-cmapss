# Public Repository Cleanup Plan

This plan outlines the steps to prepare the repository for public presentation while preserving internal diagnostics.

### 1. Retention (MUST KEEP)
- `src/`: Core logic.
- `scripts/` & `configs/`: Reproducibility assets.
- `results/canonical_summary/`: Verified evidence.
- `docs/` & `README.md`: Project narrative.

### 2. Removal (SAFE TO REMOVE)
- `.archive/`: Old backup directories.
- `.working_docs/`: Temporary cleanup plans.
- `outputs/`: Large raw training artifacts (should be in `.gitignore`).

### 3. Masking
- Ensure `CMAPSSData/` is excluded if redistribution is not permitted by NASA (check license).
- Remove any machine-specific paths in configs or logs.

### 4. Final Review
- Verify all figures in `docs/figures/` have a matching entry in `figure_manifest.md`.
- Ensure `linkedin_resume_summary.md` is up to date.
