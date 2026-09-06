# Thin-Film PINO Repository Audit

## Scope

Audit of the tracked source, documentation, metrics, figures, and reproducibility metadata on 2026-09-06. Large transient datasets and checkpoints are not present in the working tree and were not fabricated in this report.

## Structure

- `physics/`: validated simulator plus CUDA and batched transient implementations.
- `data/`: static dataset generation/inspection/validation, pulse and kinetic sweeps, transient generation, and temporal preparation.
- `training/`: static training scripts are represented by historical results; current source contains temporal operator and temporal PINO trainers.
- `results/`: figures and JSON/CSV metrics for Stages 1–3 and Stage 4C; comparison/plot helper scripts are present.
- `README.md`: currently only a project title; workflow and installation documentation are missing.

## Stage status

| Stage | Status | Evidence | Trust level |
|---|---|---|---|
| 1 simulator | COMPLETE WITH CAVEATS | `physics/simulator.py`, validation figures | validated baseline |
| 2 parametric data | COMPLETE WITH CAVEATS | generation scripts and audit figures | dataset files absent here |
| 3B MLP | COMPLETE | `results/stage3b_mlp/metrics.json` | reported scalar metrics |
| 3C FNO | COMPLETE WITH CAVEATS | metrics and training figure | concentration R2 is numerically invalid/degenerate |
| 3D physics model | COMPLETE WITH CAVEATS | metrics JSON | physics constraints are not full PDE residuals |
| 3E ablation | COMPLETE WITH CAVEATS | JSON/CSV and figures | static-only comparison |
| 4A transient data | SOURCE PRESENT, DATA ABSENT | transient generators/preparation code | must validate generated artifact |
| 4C temporal operator | NEEDS WORK | trainer and metrics | reported loss is anomalously large and requires corrected evaluation |
| 4D temporal PINO | NEEDS WORK | trainer source only | no trustworthy test metrics; phase-aware residual audit pending |

## Quantitative observations

- Stage 3B reports overall MAE `0.0012547` and mean R2 `0.99946`.
- Stage 3C reports test loss `0.0018665`; its concentration R2 is `-2.92e11`, indicating a near-zero-variance or metric-conditioning problem.
- Stage 3D reports validation loss `1.043e-4`, but its constraints are bounds/geometry/smoothness rather than the full ALD PDE.
- Stage 4C reports test loss `141.23` and normalized MAE `113.99`, not directly comparable to Stage 3 and likely affected by pipeline/loss-scaling issues.
- No Stage 4D held-out metrics or reference-vs-CUDA validation report is tracked.

## Reproducibility and missing artifacts

- `requirements.txt` lists NumPy, SciPy, Matplotlib, pandas, PyYAML, and PyTorch; CUDA/PyTorch installation guidance is absent.
- `README.md` has no setup or execution instructions.
- No tracked `.gitignore` rules exclude large `.npz` datasets or `.pt` checkpoints; this should be corrected before committing generated artifacts.
- Current results do not include all source scripts used for historical Stage 3 runs, so some results cannot be reproduced from this checkout alone.
- Transient dataset shapes, normalization, split integrity, and physical invariants need a machine-readable validation report generated from the actual Colab dataset.

## Scientific limitations and bugs to address

1. Static and transient metrics use different targets and normalization; direct loss comparisons are invalid.
2. Stage 4C must be re-evaluated with denominators covering every valid batch/time/channel/spatial element and physical-unit metrics.
3. Stage 4D must avoid central differences across ALD phase boundaries and must mask residuals by active phase and gas/surface geometry.
4. The fast CUDA simulator has not yet been quantitatively compared with the validated reference simulator at matched parameters.
5. PINO accuracy and PDE residuals must be reported on the same held-out trajectories as the supervised temporal baseline.

## Recommended order

1. Add requirements/README and large-artifact ignore rules.
2. Generate and validate the transient dataset; save audit JSON/Markdown.
3. Run controlled reference-vs-CUDA comparison.
4. Correct and re-evaluate Stage 4C metrics without overwriting historical results.
5. Make Stage 4D residuals phase-aware, then produce held-out physical metrics and trajectory plots.
6. Publish a single comparison table only after all models use documented, compatible evaluation protocols.
