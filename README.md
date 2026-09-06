# thin-film-pino

Physics-informed neural operators for transient Atomic Layer Deposition (ALD).

## Installation

Create a virtual environment and install the scientific dependencies:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA training, install the PyTorch wheel recommended for the target CUDA
version from the [official PyTorch selector](https://pytorch.org/get-started/locally/),
then verify with `python -c "import torch; print(torch.cuda.is_available())"`.
CPU execution is supported for the reference simulator and static workflows;
the fast transient and temporal training scripts require CUDA for practical runtimes.

## Workflow

1. Generate or inspect the validated static ALD data with the scripts in `data/`.
2. Generate transient trajectories with `python -m data.generate_transient_dataset_fast`.
3. Prepare normalized, trajectory-level splits with `python -m data.prepare_temporal_dataset`.
4. Validate the prepared data with `python -m data.validate_transient_dataset`.
5. Train the supervised temporal operator with `training.train_temporal_operator`.
6. Train the experimental PDE-informed temporal PINO with `training.train_temporal_pino`.
7. Compare checkpoints with `python -m results.compare_all_models` and plot losses with
   `python -m results.plot_all_losses`.

All commands accept explicit input/output paths; no environment-specific paths are required.

## Data and checkpoints

Generated `.npz` datasets and `.pt`/`.pth` checkpoints are intentionally excluded
from Git because they can be hundreds of megabytes. Keep them in Google Drive or
another artifact store and record the generation command, grid, sample count, seed,
and commit SHA alongside any reported metrics. JSON, CSV, Markdown, and PNG reports
remain suitable for version control.

## Scientific status

Stage 1–3 results are historical validated baselines with documented caveats.
Stage 4 transient/PINO results remain experimental until the reference-vs-CUDA
physics check, phase-aware residual audit, and common held-out evaluation pass.
