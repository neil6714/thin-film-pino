# Data Directory

This directory stores generated simulation datasets (`.npz` files) and associated trajectory metadata.

Datasets generated locally are excluded from version control by `.gitignore`.

To generate simulation datasets, use `scripts/generate_data.py`:

```bash
# Generate trajectory dataset
python scripts/generate_data.py --mode trajectory --num-samples 64 --batch-size 16 --output data/transient_ald_dataset.npz

# Or generate final-state dataset after N cycles
python scripts/generate_data.py --mode final_state --num-samples 64 --batch-size 16 --output data/final_state_dataset.npz

# Stratify and normalize without data leakage
python scripts/prepare_temporal_dataset.py --input data/transient_ald_dataset.npz --output data/transient_ald_prepared.npz
```

Published reference datasets can be hosted as GitHub release assets or Zenodo records.
