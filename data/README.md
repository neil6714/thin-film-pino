# Data Directory

This directory stores generated simulation datasets (`.npz` files) and associated trajectory metadata.

Datasets generated locally are excluded from version control by `.gitignore`.

To generate a trajectory dataset, use the generation scripts in `scripts/`:

```bash
# Generate accelerated batch trajectory dataset
python scripts/generate_transient_dataset_fast.py --num-samples 64 --batch-size 16 --output data/transient_ald_dataset_fast.npz

# Partition and normalize without data leakage
python scripts/prepare_temporal_dataset.py --input data/transient_ald_dataset_fast.npz --output data/transient_ald_prepared.npz
```

Published reference datasets can be hosted as GitHub release assets or Zenodo records.
