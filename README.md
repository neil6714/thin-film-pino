# Physics-Informed Neural Operator for Atomic Layer Deposition

A research framework for simulating Atomic Layer Deposition (ALD) and developing neural-operator surrogates for thin-film deposition.

## Overview

The project follows:

**Physics Simulator → Dataset Generation → ML Baselines → FNO → PINO → Transient PDE-PINO**

The goal is to learn spatial and temporal ALD dynamics while maintaining consistency with the underlying deposition physics.

## Physics Model

The simulator models:

- Precursor diffusion
- Adsorption/desorption
- Surface reaction
- Film growth
- Cyclic ALD processing

Primary fields:

- Precursor concentration `C`
- Surface coverage `θ`
- Film thickness `h`

The reference implementation is:

```text
physics/simulator.py
```

## Project Stages

### Stage 1 — Physics Simulator

Validated 2D cyclic ALD simulator with pulse, purge, reaction, and film-growth dynamics.

Validated through:

- Cycle stability
- Pulse-time sweeps
- Kinetic sweeps
- Conformality
- GPC behavior

### Stage 2 — Dataset Generation

Generated a 2,000-sample GPU-accelerated ALD dataset on a `120 × 160` grid.

Outputs include:

- Concentration
- Surface coverage
- Film thickness
- GPC
- Conformality
- Thickness metrics

GPU results were validated against the CPU reference solver with maximum relative error of approximately `1.7e-5`.

### Stage 3 — Static ML

Models evaluated:

```text
MLP → FNO → Physics-Constrained FNO
```

The MLP achieved mean `R² ≈ 0.99946` on scalar deposition metrics.

The static FNO achieved:

```text
Coverage R²  ≈ 0.9968
Thickness R² ≈ 0.9974
```

Controlled ablations showed that simple static physics regularization did not provide a meaningful improvement over the baseline FNO.

### Stage 4 — Transient Modeling

The project is moving from final-state prediction to:

```text
(parameters, time) → C(x,y,t), θ(x,y,t), h(x,y,t)
```

Current components:

```text
data/generate_transient_dataset.py
data/generate_transient_dataset_fast.py
physics/transient_cuda.py
physics/batched_transient_cuda.py
training/train_temporal_operator.py
training/train_temporal_pino.py
```

Stage 4C uses a time-conditioned spatial FNO.

Stage 4D introduces PDE residuals for transient ALD physics.

## Current Focus

The main remaining work is to:

1. Correct Stage 4 evaluation and metric aggregation.
2. Validate transient CUDA simulations against the reference solver.
3. Make PDE residuals phase-aware.
4. Match spatial derivatives to the simulator geometry.
5. Establish a shared held-out test evaluation.
6. Compare Temporal FNO vs PDE-PINO on accuracy, physics consistency, generalization, and inference speed.

## Repository Structure

```text
thin-film-pino/
├── data/
├── physics/
├── training/
├── evaluation/
├── tests/
├── results/
├── requirements.txt
└── README.md
```

Large datasets and model checkpoints are stored externally and are not committed to GitHub.

## Reproducibility

The project preserves intermediate and failed approaches rather than deleting them. Historical results are retained for comparison and debugging.

Final experiments should report:

- MAE
- RMSE
- R²
- PDE residuals
- Physical constraints
- Generalization performance
- Simulation vs neural-operator inference time

## Future Work

- Fully validated transient PDE-PINO
- Improved geometry handling
- Unseen-parameter generalization
- Inverse ALD process design
- Faster surrogate-based optimization
- More realistic ALD chemistry and geometries
