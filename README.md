# Physics-Informed Neural Operator for Transient Atomic Layer Deposition

A continuum modeling and scientific machine learning framework for simulating transient atomic layer deposition (ALD) in 2D trench geometries and learning neural operator surrogates with physics-informed residual penalties.

---

## 1. Overview and Theoretical Foundations

Atomic Layer Deposition (ALD) is a vapor-phase thin-film fabrication technique based on sequential, self-limiting surface chemical reactions. In trench structures, precursor transport is modeled via Fickian diffusion coupled to nonlinear Langmuir adsorption and desorption kinetics.

This framework includes:
1. **Continuum ALD Simulators**: 2D explicit forward-time central-space (FTCS) finite-difference solvers governed by diffusive von Neumann and reaction kinetic stability limits, with NumPy CPU and vectorized PyTorch GPU/CPU implementations sharing identical geometry and stencil code.
2. **Temporal Fourier Neural Operators (TFNO)**: Deep operator networks mapping physical process parameters and time coordinates to spatial multi-field states $(\mathbf{p}, t) \mapsto (C, \theta, h)(x, y, t)$.
3. **Physics-Informed Neural Operators (PINO)**: Operator learning regularized by transient PDE residual penalties evaluated at discrete trajectory snapshots.

For the mathematical derivation, scaling, and assumptions, see [Formulation](docs/formulation.md).
For documented simulator and surrogate limitations, see [Known Limitations](docs/known_limitations.md).

---

## 2. Mathematical and Physical Formulation

### 2.1 Governing Equations

The continuum transport-reaction system in a 2D domain $\Omega = [0, L_x] \times [0, L_y]$ is formulated in nondimensional variables:

**1. Precursor Vapor Transport (Gas Domain $\Omega_{\text{gas}}$)**:

$$\frac{\partial C}{\partial t} = D \nabla^2 C - R_{\text{ads}}(x, y, t)$$

where $C(x, y, t)$ is nondimensional precursor concentration ($C/C_0$), $D$ is the constant diffusion coefficient, and $R_{\text{ads}}$ is the surface sink rate localized to boundary cells.

**2. Surface Coverage Dynamics (Surface Boundary $\Gamma_{\text{surf}}$)**:

$$\frac{\partial \theta}{\partial t} = R_{\text{ads}} - R_{\text{des}} - R_{\text{rxn}}$$

governed by Langmuir self-limiting kinetics:

$$R_{\text{ads}} = k_{\text{ads}} C (1 - \theta), \quad R_{\text{des}} = k_{\text{des}} \theta, \quad R_{\text{rxn}} = k_{\text{rxn}} \theta$$

where $\theta(x, y, t) \in [0.0, 1.0]$ represents fractional coverage of chemisorbed precursor, $k_{\text{ads}}$ is the first-order adsorption rate constant, $k_{\text{des}}$ is the thermal desorption rate constant, and $k_{\text{rxn}}$ is the co-reactant reaction rate constant.

**3. Film Thickness Evolution ($\Gamma_{\text{surf}}$)**:

$$\frac{\partial h}{\partial t} = k_{\text{growth}} R_{\text{rxn}} = k_{\text{growth}} k_{\text{rxn}} \theta$$

where $h(x, y, t)$ is accumulated film thickness and $k_{\text{growth}}$ is the thickness conversion coefficient per unit converted coverage.

### 2.2 Boundary and Initial Conditions

- **Reactor Inlet Boundary ($y = 0$)**:
  - Dose Phase: Dirichlet boundary condition $C(x, 0, t) = C_0 = 1.0$.
  - Purge and Reaction Phases: Zero-concentration boundary condition $C(x, 0, t) = 0.0$.
- **Solid Wall and Trench Surfaces**:
  - Homogeneous Neumann zero-flux condition $\nabla C \cdot \mathbf{n} = 0$.
- **Initial State ($t = 0$)**:
  $$C(x, y, 0) = 0, \quad \theta(x, y, 0) = 0, \quad h(x, y, 0) = 0$$

### 2.3 Geometry and Regions

The domain contains a rectangular trench feature:
- Default dimensions: width $w = 0.25$, depth $d = 0.45$, corresponding to aspect ratio $AR = 0.45 / 0.25 = 1.8$.
- High-aspect-ratio preset: `SimulationConfig.high_aspect_ratio_preset(aspect_ratio=10.0)` configures a narrow centered trench ($w = 0.045, d = 0.45$).
- Explicit surface regions:
  - **Field**: Wafer top surface outside the trench mouth.
  - **Sidewall**: Vertical trench walls.
  - **Floor**: Horizontal bottom of the trench.
- **Step Coverage**: Defined as mean thickness on the floor divided by mean thickness on the field:
  $$\text{Step Coverage} = \frac{\bar{h}_{\text{floor}}}{\bar{h}_{\text{field}}}$$

### 2.4 Modelling Assumptions

1. **Operator Splitting**: The four ALD phases (dose, purge, reaction, post-purge) execute sequentially.
2. **Perfect Purge**: Gas precursor concentration is set to zero ($C \equiv 0$) at the end of each purge stage. Because the nominal purge duration ($1.0$) is much shorter than the diffusion timescale ($L^2/D \approx 25\text{--}200$), clearance is an operational model assumption rather than a physical transport result.
3. **Reaction Stage Isolation**: Chemisorption occurs only during precursor dose. Co-reactant reaction occurs only during the reaction phase. Desorption is inactive during purge.
4. **Mass Non-Conservation**: Mass is not conserved. The precursor sink in $C$ is not coupled to absolute surface site density, and desorption does not return precursor vapor back into the gas domain.
5. **Incomplete Reaction Carryover**: For $k_{\text{rxn}} \in [0.1, 0.8]$ and $t_{\text{rxn}} = 1.0$, at most $55\%$ of chemisorbed precursor reacts per cycle, and remaining unreacted coverage carries over into subsequent cycles.
6. **Bounds Enforcement**: Physical bounds $C \ge 0$ and $0 \le \theta \le 1$ are enforced by numerical clamping and monitored by activation counters (`clamp_stats`). Default parameters produce zero clamp activations.

### 2.5 Stability Criteria

The explicit 2D central-difference scheme enforces diffusive von Neumann and kinetic stability limits on time step $\Delta t$:

$$\Delta t \le \min\left(0.20 \frac{\min(\Delta x, \Delta y)^2}{D_{\max}}, \frac{0.20}{\max(k_{\text{ads}}, k_{\text{des}}, k_{\text{rxn}})}\right)$$

User-supplied time steps that exceed the von Neumann limit $\min(\Delta x, \Delta y)^2 / (4 D)$ or kinetic limit $1 / \max(k)$ are rejected with a ValueError.

---

## 3. Neural Operator Architecture

```
Physical Parameters [p] (8-dim) ──┐
Time Coordinate t (1-dim)       ──┼──► Spatial Broadcast & Concat ──► Lifting Conv2d (11 -> width)
Normalized Grid (x, y) (2-dim)  ──┘                                           │
                                                                              ▼
                                                                ┌─────────────────────────┐
                                                                │ 4x FourierBlock         │
                                                                │  - SpectralConv2d       │
                                                                │  - Pointwise Conv2d     │
                                                                │  - GELU Activation      │
                                                                └─────────────────────────┘
                                                                              │
                                                                              ▼
                                                                   Projection Conv2d
                                                                     (width -> 64 -> 3)
                                                                              │
                                                                              ▼
                                                                 Predicted Output Tensor
                                                                 [C, θ, h](x, y, t)
```

### 3.1 Spectral Convolution Layer

The 2D Fourier layer parameterizes integral kernel operators in frequency space:

$$(\mathcal{K}(v))(x) = \mathcal{F}^{-1}\left(R_\phi \cdot (\mathcal{F} v)\right)(x)$$

where $\mathcal{F}$ and $\mathcal{F}^{-1}$ denote the 2D Real Fast Fourier Transform and its inverse, with learnable complex frequency weight tensor truncated at cutoff mode $k_{\max} = 12$.

### 3.2 Physics-Informed Residual Formulation

The optimization objective combines masked data loss with phase-aware continuum residuals:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{physics}} \mathcal{L}_{\text{PDE}}$$

where the PDE residual penalty $\mathcal{L}_{\text{PDE}}$ evaluates:

**1. Precursor Transport Residual** (over active gas cells $\Omega_{\text{gas}}$):

$$\mathcal{R}_C = \frac{\partial C}{\partial t} - D \nabla^2 C + \delta_{\text{pulse}} R_{\text{ads}}$$

**2. Surface Coverage Residual** (over surface cells $\Gamma_{\text{surf}}$):

$$\mathcal{R}_\theta = \frac{\partial \theta}{\partial t} - \delta_{\text{pulse}} (R_{\text{ads}} - R_{\text{des}}) + \delta_{\text{rxn}} R_{\text{rxn}}$$

**3. Film Growth Residual** (over surface cells $\Gamma_{\text{surf}}$):

$$\mathcal{R}_h = \frac{\partial h}{\partial t} - \delta_{\text{rxn}} k_{\text{growth}} R_{\text{rxn}}$$

where $\delta_{\text{pulse}}, \delta_{\text{rxn}} \in \{0, 1\}$ indicate precursor dose and co-reactant reaction stages, respectively.

---

## 4. Parameter Ranges and Physical Specifications

All quantities are nondimensionalized as documented in [Formulation](docs/formulation.md).

| Parameter | Symbol | Range / Nominal Value | Description |
| :--- | :---: | :--- | :--- |
| Precursor Diffusivity | $D$ | $\mathcal{U}(1.0 \times 10^{-3}, 8.0 \times 10^{-3})$ | Nondimensional diffusion coefficient |
| Adsorption Rate Constant | $k_{\text{ads}}$ | $\mathcal{U}(0.5, 1.5)$ | First-order surface chemisorption rate |
| Desorption Rate Constant | $k_{\text{des}}$ | $\mathcal{U}(0.02, 0.10)$ | First-order thermal desorption rate |
| Co-reactant Reaction Rate | $k_{\text{rxn}}$ | $\mathcal{U}(0.1, 0.8)$ | Surface reaction and conversion rate |
| Growth Conversion Factor | $k_{\text{growth}}$ | $0.02$ (fixed nominal) | Thickness per converted coverage |
| Precursor Dose Time | $t_{\text{pulse}}$ | Bimodal: $85\%\; \mathcal{U}(10.0, 60.0)$, $15\%\; \mathcal{U}(60.0, 100.0)$ | Exposure stage duration |
| Purge Time | $t_{\text{purge}}$ | $1.0$ (fixed nominal) | Inert gas purge duration |
| Co-reactant Exposure Time | $t_{\text{rxn}}$ | $1.0$ (fixed nominal) | Co-reactant reaction duration |
| Spatial Domain Resolution | $N_x \times N_y$ | $160 \times 120$ (reference) / $80 \times 60$ (accelerated) | Grid node count |

---

## 5. Repository Structure

```
thin-film-pino/
├── configs/
│   └── default.yaml                # Default model hyperparameters
├── docs/
│   ├── figures/                    # Curated figures
│   ├── archive/                    # Archived historical artifacts
│   ├── formulation.md              # Mathematical formulation and scaling
│   └── known_limitations.md        # Documented simulator and surrogate limitations
├── scripts/
│   ├── cycle_validation.py         # Multi-cycle GPC stabilization and linearity validation
│   ├── kinetic_sweep.py            # 2D diffusivity vs reaction rate sensitivity sweeps
│   ├── parameter_sweep.py          # Parameter sensitivity contour generator
│   ├── pulse_saturation.py         # Self-limiting Langmuir saturation verification
│   ├── pulse_sweep.py              # Pulse duration conformality mapping
│   ├── generate_data.py            # Steady-state deposition profile dataset generator
│   ├── generate_final_dataset.py   # Verified multi-parameter steady-state dataset generator
│   ├── generate_transient_dataset.py     # Transient trajectory dataset generator
│   ├── generate_transient_dataset_fast.py # Accelerated batched transient trajectory generator
│   ├── prepare_temporal_dataset.py # Splitting and normalization pipeline without leakage
│   ├── validate_transient_dataset.py     # Dataset physical invariant and split audit suite
│   ├── validate_cuda_reference.py  # Reference CPU vs. accelerated PyTorch parity verifier
│   ├── inspect_dataset.py          # Summary statistics, distributions, and spatial plots
│   ├── compare_models.py           # Comparative benchmarking across models
│   └── plot_losses.py              # Unified loss convergence visualization
├── src/
│   └── thin_film_pino/
│       ├── data/                   # Data definitions, parameter distributions, splitting
│       ├── evaluation/             # Checkpoint evaluation on held-out test data
│       ├── models/                 # Fourier Neural Operator (FNO) architectures
│       ├── physics/                # Unified continuum ALD simulators and geometry
│       └── training/               # Unified training pipeline for FNO and PINO
├── tests/
│   ├── test_physics.py             # Physics, geometry masks, clamping, and solver parity
│   └── test_temporal_operator_loss.py # Masked reduction unit test
├── pyproject.toml                  # Packaging specification and dependencies
├── requirements.txt                # Pinned dependency requirements
└── README.md                       # Technical documentation and workflow guide
```

---

## 6. Reproducible Workflow and CLI Guide

### 6.1 Environment Setup

Install the package in editable mode with development dependencies:

```bash
pip install -e .[dev]
```

### 6.2 Simulator Parity Verification

Verify exact trajectory parity between the reference CPU solver and the accelerated PyTorch simulator on matched grids:

```bash
python scripts/validate_cuda_reference.py --output results/validation --num-cycles 5 --nx 80 --ny 60 --dtype float64
```

### 6.3 Transient Trajectory Generation and Preparation

Generate an accelerated ensemble of ALD trajectories and partition into train/val/test splits:

```bash
# 1. Generate accelerated trajectory dataset
python scripts/generate_transient_dataset_fast.py --num-samples 64 --batch-size 16 --output data/transient_ald_dataset_fast.npz

# 2. Partition into train (70%), validation (15%), and test (15%) splits
python scripts/prepare_temporal_dataset.py --input data/transient_ald_dataset_fast.npz --output data/transient_ald_prepared.npz

# 3. Audit dataset integrity and physical bounds
python scripts/validate_transient_dataset.py --input data/transient_ald_prepared.npz
```

### 6.4 Neural Operator Training

Train baseline FNO and physics-informed models:

```bash
# Train baseline Temporal Fourier Neural Operator (TFNO)
python -m thin_film_pino.training.train --model fno --input data/transient_ald_prepared.npz --epochs 50 --batch-size 4 --output-dir runs/fno

# Train Physics-Informed Neural Operator (PINO)
python -m thin_film_pino.training.train --model pino --input data/transient_ald_prepared.npz --epochs 50 --batch-size 4 --physics-weight 1.0e-4 --output-dir runs/pino
```

### 6.5 Evaluation and Comparative Benchmarking

Evaluate checkpoints on the shared held-out test split:

```bash
# Evaluate FNO checkpoint
python -m thin_film_pino.evaluation.evaluate --input data/transient_ald_prepared.npz --checkpoint runs/fno/best_model.pt --output-dir runs/evaluation

# Cross-compare models on identical test trajectories
python scripts/compare_models.py --input data/transient_ald_prepared.npz --operator runs/fno/best_model.pt --pino runs/pino/best_model.pt --output runs/comparison

# Plot unified convergence curves
python scripts/plot_losses.py --root runs --output runs/loss_trajectories.png
```

### 6.6 Unit Testing

Execute automated unit tests:

```bash
pytest tests/
```

---

## 7. Numerical Standards and Verification

- **Bounds Enforcement**: Physical bounds $\theta(x, y, t) \in [0.0, 1.0]$ and $C(x, y, t) \ge 0.0$ are enforced by numerical clamping and monitored via `clamp_stats` activation counters. Default simulation configurations produce zero clamp activations.
- **Solver Parity**: Reference NumPy and PyTorch accelerated simulators share unified geometry masks and discrete stencil operators, exhibiting exact numerical parity on matched discretizations.
- **Split Integrity**: Trajectory splits are disjoint and normalization statistics are computed strictly on the training partition without cross-split leakage.
