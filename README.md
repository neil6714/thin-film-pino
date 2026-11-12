# Physics-Informed Neural Operator for Transient Atomic Layer Deposition

A continuum modeling and scientific machine learning framework for simulating atomic layer deposition (ALD) in 2D trench geometries and learning neural operator surrogates with physics-informed residual penalties.

---

## 1. Overview & Theoretical Foundations

Atomic Layer Deposition (ALD) is a vapor-phase thin-film fabrication technique based on sequential, self-limiting surface chemical reactions. In trench structures, precursor transport is modeled via Fickian diffusion coupled to nonlinear Langmuir adsorption and desorption kinetics.

This framework includes:
1. **Continuum ALD Simulators**: 2D explicit forward-time central-space (FTCS) finite-difference solvers governed by the diffusive von Neumann stability limit, with NumPy CPU and vectorized PyTorch GPU/CPU implementations.
2. **Temporal Fourier Neural Operators (TFNO)**: Deep operator networks mapping physical process parameters and time coordinates to spatial multi-field states $(\mathbf{p}, t) \mapsto (C, \theta, h)(x, y, t)$.
3. **Physics-Informed Neural Operators (PINO)**: Operator learning regularized by transient PDE residual penalties evaluated at discrete trajectory snapshots.

For physical and numerical limitations of the model, see [Known Limitations](docs/known_limitations.md).

---

## 2. Mathematical & Physical Formulation

### 2.1 Governing Equations

The continuum transport-reaction system in a 2D domain $\Omega = [0, L_x] \times [0, L_y]$ is formulated as:

**1. Precursor Vapor Transport (Gas Domain $\Omega_{\text{gas}}$)**:

$$\frac{\partial C}{\partial t} = D \nabla^2 C - R_{\text{ads}}(x, y, t)$$

where $C(x, y, t)$ is precursor molar concentration ($\text{mol}/\text{m}^3$), $D$ is the constant diffusion coefficient ($\text{m}^2/\text{s}$), and $R_{\text{ads}}$ is the surface sink rate localized to boundary cells.

**2. Surface Coverage Dynamics (Surface Boundary $\Gamma_{\text{surf}}$)**:

$$\frac{\partial \theta}{\partial t} = R_{\text{ads}} - R_{\text{des}} - R_{\text{rxn}}$$

governed by Langmuir self-limiting kinetics:

$$R_{\text{ads}} = k_{\text{ads}} C (1 - \theta), \quad R_{\text{des}} = k_{\text{des}} \theta, \quad R_{\text{rxn}} = k_{\text{rxn}} \theta$$

where $\theta(x, y, t) \in [0.0, 1.0]$ represents fractional coverage of chemisorbed precursor, $k_{\text{ads}}$ is the first-order adsorption rate constant ($\text{s}^{-1}$), $k_{\text{des}}$ is the thermal desorption rate constant ($\text{s}^{-1}$), and $k_{\text{rxn}}$ is the co-reactant reaction rate constant ($\text{s}^{-1}$).

**3. Film Thickness Evolution ($\Gamma_{\text{surf}}$)**:

$$\frac{\partial h}{\partial t} = k_{\text{growth}} R_{\text{rxn}} = k_{\text{growth}} k_{\text{rxn}} \theta$$

where $h(x, y, t)$ is accumulated film thickness ($\text{m}$) and $k_{\text{growth}}$ is the growth conversion factor ($\text{m}/\text{cycle}$).

### 2.2 Boundary & Initial Conditions

- **Reactor Inlet Boundary ($y = 0$)**:
  - Dose Phase: Dirichlet boundary condition $C(x, 0, t) = C_0 = 1.0$.
  - Purge & Reaction Phases: Exhaust condition $C(x, 0, t) = 0.0$.
- **Solid Wall & Trench Surfaces**:
  - Homogeneous Neumann zero-flux condition $\nabla C \cdot \mathbf{n} = 0$.
- **Initial State ($t = 0$)**:
  $$C(x, y, 0) = 0, \quad \theta(x, y, 0) = 0, \quad h(x, y, 0) = 0$$

### 2.3 Stability Limits

The explicit 2D central-difference scheme enforces diffusive von Neumann and kinetic stability limits on time step $\Delta t$:

$$\Delta t \le \min\left(0.20 \frac{\min(\Delta x, \Delta y)^2}{D_{\max}}, \frac{0.20}{\max(k_{\text{ads}}, k_{\text{des}}, k_{\text{rxn}}, 10^{-12})}\right)$$

Guaranteed numerical stability requires $\Delta t < \frac{\min(\Delta x, \Delta y)^2}{4 D_{\max}}$. If $\Delta t$ violates this bound, execution raises an exception before integration starts.

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

where $\delta_{\text{pulse}}, \delta_{\text{rxn}} \in \{0, 1\}$ indicate precursor dose and co-reactant reaction stages, respectively. Residuals are computed using physical coordinates ($dx, dy, dt$) and unnormalized field values ($C, \theta, h$).

---

## 4. Parameter Ranges & Physical Specifications

| Parameter | Symbol | Unit | Range / Distribution |
|:---|:---:|:---:|:---|
| Precursor Diffusivity | $D$ | $\text{m}^2/\text{s}$ | $\mathcal{U}(1.0 \times 10^{-3}, 8.0 \times 10^{-3})$ |
| Adsorption Rate Constant | $k_{\text{ads}}$ | $\text{s}^{-1}$ | $\mathcal{U}(0.5, 1.5)$ |
| Desorption Rate Constant | $k_{\text{des}}$ | $\text{s}^{-1}$ | $\mathcal{U}(0.02, 0.10)$ |
| Co-reactant Reaction Rate | $k_{\text{rxn}}$ | $\text{s}^{-1}$ | $\mathcal{U}(0.1, 0.8)$ |
| Growth Conversion Factor | $k_{\text{growth}}$ | $\text{m}/\text{cycle}$ | $0.02$ (fixed nominal) |
| Precursor Dose Time | $t_{\text{pulse}}$ | $\text{s}$ | Bimodal: $85\%\; \mathcal{U}(10.0, 60.0)$, $15\%\; \mathcal{U}(60.0, 100.0)$ |
| Purge Time | $t_{\text{purge}}$ | $\text{s}$ | $1.0$ (fixed nominal) |
| Co-reactant Exposure Time | $t_{\text{rxn}}$ | $\text{s}$ | $1.0$ (fixed nominal) |
| Spatial Domain Resolution | $N_x \times N_y$ | grid points | $160 \times 120$ (reference) / $80 \times 60$ (accelerated) |

---

## 5. Repository Structure

```
thin-film-pino/
├── configs/
│   └── default.yaml                # Default model hyperparameters
├── docs/
│   ├── figures/                    # Curated figures
│   ├── archive/                    # Archived historical artifacts
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
│       ├── physics/                # Continuum ALD simulators (CPU and PyTorch GPU/CPU)
│       └── training/               # Unified training pipeline for FNO and PINO
├── tests/
│   └── test_temporal_operator_loss.py # Masked reduction unit test
├── pyproject.toml                  # Packaging specification and dependencies
├── requirements.txt                # Pinned dependency requirements
└── README.md                       # Technical documentation and workflow guide
```

---

## 6. Reproducible Workflow & CLI Guide

### 6.1 Environment Setup

Install the package in editable mode with development dependencies:

```bash
pip install -e .[dev]
```

### 6.2 Simulator Parity Verification

Compare trajectories from the reference CPU solver and the accelerated PyTorch simulator:

```bash
python scripts/validate_cuda_reference.py --output results/validation --num-cycles 10
```

### 6.3 Transient Trajectory Generation & Preparation

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

Train baseline FNO and physics-informed models using console entry points or module execution:

```bash
# Train baseline Temporal Fourier Neural Operator (TFNO)
python -m thin_film_pino.training.train --model fno --input data/transient_ald_prepared.npz --epochs 50 --batch-size 4 --output-dir runs/fno

# Train Physics-Informed Neural Operator (PINO)
python -m thin_film_pino.training.train --model pino --input data/transient_ald_prepared.npz --epochs 50 --batch-size 4 --physics-weight 1.0e-4 --output-dir runs/pino
```

### 6.5 Evaluation & Comparative Benchmarking

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

## 7. Numerical Standards & Verification

- **Bounds Enforcement**: Surface coverage $\theta(x, y, t) \in [0.0, 1.0]$ and precursor concentration $C(x, y, t) \ge 0.0$ at every discrete time step across CPU and GPU solvers.
- **Reference Parity**: Parity between CPU and accelerated solvers can be evaluated with `scripts/validate_cuda_reference.py`. As documented in [Known Limitations](docs/known_limitations.md), cross-grid spatial resampling of one-cell surface boundaries introduces interpolation errors.
- **Split Integrity**: Trajectory splits are disjoint and normalization statistics are computed strictly on the training partition without cross-split leakage.
