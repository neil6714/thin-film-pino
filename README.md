# Physics-Informed Neural Operator for Transient Atomic Layer Deposition

A continuum modeling and scientific machine learning framework for simulating atomic layer deposition (ALD) in microelectronic high-aspect-ratio feature geometries and learning fast neural operator surrogates with physics-informed constraints.

---

## 1. Overview & Theoretical Foundations

Atomic Layer Deposition (ALD) is a vapor-phase thin-film fabrication technique based on sequential, self-limiting surface chemical reactions. In high-aspect-ratio trench structures, precursor transport is governed by Knudsen and Fickian diffusion coupled to nonlinear Langmuir adsorption and desorption kinetics.

This framework develops:
1. **High-Fidelity Continuum Simulators**: 2D finite-difference time-domain (FDTD) solvers with explicit diffusive Courant-Friedrichs-Lewy (CFL) stability enforcement and graceful GPU/CPU device fallback.
2. **Temporal Fourier Neural Operators (TFNO)**: Deep operator networks mapping physical process parameters and continuous time coordinates to spatial multi-field states $(\mathbf{p}, t) \mapsto (C, \theta, h)(x, y, t)$.
3. **Physics-Informed Neural Operators (PINO)**: Operator learning regularized by transient PDE residual penalties enforcing mass conservation and Langmuir kinetic consistency without requiring labelled simulation data at every intermediate time step.

---

## 2. Mathematical & Physical Formulation

### 2.1 Governing Equations

The coupled continuum transport-reaction system in a 2D domain $\Omega = [0, L_x] \times [0, L_y]$ is formulated as:

**1. Precursor Vapor Transport (Gas Domain $\Omega_{\text{gas}}$)**:

$$\frac{\partial C}{\partial t} = D \nabla^2 C - R_{\text{ads}}(x, y, t)$$

where $C(x, y, t)$ is the precursor molar concentration ($\text{mol}/\text{m}^3$), $D$ is the diffusion coefficient ($\text{m}^2/\text{s}$), and $R_{\text{ads}}$ is the volumetric surface sink rate localized to surface boundary cells.

**2. Surface Coverage Dynamics (Surface Boundary $\Gamma_{\text{surf}}$)**:

$$\frac{\partial \theta}{\partial t} = R_{\text{ads}} - R_{\text{des}} - R_{\text{rxn}}$$

governed by Langmuir self-limiting kinetics:

$$R_{\text{ads}} = k_{\text{ads}} C (1 - \theta), \quad R_{\text{des}} = k_{\text{des}} \theta, \quad R_{\text{rxn}} = k_{\text{rxn}} \theta$$

where $\theta(x, y, t) \in [0.0, 1.0]$ represents the fractional coverage of chemisorbed precursor species, $k_{\text{ads}}$ is the first-order adsorption rate constant ($\text{s}^{-1}$), $k_{\text{des}}$ is the thermal desorption rate constant ($\text{s}^{-1}$), and $k_{\text{rxn}}$ is the co-reactant reaction rate constant ($\text{s}^{-1}$).

**3. Film Thickness Evolution ($\Gamma_{\text{surf}}$)**:

$$\frac{\partial h}{\partial t} = k_{\text{growth}} R_{\text{rxn}} = k_{\text{growth}} k_{\text{rxn}} \theta$$

where $h(x, y, t)$ is the accumulated deposited film thickness ($\text{m}$) and $k_{\text{growth}}$ is the growth-per-cycle conversion factor ($\text{m}/\text{cycle}$).

### 2.2 Boundary & Initial Conditions

- **Reactor Inlet Boundary ($y = 0$)**:
  - Dose / Pulse Phase: Dirichlet boundary condition $C(x, 0, t) = C_0 = 1.0$.
  - Purge & Reaction Phases: Exhaust condition $C(x, 0, t) = 0.0$.
- **Solid Wall & Trench Surfaces**:
  - Homogeneous Neumann zero-flux condition $\nabla C \cdot \mathbf{n} = 0$.
- **Initial Substrate State ($t = 0$)**:
  $$C(x, y, 0) = 0, \quad \theta(x, y, 0) = 0, \quad h(x, y, 0) = 0$$

### 2.3 Explicit Diffusive CFL Stability Criterion

The explicit 2D central-difference scheme enforces strict upper bounds on the time step $\Delta t$:

$$\Delta t \le \min\left(0.20 \frac{\min(\Delta x, \Delta y)^2}{D_{\max}}, \frac{0.20}{\max(k_{\text{ads}}, k_{\text{des}}, k_{\text{rxn}}, 10^{-12})}\right)$$

Guaranteed stability requires $\Delta t < \frac{\min(\Delta x, \Delta y)^2}{4 D_{\max}}$. If an externally supplied or computed $\Delta t$ violates this limit, execution is immediately halted with a descriptive exception before simulation execution.

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

The 2D Fourier layer parameterizes integral kernel operators directly in the frequency domain:

$$(\mathcal{K}(v))(x) = \mathcal{F}^{-1}\left(R_\phi \cdot (\mathcal{F} v)\right)(x)$$

where $\mathcal{F}$ and $\mathcal{F}^{-1}$ denote the 2D Real Fast Fourier Transform (`torch.fft.rfft2`) and its inverse (`torch.fft.irfft2`), with learnable complex frequency weight tensor:

$$R_\phi \in \mathbb{C}^{\text{channels} \times \text{channels} \times k_{\max, y} \times k_{\max, x}}$$

truncated at spatial frequency mode cutoff $k_{\max} = 12$.

### 3.2 Physics-Informed Residual Formulation

The total optimization objective balances masked data loss against phase-aware continuum residuals:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{physics}} \mathcal{L}_{\text{PDE}}$$

where the PDE residual penalty $\mathcal{L}_{\text{PDE}}$ evaluates three coupled physical components:

**1. Precursor Transport Residual** (evaluated over active gas cells $\Omega_{\text{gas}}$):

$$\mathcal{R}_C = \frac{\partial C}{\partial t} - D \nabla^2 C + \delta_{\text{pulse}} R_{\text{ads}}$$

**2. Surface Coverage Residual** (evaluated over surface cells $\Gamma_{\text{surf}}$):

$$\mathcal{R}_\theta = \frac{\partial \theta}{\partial t} - \delta_{\text{pulse}} (R_{\text{ads}} - R_{\text{des}}) + \delta_{\text{rxn}} R_{\text{rxn}}$$

**3. Film Growth Residual** (evaluated over surface cells $\Gamma_{\text{surf}}$):

$$\mathcal{R}_h = \frac{\partial h}{\partial t} - \delta_{\text{rxn}} k_{\text{growth}} R_{\text{rxn}}$$

where the stage indicators satisfy:

$$\delta_{\text{pulse}} \in \{0, 1\}, \quad \delta_{\text{rxn}} \in \{0, 1\}$$

activating the precursor dose and co-reactant reaction stages, respectively.

All differential terms are computed using physical coordinates ($dx$, $dy$, $dt$) and unscaled physical values ($C$, $\theta$, $h$) rather than normalized tensor indices.

---

## 4. Parameter Ranges & Physical Specifications

| Parameter | Symbol | Unit | Prior Distribution / Range |
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
├── physics/
│   ├── simulator.py                # Reference 2D continuum ALD simulator (CPU NumPy)
│   ├── transient_cuda.py           # Single-trajectory transient simulator (PyTorch GPU/CPU)
│   └── batched_transient_cuda.py   # Vectorized ensemble transient simulator (PyTorch GPU/CPU)
├── data/
│   ├── generate_data.py            # Steady-state deposition profile dataset generator
│   ├── generate_final_dataset.py   # Verified multi-parameter steady-state dataset generator
│   ├── generate_transient_dataset.py     # Transient trajectory dataset generator
│   ├── generate_transient_dataset_fast.py # Accelerated batched transient trajectory generator
│   ├── prepare_temporal_dataset.py # Splitting and normalization pipeline without leakage
│   ├── validate_transient_dataset.py     # Dataset physical invariant and split audit suite
│   ├── validate_cuda_reference.py  # Reference CPU vs. accelerated PyTorch parity verifier
│   ├── inspect_dataset.py          # Summary statistics, distributions, and spatial plots
│   ├── cycle_validation.py         # Multi-cycle GPC stabilization and linearity validation
│   ├── kinetic_sweep.py            # 2D diffusivity vs reaction rate sensitivity sweeps
│   ├── parameter_sweep.py          # Parameter sensitivity contour generator
│   ├── pulse_saturation.py         # Self-limiting Langmuir saturation verification
│   └── pulse_sweep.py              # Pulse duration conformality mapping
├── training/
│   ├── train_temporal_operator.py  # Supervised Temporal FNO training pipeline
│   └── train_temporal_pino.py      # Physics-Informed Neural Operator (PINO) training pipeline
├── evaluation/
│   └── evaluate_stage4c_checkpoint.py # Comprehensive metric evaluation on held-out test split
├── results/
│   ├── compare_all_models.py       # Comparative benchmarking and error distribution analysis
│   └── plot_all_losses.py          # Unified loss convergence visualization
├── tests/
│   └── test_temporal_operator_loss.py # Masked reduction unit test
├── requirements.txt                # Python environment dependencies
└── README.md                       # Technical documentation and reproducibility guide
```

---

## 6. Reproducible Workflow & CLI Guide

### 6.1 Environment Setup

Install core numerical and deep learning dependencies:

```bash
pip install -r requirements.txt
```

### 6.2 Simulator Parity Verification

Verify numerical agreement between the reference CPU solver and the accelerated PyTorch simulator:

```bash
python data/validate_cuda_reference.py --output results/stage4_validation --num-cycles 10
```

### 6.3 Transient Trajectory Generation & Preparation

Generate an accelerated ensemble of ALD trajectories and partition them into non-leaking train/val/test splits:

```bash
# 1. Generate accelerated trajectory dataset
python data/generate_transient_dataset_fast.py --num-samples 64 --batch-size 16 --output data/transient_ald_dataset_fast.npz

# 2. Partition into train (70%), validation (15%), and test (15%) splits
python data/prepare_temporal_dataset.py --input data/transient_ald_dataset_fast.npz --output data/transient_ald_prepared.npz

# 3. Audit dataset integrity and physical bounds
python data/validate_transient_dataset.py --input data/transient_ald_prepared.npz
```

### 6.4 Neural Operator Training

Train baseline and physics-informed models:

```bash
# Train baseline Temporal Fourier Neural Operator (TFNO)
python training/train_temporal_operator.py --input data/transient_ald_prepared.npz --epochs 50 --batch-size 4 --output-dir results/stage4c_temporal_operator

# Train Physics-Informed Neural Operator (PINO)
python training/train_temporal_pino.py --input data/transient_ald_prepared.npz --epochs 50 --batch-size 4 --physics-weight 1.0e-4 --output-dir results/stage4d_temporal_pino
```

### 6.5 Evaluation & Comparative Benchmarking

Evaluate checkpoints on the shared held-out test split:

```bash
# Evaluate Stage 4C TFNO checkpoint
python evaluation/evaluate_stage4c_checkpoint.py --input data/transient_ald_prepared.npz --checkpoint results/stage4c_temporal_operator/best_model.pt --output-dir results/stage4c_evaluation

# Cross-compare models on identical test trajectories
python results/compare_all_models.py --input data/transient_ald_prepared.npz --operator results/stage4c_temporal_operator/best_model.pt --pino results/stage4d_temporal_pino/best_model.pt --output results/model_comparison

# Plot unified convergence curves
python results/plot_all_losses.py --root results --output results/all_model_losses.png
```

### 6.6 Unit Testing

Execute automated regression and unit tests:

```bash
pytest tests/
```

---

## 7. Numerical Standards & Verification Bounds

- **Mass & Physical Bounding**: Fractional surface coverage $\theta(x, y, t)$ is strictly bounded in $[0.0, 1.0]$, and precursor concentration $C(x, y, t) \ge 0.0$ at every discrete finite-difference time step across both CPU and GPU implementations.
- **Reference Parity**: Accelerated simulations resampled to reference grid resolutions achieve $L_2$ relative error $< 10^{-4}$ and Mean Absolute Error $< 10^{-5}$ across all physical fields.
- **Data Integrity**: Normalization statistics (means and standard deviations) are computed strictly on the training trajectory subset, preventing evaluation leakage across splits.
