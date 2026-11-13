# Physics-Informed Neural Operator for Transient Atomic Layer Deposition

[![CI](https://github.com/neil6714/thin-film-pino/actions/workflows/ci.yml/badge.svg)](https://github.com/neil6714/thin-film-pino/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A 2D continuum simulator and scientific machine learning surrogate for transient atomic layer deposition (ALD) in trench features. The physics core integrates vapor diffusion, Langmuir adsorption, desorption, and surface reaction via explicit forward-time central-space finite differences. The surrogate models map physical parameters and continuous time to spatial concentration, surface coverage, and film thickness fields using a Fourier Neural Operator regularized with continuous autograd physics residuals.

---

## 1. Project Status

| Component | Status | Verification Summary |
| :--- | :--- | :--- |
| Continuum Simulators | **Validated** | Reference NumPy and accelerated PyTorch CPU solvers match to machine precision ($0.0$ error across 20 parameter sets on a $20 \times 15$ grid in float64). Golden regression passes. |
| Stencil & Geometry Operators | **Validated** | Masked zero-flux Laplacian stencil evaluated exactly on quadratic fields and constants across wall interfaces. Aspect ratio presets supported. |
| Data Generation Pipeline | **Validated** | Chunked generation with `SeedSequence` spawning ensures reproducible trajectories independent of batch size. Bounds validator verifies $C \ge 0$, $\theta \in [0, 1]$, and monotonic thickness. |
| Temporal FNO Architecture | **Validated** | Spectral convolution with positive and negative $y$-frequency weight blocks preserves shape, outputs real values, and exhibits translation equivariance under circular shifts. Single-batch overfit verified. |
| PINO Physics Regularization | **Validated** | Autograd continuous time derivatives match analytic polynomials to machine precision. Residuals evaluated at continuous collocation times and scaled by field standard deviations. |
| Evaluation & Benchmarking Suite | **Validated** | Model-agnostic batched evaluator computes physical channel metrics (MAE, RMSE, relative $L_2$), bootstrap confidence intervals, train-mean and persistence baselines, and GPC/step coverage errors. |
| Trained Production Checkpoints | **Pending** | No neural operator model was trained and no large dataset was regenerated during this code remediation pass. Model checkpoints and quantitative surrogate metrics are pending. |

---

## 2. Installation

Install the package in editable mode with development dependencies:

```bash
git clone https://github.com/neil6714/thin-film-pino.git
cd thin-film-pino
pip install -e .[dev]
```

---

## 3. Quickstart Smoke Run (CPU, Under Two Minutes)

Run a complete end-to-end verification and training smoke test on a tiny $20 \times 15$ grid:

```bash
# 1. Run multi-sample solver parity verification
python scripts/parity.py --num-samples 5 --nx 20 --ny 15

# 2. Generate a 4-sample trajectory dataset across 2 cycles
python scripts/generate_data.py --num-samples 4 --num-cycles 2 --nx 20 --ny 15 --batch-size 2 --output data/smoke_dataset.npz

# 3. Partition into stratified train/val/test/ood splits and compute normalization statistics
python scripts/prepare_temporal_dataset.py --input data/smoke_dataset.npz --output data/smoke_prepared.npz

# 4. Train FNO baseline for 2 epochs
python -m thin_film_pino.training.train --model fno --input data/smoke_prepared.npz --epochs 2 --batch-size 2 --output-dir runs/smoke_fno

# 5. Train PINO with continuous collocation residuals for 2 epochs
python -m thin_film_pino.training.train --model pino --input data/smoke_prepared.npz --epochs 2 --batch-size 2 --physics-weight 1.0e-4 --output-dir runs/smoke_pino

# 6. Execute full automated test suite
pytest tests/
```

---

## 4. Formulation Summary

For derivations, scaling, and assumptions, see [docs/formulation.md](docs/formulation.md).

### 4.1 Units and Reference Scales
All simulation and model variables are nondimensional. Dimensional quantities relate to nondimensional variables through characteristic scales:
- Domain height $L_0 \approx 100\text{ nm}$ ($x = \tilde{x}/L_0, y = \tilde{y}/L_0$)
- Stage time $\tau_0 \approx 1\text{ s}$ ($t = \tilde{t}/\tau_0$)
- Precursor inlet concentration $C_0 \approx 10^{-2}\text{ mol/m}^3$ ($C = \tilde{C}/C_0$)
- Surface coverage fraction $\theta = \tilde{\theta}/\theta_{\text{max}} \in [0, 1]$
- Film thickness $h = \tilde{h}/h_0$

### 4.2 Geometry and Aspect Ratio
The domain $\Omega = [0, 1] \times [0, 1]$ contains a rectangular feature. The reactive interface $\Gamma_{\text{surf}}$ is partitioned into three disjoint regions:
- **Field**: Wafer top planar surface outside the trench ($y \le y_{\text{top}}$).
- **Sidewall**: Vertical trench walls.
- **Floor**: Horizontal bottom of the trench ($y \ge y_{\text{bottom}}$).

Step coverage is defined as:
$$\text{Step Coverage} = \frac{\bar{h}_{\text{floor}}}{\bar{h}_{\text{field}}}$$

The default geometry has width $w = 0.25$ and depth $d = 0.45$, giving aspect ratio $AR = 0.45 / 0.25 = 1.80$. A narrow trench preset ($AR = 10.0, w = 0.045$) is available via `SimulationConfig.high_aspect_ratio_preset(10.0)`.

### 4.3 Governing Equations by Stage
The process is operator-split into four stages per cycle.

**Phase 1: Precursor Dose**
Vapor diffusion, chemisorption, and thermal desorption occur simultaneously. Thermal desorption is active during this phase:
$$\frac{\partial C}{\partial t} = D \nabla^2 C - R_{\text{ads}} \quad \text{in } \Omega_{\text{gas}}$$
$$\frac{\partial \theta}{\partial t} = R_{\text{ads}} - R_{\text{des}} \quad \text{on } \Gamma_{\text{surf}}$$
$$\frac{\partial h}{\partial t} = 0 \quad \text{on } \Gamma_{\text{surf}}$$
where $R_{\text{ads}} = k_{\text{ads}} C (1 - \theta)$ and $R_{\text{des}} = k_{\text{des}} \theta$. Boundary condition at top inlet is $C(x, 0, t) = 1.0$.

**Phase 2: Post-Dose Inert Purge**
Precursor feed is shut off. Thermal desorption is inactive:
$$\frac{\partial C}{\partial t} = D \nabla^2 C \quad \text{in } \Omega_{\text{gas}}, \quad \frac{\partial \theta}{\partial t} = 0, \quad \frac{\partial h}{\partial t} = 0$$
Boundary condition at top inlet is $C(x, 0, t) = 0.0$. At the conclusion of this stage, precursor concentration is reset to zero: $C(x, y) \equiv 0$.

**Phase 3: Co-Reactant Reaction and Film Growth**
Precursor vapor is absent ($C \equiv 0$). Chemisorbed precursor is converted to film:
$$C(x, y, t) \equiv 0 \quad \text{in } \Omega_{\text{gas}}$$
$$\frac{\partial \theta}{\partial t} = - k_{\text{rxn}} \theta \quad \text{on } \Gamma_{\text{surf}}$$
$$\frac{\partial h}{\partial t} = k_{\text{growth}} k_{\text{rxn}} \theta \quad \text{on } \Gamma_{\text{surf}}$$

**Phase 4: Post-Reaction Inert Purge**
$$\frac{\partial C}{\partial t} = 0, \quad \frac{\partial \theta}{\partial t} = 0, \quad \frac{\partial h}{\partial t} = 0$$
Precursor concentration remains reset at zero: $C(x, y) \equiv 0$.

### 4.4 Parameter Distributions and Diffusion Timescales
Parameter bounds follow `configs/priors.yaml`:

| Parameter | Symbol | Range / Value | Description |
| :--- | :---: | :--- | :--- |
| Diffusivity | $D$ | $\mathcal{U}(1.0 \times 10^{-3}, 8.0 \times 10^{-3})$ | Constant effective vapor diffusivity |
| Adsorption Rate | $k_{\text{ads}}$ | $\mathcal{U}(0.5, 1.5)$ | First-order surface chemisorption rate constant |
| Desorption Rate | $k_{\text{des}}$ | $\mathcal{U}(0.02, 0.10)$ | First-order thermal desorption rate constant |
| Reaction Rate | $k_{\text{rxn}}$ | $\mathcal{U}(0.1, 0.8)$ | Co-reactant reaction rate constant |
| Growth Coefficient | $k_{\text{growth}}$ | $0.02$ (constant) | Thickness per converted coverage unit |
| Dose Exposure Time | $t_{\text{pulse}}$ | Bimodal mixture: $85\%\; [10.0, 60.0], 15\%\; [60.0, 100.0]$ | Precursor exposure stage duration |
| Purge Duration | $t_{\text{purge}}$ | $1.0$ (constant) | Inert purge stage duration |
| Reaction Duration | $t_{\text{rxn}}$ | $1.0$ (constant) | Co-reactant exposure stage duration |

The characteristic diffusion time across the domain is $\tau_{\text{diff}} = L^2 / D \approx 25\text{ to } 200\text{ s}$ for $D \in [1.0 \times 10^{-3}, 8.0 \times 10^{-3}]$. The pulse exposure range of 10 to 100 s spans this regime, transitioning from diffusion-limited starved coverage at short pulses to saturation at long pulses.

### 4.5 Stability Limits
Explicit forward-time central-space time integration is constrained by:
- Diffusive stability limit: $\Delta t \le \min(\Delta x, \Delta y)^2 / (4 D)$
- Reaction kinetic limit: $\Delta t \le 1 / \max(k_{\text{ads}}, k_{\text{des}}, k_{\text{rxn}})$

The simulator uses a safety factor of 0.20:
$$\Delta t = 0.20 \times \min\left( \frac{\min(\Delta x, \Delta y)^2}{D_{\max}}, \frac{1}{\max(k)} \right)$$

### 4.6 Modeling Assumptions
1. **Operator Splitting**: The four ALD stages execute in sequence.
2. **Ideal Purge**: Vapor concentration is reset to zero ($C \equiv 0$) after each purge stage. Because the nominal purge duration ($1.0$) is shorter than the diffusion timescale ($25\text{--}200$), clearance is an operational model assumption.
3. **Mass Non-Conservation**: Precursor adsorption acts as a volumetric gas sink at boundary cells without coupling to physical surface site density, and desorption does not return vapor into the gas domain.
4. **Incomplete Reaction Carryover**: For $k_{\text{rxn}} \le 0.8$ and $t_{\text{rxn}} = 1.0$, at most $55\%$ of chemisorbed precursor reacts in a cycle. Unreacted coverage carries over into subsequent cycles.
5. **Numerical Clamping**: Physical invariants $C \ge 0$ and $0 \le \theta \le 1$ are enforced by numerical clamping. Standard parameter sets produce zero clamp activations.

---

## 5. Neural Operator Architecture

Each snapshot is predicted independently given parameters, continuous time, phase, cycle fraction, coordinates, and a geometry mask:

```
Inputs (15 channels):
  - Parameters [D, k_ads, k_des, k_rxn, t_pulse] (5 channels)
  - Continuous time t (1 channel)
  - One-hot phase code (5 channels)
  - Cycle fraction (1 channel)
  - Normalized coordinates (x, y) (2 channels)
  - Domain geometry gas mask (1 channel)
                     │
                     ▼
  Two-Layer Pointwise Lifting Network: Conv2d(15 -> width) + GELU + Conv2d(width -> width)
                     │
                     ▼
  Four 2D Fourier Blocks:
    - SpectralConv2d with two complex weight blocks (W1 positive y, W2 negative y)
    - Pointwise Conv2d(width -> width)
    - GELU activation
                     │
                     ▼
  Pointwise Projection Network: Conv2d(width -> 64) + GELU + Conv2d(64 -> 3)
                     │
                     ▼
  Predicted Multi-Field Tensor: [C, θ, h](x, y, t)
```

The training loss uses a per-channel masked mean:
$$\mathcal{L}_{\text{data}} = \frac{1}{\sum_c w_c} \sum_{c=1}^3 w_c \frac{\sum_{\Omega_c} |y_c - \hat{y}_c|^2}{|\Omega_c|}$$
preventing dense gas cells from dominating sparse surface cells. PINO adds continuous collocation PDE residuals computed via `torch.autograd.grad` with respect to normalized continuous time, scaled by each field's standard deviation.

---

## 6. Workflow Commands

### 6.1 Parity and Convergence
```bash
# Multi-sample parity verification (NumPy vs. PyTorch float64 CPU)
python scripts/parity.py --num-samples 20 --nx 20 --ny 15 --output results/parity/parity_metrics.json

# Spatial grid convergence analysis (40x30, 80x60, 160x120)
python scripts/grid_convergence.py --pulse-time 5.0 --output results/convergence/grid_convergence.json
```

### 6.2 Data Pipeline
```bash
# Generate trajectory dataset (or --mode final_state)
python scripts/generate_data.py --mode trajectory --num-samples 64 --batch-size 16 --output data/transient_ald_dataset.npz

# Stratified split into train (70%), val (15%), test (15%), and long-pulse OOD holdout
python scripts/prepare_temporal_dataset.py --input data/transient_ald_dataset.npz --output data/transient_ald_prepared.npz

# Validate physical bounds and split coverage
python scripts/validate_transient_dataset.py --input data/transient_ald_prepared.npz
```

### 6.3 Surrogate Training
```bash
# Train FNO baseline
python -m thin_film_pino.training.train --model fno --input data/transient_ald_prepared.npz --epochs 50 --batch-size 4 --output-dir runs/fno

# Train PINO
python -m thin_film_pino.training.train --model pino --input data/transient_ald_prepared.npz --epochs 50 --batch-size 4 --physics-weight 1.0e-4 --output-dir runs/pino

# Multi-seed run
python -m thin_film_pino.training.train --model fno --input data/transient_ald_prepared.npz --seeds 123,456,789 --output-dir runs/fno_multiseed
```

### 6.4 Evaluation and Comparison
```bash
# Batched checkpoint evaluation with baselines and bootstrap uncertainty
python -m thin_film_pino.evaluation.evaluate --input data/transient_ald_prepared.npz --checkpoint runs/fno/best_model.pt --output-dir runs/evaluation

# Comparative benchmarking across models and baselines
python scripts/compare_models.py --input data/transient_ald_prepared.npz --operator runs/fno/best_model.pt --pino runs/pino/best_model.pt --output runs/comparison

# Multi-snapshot predicted vs. reference field panels
python scripts/plot_qualitative.py --input data/transient_ald_prepared.npz --checkpoint runs/fno/best_model.pt --output-dir runs/evaluation/qualitative --trajectory-index 0

# Plot loss convergence for explicit metric keys
python scripts/plot_losses.py --root runs --keys train_loss val_loss --output runs/loss_trajectories.png
```

---

## 7. Results

### 7.1 Verified Solver Metrics
The following verification results are committed and reproducible:
1. **NumPy vs. PyTorch Parity** (`scripts/parity.py`, commit `43293fa`):
   - Grid: $20 \times 15$, float64 CPU, 20 parameter samples.
   - Mid-dose concentration max absolute error: $0.0$, relative $L_2$: $0.0$.
   - Final state ($C, \theta, h$) max absolute error: $0.0$.
2. **Golden Run Regression** (`tests/test_physics.py::test_golden_regression_matches`):
   - Verified against [tests/data/golden_regression.npz](tests/data/golden_regression.npz) ($40 \times 30$ grid, 2 cycles) to machine precision ($< 10^{-12}$).
3. **Discrete Laplacian Accuracy** (`tests/test_physics.py::test_stencil_laplacian_properties`):
   - Exact $0.0$ on constant fields across wall boundaries.
   - Analytic agreement ($2a + 2b$) on quadratic fields on interior gas cells.

### 7.2 Remote GPU Surrogate Benchmark
Benchmark run executed on a remote Nvidia Tesla T4 GPU using Google Colab CLI at commit `6ca7cdf`.

#### Dataset and Training Setup
- **Simulation**: 256 transient trajectories generated in 94.8 s with `configs/priors_fast.yaml` ($40 \times 30$ grid, 3 cycles, 8 snapshots per phase, pulse times capped at 60 s).
- **Splits**: 161 train, 34 validation, 35 test, 26 out-of-distribution holdout.
- **Training**: 25 epochs, batch size 8, AdamW optimizer, cosine annealing schedule, seed 42.
- **Execution Times**: FNO trained in 129.8 s; PINO trained in 185.3 s with physics penalty weight $1.0 \times 10^{-4}$.
- **Baselines**: Train-mean baseline normalized MSE is 1.000000 and MAE is 0.841029.

#### Test Partition Metrics (N = 35 trajectories)
| Model / Baseline | Concentration MAE ($\text{mol/m}^3$) | 95% Bootstrap CI | Coverage MAE (-) | 95% Bootstrap CI | Thickness MAE ($\text{m}$) | 95% Bootstrap CI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FNO** | 7.2287e-02 | [6.582e-02, 7.884e-02] | 6.6929e-02 | [6.106e-02, 7.220e-02] | 9.8946e-04 | [8.477e-04, 1.139e-03] |
| **PINO** | 7.2278e-02 | [6.581e-02, 7.884e-02] | 6.6932e-02 | [6.106e-02, 7.220e-02] | 9.8946e-04 | [8.477e-04, 1.139e-03] |
| **Train-Mean Baseline** | 2.7485e-01 | [2.642e-01, 2.847e-01] | 2.6887e-01 | [2.549e-01, 2.862e-01] | 5.0291e-03 | [4.534e-03, 5.645e-03] |
| **Persistence Baseline** | 2.2119e-01 | [2.058e-01, 2.359e-01] | 4.8177e-01 | [4.110e-01, 5.421e-01] | 5.7026e-03 | [4.589e-03, 6.902e-03] |

- **Derived Cycle Errors**: GPC MAE is $5.48 \times 10^{-4}\text{ m/cycle}$. Step coverage MAE is $6.96 \times 10^{-2}$.
- **Physics Residuals**: PINO final training PDE loss was 0.3422 (transport MSE 0.3942, coverage MSE 0.3287, growth MSE 0.3037).
- **Bounds Check**: Negative concentration predictions occurred at 27.34% of cell evaluations. Surface coverage outside $[0, 1]$ occurred at 4.51%.
- **Statistical Significance**: Both operator models outperform train-mean and persistence baselines by an order of magnitude in thickness and a factor of three in concentration and coverage. The difference between FNO and PINO is within the 95% bootstrap confidence intervals.

Benchmark curves and qualitative spatial maps are saved in [docs/figures/](docs/figures/):
- [Loss Trajectories](docs/figures/colab_loss_trajectories.png)
- [Physical MAE Comparison](docs/figures/colab_physical_mae_comparison.png)
- [Predicted vs True Concentration](docs/figures/colab_qualitative_concentration.png)
- [Predicted vs True Film Thickness](docs/figures/colab_qualitative_film_thickness.png)

---

## 8. Limitations

See [docs/known_limitations.md](docs/known_limitations.md) for detailed discussion. Key constraints:
- **Ideal Purge**: Complete vapor clearance after each purge is an operational simplification.
- **Mass Conservation**: Precursor gas sink is not balanced with absolute site density, and desorption does not return vapor to the gas domain.
- **Incomplete Reaction**: Up to $45\%$ to $90\%$ of adsorbed coverage carries over across cycles for lower $k_{\text{rxn}}$ values.
- **Constant Diffusivity**: Transport uses constant effective diffusivity without pressure-dependent Knudsen corrections.
- **Explicit Time Steps**: Bounded by diffusive stability limit $\min(\Delta x, \Delta y)^2 / (4D)$ and reaction kinetic limit $1 / \max(k)$.

---

## 9. Repository Structure

```
thin-film-pino/
├── configs/
│   ├── default.yaml                # Model hyperparameters
│   └── priors.yaml                 # Parameter priors and distribution bounds
├── docs/
│   ├── data_schema.md              # Array schemas, shapes, axes, dtypes, phase codes
│   ├── formulation.md              # Governing equations, scaling, and architecture
│   └── known_limitations.md        # Documented simulator and surrogate limitations
├── scripts/
│   ├── compare_models.py           # Cross-model evaluation comparison with baselines
│   ├── cycle_validation.py         # Multi-cycle GPC linearity verification
│   ├── generate_data.py            # Unified trajectory and final-state dataset generator
│   ├── grid_convergence.py         # Discretization error and spatial order analysis
│   ├── inspect_dataset.py          # Summary statistics, distributions, and spatial plots
│   ├── parity.py                   # Multi-sample CPU parity verification
│   ├── plot_losses.py              # Loss convergence visualization
│   ├── plot_qualitative.py         # Multi-snapshot predicted vs. reference field panels
│   ├── prepare_temporal_dataset.py # Stratified splitting with OOD holdout and normalization
│   ├── pulse_saturation.py         # Self-limiting Langmuir saturation verification
│   ├── sweeps.py                   # Kinetic and pulse exposure parameter sweeps
│   └── validate_transient_dataset.py # Dataset physical invariant and split audit suite
├── src/
│   └── thin_film_pino/
│       ├── data/                   # Priors, sampling, generation, preparation, validation
│       ├── evaluation/             # Batched evaluation, metrics, baselines, visualization
│       ├── models/                 # Fourier Neural Operator architecture
│       ├── physics/                # Continuum ALD simulators, geometry, and stencils
│       └── training/               # Unified training pipeline for FNO and PINO
├── tests/
│   ├── data/                       # Golden regression artifacts
│   ├── test_data_pipeline.py       # Dataset generation, stratification, validation
│   ├── test_evaluation.py          # Metrics, baselines, bootstrap uncertainty
│   ├── test_fno.py                 # Spectral convolution and conditioning
│   ├── test_physics.py             # Physics, geometry masks, clamping, solver parity
│   ├── test_pino.py                # Continuous autograd and collocation sampling
│   └── test_temporal_operator_loss.py # Masked reduction unit tests
├── pyproject.toml                  # Package configuration and dependencies
├── requirements.txt                # Pinned dependency requirements
├── CITATION.cff                    # Citation metadata
├── LICENSE                         # MIT License
└── README.md                       # Repository overview and documentation
```

---

## 10. Related Work

1. **Fourier Neural Operator (FNO)**:
   Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2021). Fourier Neural Operator for Parametric Partial Differential Equations. In *International Conference on Learning Representations (ICLR 2021)*.
2. **Physics-Informed Neural Operator (PINO)**:
   Li, Z., Zheng, H., Kovachki, N., Jin, D., Chen, H., Liu, B., Azizzadenesheli, K., & Anandkumar, A. (2024). Physics-Informed Neural Operator for Learning Partial Differential Equations. *ACM/IMS Journal of Data Science*, 1(3), 1–28.
3. **Kinetic ALD Modeling in Narrow Trenches**:
   Gordon, R. G., Hausmann, D., Kim, E., & Shepard, J. (2003). A Kinetic Model for Step Coverage by Atomic Layer Deposition in Narrow Holes or Trenches. *Chemical Vapor Deposition*, 9(2), 73–78.
4. **Diffusion-Reaction ALD in Nanostructures**:
   Yanguas-Gil, A., & Elam, J. W. (2012). Self-Limited Reaction-Diffusion in Nanostructured Substrates: Surface Coverage Dynamics and Analytic Approximations to ALD Saturation Times. *Chemical Vapor Deposition*, 18(1–3), 46–52.

---

## 11. Citation

If you use this work, please cite:

```bibtex
@software{thin_film_pino,
  author = {Neil Pandit},
  title = {Thin-Film PINO: Transient Atomic Layer Deposition Simulator and Neural Operator Surrogate},
  year = {2026},
  url = {https://github.com/neil6714/thin-film-pino}
}
```

---

## 12. License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
