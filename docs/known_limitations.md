# Known Limitations

This document records physical, mathematical, and numerical limitations identified in the continuum ALD simulator and neural operator surrogates.

## 1. Physical Model and Governing Equations

- **Operator Splitting by Phase**: The continuum transport-reaction system is operator-split into four discrete ALD phases per cycle: precursor dose, purge, co-reactant reaction, and post-reaction purge. Adsorption and desorption are active only during the precursor dose. Co-reactant surface reaction is active only during the reaction phase. Desorption is inactive during purge.
- **Perfect Purge vs. Diffusion Timescale**: Precursor concentration in the gas domain is cleared ($C \equiv 0$) after each purge phase. Because the nominal purge time ($1.0 \text{ s}$) is much shorter than the characteristic diffusion time $\tau_{\text{diff}} = L^2 / D \approx 25\text{--}200 \text{ s}$, instantaneous clearance is an operational assumption rather than a physical transport outcome.
- **Mass Non-Conservation**: The coupled equations do not conserve global mass. Precursor chemisorption acts as a volumetric gas sink at boundary nodes without coupling to physical surface site density, and thermal desorption does not return vapor into the gas domain.
- **Incomplete Reaction Carryover**: For $k_{\text{rxn}} \in [0.1, 0.8] \text{ s}^{-1}$ and $t_{\text{rxn}} = 1.0 \text{ s}$, at most $1 - e^{-0.8} \approx 55.1\%$ of chemisorbed precursor reacts per cycle. The remaining unreacted coverage carries over into subsequent cycles.
- **Constant Diffusivity**: Precursor transport is modeled via Fickian diffusion with a scalar constant diffusivity $D$. It does not incorporate Knudsen diffusion, pressure-dependent mean free path adjustments, or transition-regime rarefied gas transport.
- **Trench Aspect Ratio**: The default simulated trench geometry has a width of 0.25 and depth of 0.45, giving an aspect ratio of 1.8. This corresponds to moderate aspect ratio features. A narrow high-aspect-ratio trench ($AR = 10.0$) is available via `SimulationConfig.high_aspect_ratio_preset(10.0)`.

## 2. Numerical Discretization and Solvers

- **Explicit Time Integration**: Precursor diffusion is integrated using an explicit forward-time central-space (FTCS) scheme bounded by the diffusive von Neumann stability limit $\Delta t \le 0.20 \min(\Delta x, \Delta y)^2 / D$ and reaction kinetic limit $\Delta t \le 0.20 / \max(k)$. For fine spatial grids, this requires small time steps.
- **Time Step Consistency**: In `BatchedTransientALDCudaSimulator`, the integration time step $\Delta t$ is computed from fixed prior parameter upper bounds (or per-sample when batch size is 1), ensuring that a trajectory does not depend on which other samples share its batch.
- **Physical Bounds Enforced by Clamping**: Non-negativity $C \ge 0$ and fractional coverage bounds $0 \le \theta \le 1$ are enforced by numerical clamping and monitored via activation counters (`clamp_stats`). For default parameter sets, zero clamp activations occur.
- **Direct Parity on Matched Grids**: Parity between the NumPy reference simulator and the PyTorch accelerated simulator is evaluated on matched grids and identical float64 precision without spatial interpolation, demonstrating machine-precision parity.

## 3. Neural Operator and PINO Surrogate

- **Discrete Snapshot Residuals**: The PINO loss evaluates continuum PDE residuals at discrete saved trajectory snapshots using numerical finite differences, rather than continuous spatio-temporal coordinates via automatic differentiation.
- **Phase Boundary Derivatives**: Finite temporal differences that span phase transitions (such as dose-to-purge or reaction-to-purge) produce artificial derivative spikes. Residual evaluations must be masked to exclude transition intervals.
- **Nondimensional Units**: The physical system operates in nondimensional coordinates ($x/L_x, y/L_y, C/C_0, \theta \in [0, 1]$). Evaluation metrics must report nondimensional errors or state explicit reference scaling.
