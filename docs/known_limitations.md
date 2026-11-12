# Known Limitations

This document records physical, mathematical, and numerical limitations identified in the current continuum ALD simulator and neural operator surrogates.

## 1. Physical Model and Governing Equations

- **Operator Splitting by Phase**: The continuum transport-reaction system is operator-split into four discrete ALD phases per cycle: precursor dose, purge, co-reactant reaction, and post-reaction purge. Adsorption and desorption terms are active only during the precursor dose. Co-reactant surface reaction is active only during the reaction phase.
- **Idealized Purge Boundary Condition**: Precursor concentration in the gas domain is cleared ($C \equiv 0$) at the start of each purge phase. The model assumes an idealized purge that removes all vapor-phase precursor instantly rather than simulating convective purge flow or finite purge pump-down kinetics.
- **Mass Conservation**: The coupled equations do not enforce strict global mass conservation across fluid domain boundaries and surface sites. Adsorption removes precursor from boundary cells as a sink term without accounting for closed-volume conservation.
- **Constant Diffusivity**: Precursor transport is modeled via Fickian diffusion with a scalar constant diffusivity $D$. It does not incorporate Knudsen diffusion, pressure-dependent mean free path adjustments, or transition-regime rarefied gas transport.
- **Trench Aspect Ratio**: The default simulated trench geometry has a width of 0.25 and depth of 0.45, giving an aspect ratio of 1.8. This corresponds to moderate aspect ratio features rather than extreme high-aspect-ratio structures.

## 2. Numerical Discretization and Solvers

- **Explicit Time Integration**: Precursor diffusion is integrated using an explicit forward-time central-space (FTCS) scheme bounded by the diffusive von Neumann stability limit $\Delta t \le 0.20 \min(\Delta x, \Delta y)^2 / D_{\max}$. For fine spatial grids, this requires small time steps.
- **Batched Simulation Time Steps**: In `BatchedTransientALDCudaSimulator`, the integration time step $\Delta t$ is computed from the maximum diffusivity and rate constants across the entire batch. Consequently, the exact time step used for a specific parameter sample depends on which other samples share its execution batch.
- **Grid Resampling and Parity Verification**: Comparing reference simulations (160x120, float64) with accelerated simulations (80x60, float32) requires spatial interpolation. Because surface coverage and deposition thickness are localized to single-cell boundaries, bilinear resampling introduces interpolation errors that can dominate the reported cross-resolution difference.

## 3. Neural Operator and PINO Surrogate

- **Discrete Snapshot Residuals**: The PINO loss evaluates continuum PDE residuals at discrete saved trajectory snapshots using numerical finite differences, rather than continuous spatio-temporal coordinates via automatic differentiation.
- **Phase Boundary Derivatives**: Second-order or central temporal differences that span phase transitions (such as dose-to-purge or reaction-to-purge) produce artificial derivative spikes. Residual evaluations must be masked to exclude transition intervals.
- **Held-out Evaluation Units**: Surrogate model accuracy must be evaluated on held-out trajectories using physical units (such as concentration in mol/m^3 and thickness in meters) rather than normalized internal tensor scales.
