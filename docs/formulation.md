# Mathematical Formulation and Physical Modelling

This document outlines the governing equations, nondimensional scaling, numerical discretization, boundary conditions, and explicit physical assumptions implemented in the 2D continuum atomic layer deposition (ALD) simulator and PINO surrogate.

## 1. Domain Geometry and Coordinate System

The simulation domain $\Omega = [0, L_x] \times [0, L_y]$ represents a two-dimensional vertical cross-section of a patterned wafer surface containing a rectangular feature (trench or via).

```
y = 0 ------------------- Inlet (Gas Boundary) -------------------
      |                                                          |
      |                      Gas Phase                           |
      |                                                          |
y = y_top --- Field (Wafer Top)              Field (Wafer Top) ---
             |   \                                       /   |
             |    \               Trench                /    |
             |     |                                   |     |
             |     | Sidewall                 Sidewall |     |
             |     |                                   |     |
y = y_bottom ------|-----------------------------------|------
             |     |               Floor               |     |
             |     -------------------------------------     |
             |                   Substrate                   |
y = L_y      -------------------------------------------------
             x = 0     x = x_left             x = x_right    x = L_x
```

The domain is partitioned into two complementary regions:
- Gas domain $\Omega_{\text{gas}}$: fluid volume accessible to precursor vapor.
- Solid domain $\Omega_{\text{solid}}$: impenetrable substrate and feature walls.

The reactive interface $\Gamma_{\text{surf}} \subset \Omega_{\text{gas}}$ consists of all gas nodes that share at least one face-connected (4-neighbor) edge with a solid substrate node.

### Surface Region Partitioning
The reactive interface $\Gamma_{\text{surf}}$ is partitioned into three mutually disjoint subregions:
1. Field ($\Gamma_{\text{field}}$): planar top surfaces of the wafer on either side of the trench opening ($y \le y_{\text{top}} + 0.5 \Delta y$, with $x < x_{\text{left}}$ or $x > x_{\text{right}}$).
2. Floor ($\Gamma_{\text{floor}}$): bottom horizontal surface of the trench ($y \ge y_{\text{bottom}} - 1.5 \Delta y$, with $x_{\text{left}} - 0.5 \Delta x \le x \le x_{\text{right}} + 0.5 \Delta x$).
3. Sidewall ($\Gamma_{\text{sidewall}}$): vertical trench sidewalls ($\Gamma_{\text{surf}} \setminus (\Gamma_{\text{field}} \cup \Gamma_{\text{floor}})$).

### Aspect Ratio
The aspect ratio $AR$ of the feature is defined as the depth divided by the width:
$$AR = \frac{y_{\text{bottom}} - y_{\text{top}}}{x_{\text{right}} - x_{\text{left}}}$$

In default configuration:
- $L_x = 1.0$, $L_y = 1.0$
- $x_{\text{left}} = 0.375$, $x_{\text{right}} = 0.625$ (width $w = 0.25$)
- $y_{\text{top}} = 0.50$, $y_{\text{bottom}} = 0.95$ (depth $d = 0.45$)
- Default aspect ratio $AR = 0.45 / 0.25 = 1.80$.

A high-aspect-ratio preset is provided via `SimulationConfig.high_aspect_ratio_preset(aspect_ratio=10.0)` which centers a narrow trench of width $w = 0.45 / 10.0 = 0.045$.

---

## 2. Nondimensionalization and Reference Scales

The simulator operates in nondimensional variables. Dimensional quantities (denoted with tildes $\tilde{\cdot}$) are normalized by characteristic reference scales:

| Dimensional Variable | Description | Reference Scale | Nondimensional Form |
| :--- | :--- | :--- | :--- |
| Spatial Coordinates $\tilde{x}, \tilde{y}$ | Trench spatial dimensions | Domain height $L_0$ | $x = \tilde{x}/L_0, \quad y = \tilde{y}/L_0$ |
| Time $\tilde{t}$ | Stage duration | Characteristic time $\tau_0$ | $t = \tilde{t}/\tau_0$ |
| Vapor Concentration $\tilde{C}$ | Precursor molar density | Inlet concentration $C_0$ | $C = \tilde{C}/C_0$ |
| Surface Coverage $\tilde{\theta}$ | Adsorbed precursor fraction | Monolayer site capacity $\theta_{\text{max}}$ | $\theta = \tilde{\theta}/\theta_{\text{max}} \in [0, 1]$ |
| Film Thickness $\tilde{h}$ | Accumulated solid layer | Monolayer scale $h_0$ | $h = \tilde{h}/h_0$ |

### Scaled Kinetic and Transport Parameters
Using reference scales $L_0 \approx 100 \text{ nm}$, $\tau_0 \approx 1 \text{ s}$, $C_0 \approx 10^{-2} \text{ mol/m}^3$, the dimensionless parameters relate to physical properties by:
- Dimensionless Diffusivity: $D = \frac{\tilde{D} \tau_0}{L_0^2}$
- Dimensionless Adsorption Rate: $k_{\text{ads}} = \tilde{k}_{\text{ads}} C_0 \tau_0$
- Dimensionless Desorption Rate: $k_{\text{des}} = \tilde{k}_{\text{des}} \tau_0$
- Dimensionless Reaction Rate: $k_{\text{rxn}} = \tilde{k}_{\text{rxn}} \tau_0$
- Growth Coefficient: $k_{\text{growth}} = \frac{\tilde{k}_{\text{growth}} \theta_{\text{max}}}{h_0}$ (length per unit converted coverage)

---

## 3. Governing Equations

Each ALD cycle consists of cyclic exposure to precursor vapor, inert gas purge, co-reactant exposure, and post-reaction purge.

### 3.1 Precursor Dose Phase
During precursor dosing, gas transport, Langmuir chemisorption, and thermal desorption occur simultaneously:

$$\frac{\partial C}{\partial t} = D \nabla^2 C - R_{\text{ads}}(x, y, t) \quad \text{for } (x, y) \in \Omega_{\text{gas}}$$

$$\frac{\partial \theta}{\partial t} = R_{\text{ads}} - R_{\text{des}} \quad \text{for } (x, y) \in \Gamma_{\text{surf}}$$

where the localized surface reaction rates are:
$$R_{\text{ads}} = k_{\text{ads}} C (1 - \theta) \quad \text{on } \Gamma_{\text{surf}}$$
$$R_{\text{des}} = k_{\text{des}} \theta \quad \text{on } \Gamma_{\text{surf}}$$

### 3.2 Purge Phases
During purge stages, precursor delivery is shut off ($C = 0$ at $y = 0$). Diffusion continues in the gas domain:
$$\frac{\partial C}{\partial t} = D \nabla^2 C \quad \text{for } (x, y) \in \Omega_{\text{gas}}$$

At the conclusion of each purge stage, an operational perfect purge condition is enforced:
$$C(x, y) \equiv 0 \quad \forall (x, y) \in \Omega_{\text{gas}}$$

### 3.3 Co-Reactant Reaction and Film Growth Phase
During the co-reactant exposure stage, chemisorbed precursor molecules $\theta$ react to form the solid thin-film monolayer $h$:

$$\frac{\partial \theta}{\partial t} = - R_{\text{rxn}} = - k_{\text{rxn}} \theta \quad \text{for } (x, y) \in \Gamma_{\text{surf}}$$

$$\frac{\partial h}{\partial t} = k_{\text{growth}} R_{\text{rxn}} = k_{\text{growth}} k_{\text{rxn}} \theta \quad \text{for } (x, y) \in \Gamma_{\text{surf}}$$

---

## 4. Boundary Conditions

1. Top Inlet ($y = 0$):
   - During precursor dose: $C(x, 0, t) = C_0 = 1.0$ for $x \in \Omega_{\text{gas}}$.
   - During purge: $C(x, 0, t) = 0.0$.
2. Lateral Boundaries ($x = 0, x = L_x$):
   - Zero-flux homogeneous Neumann: $\frac{\partial C}{\partial x} = 0$.
3. Bottom Boundary ($y = L_y$):
   - Zero-flux homogeneous Neumann: $\frac{\partial C}{\partial y} = 0$.
4. Substrate Wall Boundaries:
   - Zero-flux Neumann condition: $\mathbf{n} \cdot \nabla C = 0$, where $\mathbf{n}$ is the unit normal to the solid interface. Solid nodes retain $C = 0$.

---

## 5. Numerical Discretization and Stability Criteria

The equations are discretized on a uniform spatial Cartesian grid $(x_i, y_j)$ with step sizes $\Delta x = L_x / (N_x - 1)$ and $\Delta y = L_y / (N_y - 1)$.

### Discrete Laplacian
The 2D Laplacian $\nabla^2 C$ is computed using a 4-neighbor central-difference stencil with internal zero-flux reflection at gas-solid interfaces:
$$(\nabla^2 C)_{i, j} = \frac{C_{i+1, j}^* - 2 C_{i, j} + C_{i-1, j}^*}{\Delta x^2} + \frac{C_{i, j+1}^* - 2 C_{i, j} + C_{i, j-1}^*}{\Delta y^2}$$
where neighbor values $C^*$ in solid cells are replaced by the interior cell value $C_{i, j}$ to enforce $\partial C / \partial n = 0$.

### Forward-Time Central-Space (FTCS) Update
Equations are integrated forward in time using explicit forward Euler time stepping:
$$C^{n+1} = C^n + \Delta t \left( D \nabla^2 C^n - R_{\text{ads}}^n \right)$$
$$\theta^{n+1} = \theta^n + \Delta t \left( R_{\text{ads}}^n - R_{\text{des}}^n \right)$$

### Time Step Selection
Explicit integration is subject to diffusive and kinetic stability constraints:
1. Diffusive von Neumann stability criterion:
   $$\Delta t \le \frac{\min(\Delta x, \Delta y)^2}{4 D}$$
2. Kinetic stability criterion:
   $$\Delta t \le \frac{1}{\max(k_{\text{ads}}, k_{\text{des}}, k_{\text{rxn}})}$$

The solver chooses:
$$\Delta t = 0.20 \times \min\left( \frac{\min(\Delta x, \Delta y)^2}{D}, \frac{1}{\max(k_{\text{ads}}, k_{\text{des}}, k_{\text{rxn}})} \right)$$
which operates with a safety factor of 0.20 (strictly below the 0.25 diffusive limit).

User-supplied $\Delta t$ values are validated against both limits and rejected if unstable.

---

## 6. Modelling Assumptions and Physical Scope

1. **Operator Splitting**: The four ALD phases (dose, purge, reaction, purge) are executed sequentially as decoupled operational stages.
2. **Perfect Purge**: Gas-phase concentration is set to zero ($C \equiv 0$) at the end of each purge. Because the nominal purge duration ($1.0 \text{ s}$) is significantly smaller than the diffusion timescale $\tau_{\text{diff}} = L^2 / D \approx 25\text{--}200 \text{ s}$, complete clearance is an idealized processing assumption rather than a diffusive outcome.
3. **Reaction Stage Isolation**: Chemisorption occurs only during precursor dose. Co-reactant reaction occurs only during the reaction phase. Desorption is inactive during purge.
4. **Mass Non-Conservation**: The precursor sink term $R_{\text{ads}}$ acts as a volumetric gas sink at boundary nodes but is not coupled to absolute surface site densities, and desorption does not return precursor vapor back into the gas domain.
5. **Incomplete Reaction Carryover**: For rate constants $k_{\text{rxn}} \in [0.1, 0.8] \text{ s}^{-1}$ with stage duration $t_{\text{rxn}} = 1.0 \text{ s}$, the fractional consumption is $1 - e^{-k_{\text{rxn}} t_{\text{rxn}}} \in [0.095, 0.551]$. Up to 45% to 90% of adsorbed coverage remains unreacted at the end of a cycle and carries over into subsequent cycles.
6. **Numerical Invariant Clamping**: Bounds $C \ge 0$ and $0 \le \theta \le 1$ are enforced by numerical clamping and verified with activation counters (`clamp_stats`). For standard parameters, zero clamp activations occur.

---

## 7. Metrics and Observables

- **Growth Per Cycle (GPC)**: Increase in mean film thickness on $\Gamma_{\text{surf}}$ over one cycle:
  $$\text{GPC}_k = \bar{h}_k - \bar{h}_{k-1}$$
- **Step Coverage**: Ratio of deposited thickness on trench floor to deposited thickness on top planar field:
  $$\text{Step Coverage} = \frac{\bar{h}_{\text{floor}}}{\bar{h}_{\text{field}}}$$
- **Saturation Coverage**: Mean fractional coverage $\bar{\theta}$ measured at the conclusion of the precursor dose stage, prior to co-reactant consumption.

---

## 8. Neural Operator Surrogate Formulation

The surrogate model $\mathcal{G}_\phi$ parameterizes the mapping from process parameters, continuous time, discrete phase, and geometry to continuous spatial field distributions:

$$\mathcal{G}_\phi: (\mathbf{p}, t, \text{phase}, \text{cycle\_fraction}, x, y, \text{mask}) \mapsto (C, \theta, h)(x, y, t)$$

### 8.1 Snapshot-Independent Formulation
Each trajectory snapshot at time $t$ is predicted independently:
1. Physical process parameters $\mathbf{p} \in \mathbb{R}^5$ (active parameters: $D, k_{\text{ads}}, k_{\text{des}}, k_{\text{rxn}}, t_{\text{pulse}}$) and continuous time $t$ are augmented with a one-hot encoding of the discrete stage index ($\text{phase} \in \{0, 1, 2, 3, 4\}$) and normalized cycle progress ($\text{cycle\_fraction} \in [0, 1]$).
2. Scalar conditionings are spatially broadcast across the 2D grid and concatenated with normalized spatial coordinates $(x, y) \in [-1, 1]^2$ and a binary geometry mask $\mathbf{1}_{\Omega_{\text{gas}}}$.
3. The 15-channel composite representation is lifted via pointwise linear projections into hidden dimension $d_v = 32$, passed through 4 stacked Fourier blocks, and projected to 3 output field channels $(C, \theta, h)$.

### 8.2 Spectral Convolution Kernels
The 2D Fourier layers compute integral kernel transforms using two complex weight tensors $W_1, W_2$:
- $W_1$ acts on positive $y$-frequencies and non-negative $x$-frequencies.
- $W_2$ acts on negative $y$-frequencies and non-negative $x$-frequencies.
High frequencies beyond mode cutoff $k_{\max} = 12$ are truncated, guaranteeing that the layer is real-valued, shape-preserving, and translation-equivariant for periodic inputs.

### 8.3 Loss Weighting and Physics Regularization
The training loss uses per-channel masked mean aggregation:
$$\mathcal{L}_{\text{data}} = \frac{1}{\sum_c w_c} \sum_{c=1}^3 w_c \frac{\sum_{\Omega_c} |y_c - \hat{y}_c|^2}{|\Omega_c|}$$
preventing the thousands of gas nodes from dominating the sparse surface nodes. In PINO training, intra-phase central difference PDE residuals $\mathcal{R}_C, \mathcal{R}_\theta, \mathcal{R}_h$ regularize training without crossing discontinuous phase boundaries.
