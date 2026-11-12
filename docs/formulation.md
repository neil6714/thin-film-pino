# Mathematical Formulation and Physical Modeling

This document details the governing equations per ALD stage, nondimensional reference scaling, numerical discretization, boundary conditions, modeling assumptions, and neural operator architecture.

---

## 1. Domain Geometry and Coordinate System

The domain $\Omega = [0, L_x] \times [0, L_y]$ represents a 2D vertical cross-section of a patterned wafer with a rectangular feature:

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

The domain is partitioned into:
- Gas domain $\Omega_{\text{gas}}$: region accessible to vapor.
- Solid domain $\Omega_{\text{solid}}$: substrate and feature walls.
- Reactive interface $\Gamma_{\text{surf}} \subset \Omega_{\text{gas}}$: gas nodes sharing a 4-neighbor face with a solid node.

### 1.1 Surface Region Partitioning
$\Gamma_{\text{surf}}$ is partitioned into three disjoint subregions:
1. Field ($\Gamma_{\text{field}}$): planar top surfaces outside the trench mouth ($y \le y_{\text{top}} + 0.5 \Delta y$, with $x < x_{\text{left}}$ or $x > x_{\text{right}}$).
2. Floor ($\Gamma_{\text{floor}}$): bottom horizontal surface of the trench ($y \ge y_{\text{bottom}} - 1.5 \Delta y$, with $x_{\text{left}} - 0.5 \Delta x \le x \le x_{\text{right}} + 0.5 \Delta x$).
3. Sidewall ($\Gamma_{\text{sidewall}}$): vertical trench sidewalls ($\Gamma_{\text{surf}} \setminus (\Gamma_{\text{field}} \cup \Gamma_{\text{floor}})$).

### 1.2 Aspect Ratio
The aspect ratio $AR$ is the trench depth divided by its width:
$$AR = \frac{y_{\text{bottom}} - y_{\text{top}}}{x_{\text{right}} - x_{\text{left}}}$$

Default configuration:
- $L_x = 1.0, L_y = 1.0$
- $x_{\text{left}} = 0.375, x_{\text{right}} = 0.625$ (width $w = 0.25$)
- $y_{\text{top}} = 0.50, y_{\text{bottom}} = 0.95$ (depth $d = 0.45$)
- Aspect ratio $AR = 0.45 / 0.25 = 1.80$.

A preset with $AR = 10.0$ is provided via `SimulationConfig.high_aspect_ratio_preset(aspect_ratio=10.0)`, centering a trench with width $w = 0.45 / 10.0 = 0.045$.

---

## 2. Nondimensionalization and Reference Scales

The simulator operates in nondimensional variables. Dimensional variables (denoted with tildes $\tilde{\cdot}$) relate to nondimensional variables by characteristic scales:

| Variable | Description | Reference Scale | Dimensionless Variable |
| :--- | :--- | :--- | :--- |
| Spatial Coordinates $\tilde{x}, \tilde{y}$ | Trench dimensions | Domain height $L_0 \approx 100\text{ nm}$ | $x = \tilde{x}/L_0, \quad y = \tilde{y}/L_0$ |
| Time $\tilde{t}$ | Stage duration | Scale $\tau_0 \approx 1\text{ s}$ | $t = \tilde{t}/\tau_0$ |
| Vapor Concentration $\tilde{C}$ | Precursor molar density | Inlet concentration $C_0 \approx 10^{-2}\text{ mol/m}^3$ | $C = \tilde{C}/C_0$ |
| Surface Coverage $\tilde{\theta}$ | Adsorbed precursor fraction | Monolayer site capacity $\theta_{\text{max}}$ | $\theta = \tilde{\theta}/\theta_{\text{max}} \in [0, 1]$ |
| Film Thickness $\tilde{h}$ | Accumulated solid layer | Monolayer scale $h_0$ | $h = \tilde{h}/h_0$ |

Dimensionless kinetic and transport parameters:
- Effective Diffusivity: $D = \tilde{D} \tau_0 / L_0^2$
- Adsorption Rate Constant: $k_{\text{ads}} = \tilde{k}_{\text{ads}} C_0 \tau_0$
- Desorption Rate Constant: $k_{\text{des}} = \tilde{k}_{\text{des}} \tau_0$
- Reaction Rate Constant: $k_{\text{rxn}} = \tilde{k}_{\text{rxn}} \tau_0$
- Growth Coefficient: $k_{\text{growth}} = \tilde{k}_{\text{growth}} \theta_{\text{max}} / h_0$ (length per converted coverage)

---

## 3. Governing Equations by Stage

Instead of a single combined equation, the process is operator-split into four distinct operational stages per cycle.

### 3.1 Precursor Dose Stage (Phase 1)
During precursor dosing, vapor diffusion, Langmuir chemisorption, and thermal desorption occur simultaneously. Thermal desorption is active in this phase:

$$\frac{\partial C}{\partial t} = D \nabla^2 C - R_{\text{ads}}(x, y, t) \quad \text{in } \Omega_{\text{gas}}$$
$$\frac{\partial \theta}{\partial t} = R_{\text{ads}} - R_{\text{des}} \quad \text{on } \Gamma_{\text{surf}}$$
$$\frac{\partial h}{\partial t} = 0 \quad \text{on } \Gamma_{\text{surf}}$$

where reaction rates are:
$$R_{\text{ads}} = k_{\text{ads}} C (1 - \theta) \quad \text{on } \Gamma_{\text{surf}}$$
$$R_{\text{des}} = k_{\text{des}} \theta \quad \text{on } \Gamma_{\text{surf}}$$

Boundary conditions during dose:
- Top inlet: $C(x, 0, t) = 1.0$
- Lateral and substrate boundaries: $\mathbf{n} \cdot \nabla C = 0$

### 3.2 Post-Dose Inert Purge Stage (Phase 2)
Precursor delivery is shut off. Thermal desorption is inactive during purge:

$$\frac{\partial C}{\partial t} = D \nabla^2 C \quad \text{in } \Omega_{\text{gas}}$$
$$\frac{\partial \theta}{\partial t} = 0 \quad \text{on } \Gamma_{\text{surf}}$$
$$\frac{\partial h}{\partial t} = 0 \quad \text{on } \Gamma_{\text{surf}}$$

Boundary conditions:
- Top inlet: $C(x, 0, t) = 0.0$
- Lateral and substrate boundaries: $\mathbf{n} \cdot \nabla C = 0$

**Purge Reset**: At the end of the post-dose purge stage, precursor concentration is reset to zero across the entire domain:
$$C(x, y) \equiv 0 \quad \forall (x, y) \in \Omega_{\text{gas}}$$

### 3.3 Co-Reactant Reaction and Growth Stage (Phase 3)
Precursor vapor is absent ($C \equiv 0$). Chemisorbed precursor reacts with co-reactant to form the solid film:

$$C(x, y, t) \equiv 0 \quad \text{in } \Omega_{\text{gas}}$$
$$\frac{\partial \theta}{\partial t} = - R_{\text{rxn}} = - k_{\text{rxn}} \theta \quad \text{on } \Gamma_{\text{surf}}$$
$$\frac{\partial h}{\partial t} = k_{\text{growth}} R_{\text{rxn}} = k_{\text{growth}} k_{\text{rxn}} \theta \quad \text{on } \Gamma_{\text{surf}}$$

### 3.4 Post-Reaction Inert Purge Stage (Phase 4)
Any reaction byproducts are purged:

$$\frac{\partial C}{\partial t} = 0, \quad \frac{\partial \theta}{\partial t} = 0, \quad \frac{\partial h}{\partial t} = 0$$

At the end of this stage, $C \equiv 0$ is re-enforced prior to beginning the next cycle.

---

## 4. Numerical Discretization and Stability Criteria

Spatial discretization uses a uniform Cartesian grid with step sizes $\Delta x = L_x / (N_x - 1)$ and $\Delta y = L_y / (N_y - 1)$.

### 4.1 Explicit Forward-Time Central-Space (FTCS) Scheme
The 2D diffusion equation is discretized using a 4-neighbor central-difference stencil:
$$(\nabla^2 C)_{i, j} = \frac{C_{i+1, j}^* - 2 C_{i, j} + C_{i-1, j}^*}{\Delta x^2} + \frac{C_{i, j+1}^* - 2 C_{i, j} + C_{i, j-1}^*}{\Delta y^2}$$
where neighbor values $C^*$ in solid cells are replaced by $C_{i, j}$ to enforce zero-flux boundary conditions $\mathbf{n} \cdot \nabla C = 0$.

### 4.2 Stability Limits
Explicit FTCS integration is subject to two stability limits:
1. Diffusive stability limit:
   $$\Delta t \le \frac{\min(\Delta x, \Delta y)^2}{4 D}$$
2. Reaction kinetic limit:
   $$\Delta t \le \frac{1}{\max(k_{\text{ads}}, k_{\text{des}}, k_{\text{rxn}})}$$

The solver chooses:
$$\Delta t = 0.20 \times \min\left( \frac{\min(\Delta x, \Delta y)^2}{D_{\max}}, \frac{1}{\max(k)} \right)$$
User-supplied $\Delta t$ values exceeding $\min(\Delta x, \Delta y)^2 / (4D)$ or $1 / \max(k)$ are rejected with a ValueError.

---

## 5. Parameter Distributions and Diffusion Timescale

Parameters follow `configs/priors.yaml`:

| Parameter | Symbol | Distribution | Range / Value |
| :--- | :---: | :--- | :--- |
| Diffusivity | $D$ | Uniform | $[1.0 \times 10^{-3}, 8.0 \times 10^{-3}]$ |
| Adsorption Rate | $k_{\text{ads}}$ | Uniform | $[0.5, 1.5]$ |
| Desorption Rate | $k_{\text{des}}$ | Uniform | $[0.02, 0.10]$ |
| Reaction Rate | $k_{\text{rxn}}$ | Uniform | $[0.1, 0.8]$ |
| Growth Coefficient | $k_{\text{growth}}$ | Constant | $0.02$ |
| Dose Exposure Time | $t_{\text{pulse}}$ | Bimodal mixture | $85\%\; [10.0, 60.0], \quad 15\%\; [60.0, 100.0]$ |
| Purge Duration | $t_{\text{purge}}$ | Constant | $1.0$ |
| Reaction Duration | $t_{\text{rxn}}$ | Constant | $1.0$ |

### Why Pulse Range is 10 to 100 s
The characteristic diffusion time across the trench domain is:
$$\tau_{\text{diff}} = \frac{L^2}{D}$$
For $L = 1.0$ and $D \in [1.0 \times 10^{-3}, 8.0 \times 10^{-3}]$, this yields $\tau_{\text{diff}} \approx 25\text{ to } 200\text{ s}$.
The pulse exposure range of 10 to 100 s directly spans this regime, transitioning from diffusion-limited starved profiles at short pulses to saturation at long pulses.

---

## 6. Neural Operator Architecture

```
Conditioning Channels (15 channels)
  - Active Parameters [D, k_ads, k_des, k_rxn, t_pulse] (5 channels)
  - Time t (1 channel)
  - Phase Code [one-hot] (5 channels)
  - Cycle Fraction (1 channel)
  - Coordinates (x, y) (2 channels)
  - Geometry Gas Mask (1 channel)
                   │
                   ▼
     Two-Layer Pointwise Lifting Network
       Conv2d(15 -> width, 1x1) + GELU + Conv2d(width -> width, 1x1)
                   │
                   ▼
     Stacked 2D Fourier Blocks (4 blocks)
       - SpectralConv2d with two complex weight blocks:
           W1: positive y-frequencies (0 to modes_y - 1)
           W2: negative y-frequencies (ny - modes_y to ny - 1)
       - Pointwise Conv2d(width -> width, 1x1)
       - GELU Activation
                   │
                   ▼
     Pointwise Projection Network
       Conv2d(width -> 64, 1x1) + GELU + Conv2d(64 -> 3, 1x1)
                   │
                   ▼
     Output Multi-Field Tensor: [C, θ, h](x, y, t)
```

Each snapshot is predicted independently. The spectral convolution uses two weight blocks ($W_1$ and $W_2$) to handle positive and negative spatial frequencies along the non-Hermitian axis of the 2D real FFT.

---

## 7. Related Work

1. **Fourier Neural Operator (FNO)**:
   Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2021). Fourier Neural Operator for Parametric Partial Differential Equations. In *International Conference on Learning Representations (ICLR 2021)*.
2. **Physics-Informed Neural Operator (PINO)**:
   Li, Z., Zheng, H., Kovachki, N., Jin, D., Chen, H., Liu, B., Azizzadenesheli, K., & Anandkumar, A. (2024). Physics-Informed Neural Operator for Learning Partial Differential Equations. *ACM/IMS Journal of Data Science*, 1(3), 1–28.
3. **ALD Kinetics in Narrow Trenches**:
   Gordon, R. G., Hausmann, D., Kim, E., & Shepard, J. (2003). A Kinetic Model for Step Coverage by Atomic Layer Deposition in Narrow Holes or Trenches. *Chemical Vapor Deposition*, 9(2), 73–78.
4. **Diffusion-Reaction Modeling in Nanostructures**:
   Yanguas-Gil, A., & Elam, J. W. (2012). Self-Limited Reaction-Diffusion in Nanostructured Substrates: Surface Coverage Dynamics and Analytic Approximations to ALD Saturation Times. *Chemical Vapor Deposition*, 18(1–3), 46–52.
