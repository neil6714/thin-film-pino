# Dataset Schema Specification

This document defines the array schemas, axis order, data types, physical units, and metadata structures for datasets produced by the data generation and preparation pipeline.

---

## 1. Raw Dataset Archive (`.npz`)

Raw simulation archives produced by `data/generate.py` contain unnormalized physical arrays, region masks, and per-cycle metrics.

### 1.1 State Arrays

| Array Name | Shape (Trajectory) | Shape (Final State) | Axes | Dtype | Units | Valid Region |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `concentration` | `(N, T, ny, nx)` | `(N, ny, nx)` | `(sample, time, y, x)` | `float32` | $C / C_0$ (nondimensional, $[0, 1]$) | `gas_mask` |
| `surface_coverage` | `(N, T, ny, nx)` | `(N, ny, nx)` | `(sample, time, y, x)` | `float32` | $\theta / \theta_{\text{max}}$ (nondimensional, $[0, 1]$) | `surface_mask` |
| `film_thickness` | `(N, T, ny, nx)` | `(N, ny, nx)` | `(sample, time, y, x)` | `float32` | $h / h_0$ (nondimensional, $\ge 0$) | `surface_mask` |
| `time` | `(N, T)` | `(N,)` | `(sample, time)` | `float32` | $t / \tau_0$ (nondimensional, $\ge 0$) | Global |
| `cycle` | `(N, T)` | `(N,)` | `(sample, time)` | `int16` | Cycle index ($1 \dots N_{\text{cycles}}$, $0$ at initial state) | Global |
| `phase` | `(N, T)` | `(N,)` | `(sample, time)` | `int8` | Phase code ($0, 1, 2, 3, 4$) | Global |
| `parameters` | `(N, 8)` | `(N, 8)` | `(sample, parameter)` | `float32` | Mixed (see Section 1.2) | Global |

### 1.2 Parameter Vector Columns

The `parameters` array stores 8 parameters per sample in the following order:

| Index | Name | Dtype | Range / Value | Units | Description |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 0 | `diffusivity` | `float32` | $[1.0 \times 10^{-3}, 8.0 \times 10^{-3}]$ | Nondimensional ($D = \tilde{D}\tau_0 / L_0^2$) | Vapor diffusivity in gas domain |
| 1 | `k_ads` | `float32` | $[0.5, 1.5]$ | $1/\text{time}$ | First-order chemisorption rate constant |
| 2 | `k_des` | `float32` | $[0.02, 0.10]$ | $1/\text{time}$ | First-order thermal desorption rate constant |
| 3 | `k_rxn` | `float32` | $[0.1, 0.8]$ | $1/\text{time}$ | Co-reactant reaction rate constant |
| 4 | `k_growth` | `float32` | $0.02$ (constant) | Length per converted coverage | Thickness conversion coefficient |
| 5 | `pulse_time` | `float32` | $[10.0, 100.0]$ | Time | Precursor exposure stage duration |
| 6 | `purge_time` | `float32` | $1.0$ (constant) | Time | Inert gas purge duration |
| 7 | `reaction_time` | `float32` | $1.0$ (constant) | Time | Co-reactant exposure stage duration |

Indices 0, 1, 2, 3, and 5 vary across samples and form the active conditioning parameters. Indices 4, 6, and 7 are constant and omitted from network conditioning.

### 1.3 Phase Codes

The `phase` array identifies the operational ALD stage that ends at or generated the snapshot:

| Code | Phase Name | Physical Activity | Concentration at Wall | Desorption Active? |
| :---: | :--- | :--- | :--- | :---: |
| 0 | `initial` | Domain before cycle 1 ($t = 0$) | $C = 0$ | No |
| 1 | `dose` | Precursor vapor exposure and chemisorption | $C(x, 0) = 1.0$ | Yes |
| 2 | `post_dose_purge` | Inert purge clearing residual precursor | $C \to 0$, reset to $0$ at end | No |
| 3 | `reaction` | Co-reactant exposure and surface conversion | $C = 0$ | No |
| 4 | `post_rxn_purge` | Inert purge clearing reaction byproducts | $C \to 0$, reset to $0$ at end | No |

### 1.4 Geometry and Mask Arrays

Spatial masks are 2D boolean arrays of shape `(ny, nx)`:

| Array Name | Dtype | Definition |
| :--- | :--- | :--- |
| `solid_mask` | `bool` | True on solid substrate and trench walls; False in gas domain |
| `gas_mask` | `bool` | True in fluid volume accessible to precursor vapor; False in substrate |
| `surface_mask` | `bool` | Gas nodes sharing a 4-neighbor interface with solid substrate |
| `field_mask` | `bool` | Reactive surface nodes on wafer planar top ($y \le y_{\text{top}}$) |
| `sidewall_mask` | `bool` | Reactive surface nodes on vertical trench sidewalls |
| `floor_mask` | `bool` | Reactive surface nodes on horizontal trench bottom ($y \ge y_{\text{bottom}}$) |
| `x` | `float32` | 1D horizontal coordinates $[0, L_x]$ of length `nx` |
| `y` | `float32` | 1D vertical coordinates $[0, L_y]$ of length `ny` |
| `parameter_names` | `str` | 1D array of 8 parameter strings |
| `phase_names` | `str` | 1D array of 5 phase strings |

### 1.5 Cycle Metric Arrays

Per-cycle metrics have shape `(N, num_cycles)` and dtype `float32`:

| Metric Key | Description |
| :--- | :--- |
| `metric_gpc` | Growth per cycle: $\bar{h}_k - \bar{h}_{k-1}$ averaged over $\Gamma_{\text{surf}}$ |
| `metric_mean_thickness` | Mean accumulated thickness on $\Gamma_{\text{surf}}$ at cycle conclusion |
| `metric_field_thickness` | Mean accumulated thickness on planar field $\Gamma_{\text{field}}$ |
| `metric_floor_thickness` | Mean accumulated thickness on trench floor $\Gamma_{\text{floor}}$ |
| `metric_sidewall_thickness` | Mean accumulated thickness on trench sidewalls $\Gamma_{\text{sidewall}}$ |
| `metric_step_coverage` | Ratio $\bar{h}_{\text{floor}} / \bar{h}_{\text{field}}$ |
| `metric_dose_coverage` | Mean surface coverage $\bar{\theta}$ at conclusion of precursor dose |
| `metric_surface_coverage` | Mean surface coverage $\bar{\theta}$ at conclusion of full cycle |

---

## 2. Prepared Dataset Archive (`.npz`)

Prepared archives produced by `scripts/prepare_temporal_dataset.py` standardize fields and parameters using training-set statistics and attach partition indices.

### 2.1 Standardized Fields

Fields are standardized with masked zero-padding:

$$u_{\text{norm}}(x, y) = \begin{cases} \frac{u(x, y) - \mu_u}{\sigma_u} & (x, y) \in \Omega_{\text{active}} \\ 0 & (x, y) \notin \Omega_{\text{active}} \end{cases}$$

Because $0 \cdot \sigma + \mu = \mu \ne 0$, cells outside the active mask are explicitly re-zeroed during denormalization.

### 2.2 Split Indices

| Key | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `train_indices` | `(N_train,)` | `int64` | In-distribution training trajectories |
| `val_indices` | `(N_val,)` | `int64` | In-distribution validation trajectories |
| `test_indices` | `(N_test,)` | `int64` | In-distribution test trajectories |
| `ood_indices` | `(N_ood,)` | `int64` | Out-of-distribution test trajectories (longest pulse times) |

Splits are mutually disjoint and cover all samples:
$$\Omega_{\text{all}} = \Omega_{\text{train}} \cup \Omega_{\text{val}} \cup \Omega_{\text{test}} \cup \Omega_{\text{ood}}$$

### 2.3 Normalization Statistics

| Key | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `field_mean` | `(3,)` | `float32` | Means $[\mu_C, \mu_\theta, \mu_h]$ computed on active masks of training split |
| `field_std` | `(3,)` | `float32` | Standard deviations $[\sigma_C, \sigma_\theta, \sigma_h]$ computed on active masks |
| `parameter_mean` | `(5,)` | `float32` | Means of active parameters ($D, k_{\text{ads}}, k_{\text{des}}, k_{\text{rxn}}, t_{\text{pulse}}$) on training split |
| `parameter_std` | `(5,)` | `float32` | Standard deviations of active parameters on training split |
| `time_mean` | Scalar | `float32` | Mean time coordinate across training trajectories |
| `time_std` | Scalar | `float32` | Standard deviation of time coordinate across training trajectories |

---

## 3. Metadata Descriptor (`.json`)

Each dataset archive is accompanied by a JSON descriptor with the same base name:

```json
{
  "schema_version": "2.0.0",
  "mode": "trajectory",
  "created_at": "2026-11-12T17:00:00+00:00",
  "git_commit": "3485e11",
  "package_versions": {
    "numpy": "2.2.0",
    "torch": "2.6.0",
    "thin_film_pino": "0.1.0"
  },
  "config_hash": "a1b2c3d4e5f60718",
  "num_samples": 64,
  "num_cycles": 10,
  "snapshots_per_phase": 8,
  "grid": {
    "nx": 80,
    "ny": 60,
    "lx": 1.0,
    "ly": 1.0
  },
  "archive_path": "data/transient_ald_dataset.npz"
}
```
