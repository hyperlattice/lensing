# Dataset Schema: Strong Gravitational Lenses

This document defines the "Dataset Contract" for the lensing project. Every sample used in the machine learning pipeline—whether extracted from real CASTLES data or generated via simulation—must strictly adhere to this schema.

## 1. Sample Schema
A "usable sample" is defined as a record containing the following fields.

| Field Name | Data Type | Requirement | Description | Example Value |
| :--- | :--- | :--- | :--- | :--- |
| `sample_id` | String | **Required** | Unique identifier for an image. | `"Q2237_FIT_NIC2H_q256"` |
| `lens_id` | String | **Required** | Identifier shared by every image from one lens system. | `"Q2237"` |
| `raw_path` | Path | **Required** | Repository-relative path to the original `.fits` file. | `data/raw/castles/Q2237/FIT/NIC2H/q256.fits` |
| `processed_path` | Path | **Required** | Repository-relative path to the standardized `.npy` array. | `data/processed/normalized/Q2237_FIT_NIC2H_q256.npy` |
| `lens_redshift` | Float | Conditional | Redshift of the foreground lens ($z_l$), required for simulation/ML labels. | `0.0395` |
| `source_redshift`| Float | Conditional | Redshift of the background source ($z_s$), required for simulation/ML labels. | `1.695` |
| `pixel_scale` | Float | **Required** | Angular size of one pixel. | `0.045` |
| `orientation` | Float | Optional | Image orientation from `ORIENTAT`, in degrees. | `12.5` |
| `photflam` | Float | Optional | FITS photometric flux calibration keyword. | `1.2e-19` |
| `photfnu` | Float | Optional | FITS photometric flux calibration keyword. | `3.4e-7` |
| `photzpt` | Float | Optional | FITS photometric zero point. | `25.0` |
| `einstein_radius`| Float | Optional | The angular Einstein radius ($\theta_E$). | `1.2` |
| `center_x` | Integer | Optional | X-coordinate of the lens center in pixels. | `128` |
| `center_y` | Integer | Optional | Y-coordinate of the lens center in pixels. | `128` |
| `instrument` | String | Optional | Telescope/Camera used (for PSF lookup). | `"HST-WFC3"` |
| `mask_path` | Path | Optional | Path to the associated bad-pixel mask. | `/data/proc/q2237_mask.npy` |

## 2. Physical Units Registry
To prevent calculation errors, the following units are mandatory across all scripts and data files.

| Measure | Unit | Symbol | Notes |
| :--- | :--- | :--- | :--- |
| **Angles** | Arcseconds | $\text{arcsec}$ | Einstein radius, offsets, and separations. |
| **Pixel Scale** | Arcsec per Pixel | $\text{arcsec/px}$ | WCS-derived scaling factor. |
| **Mass** | Solar Masses | $M_\odot$ | Total mass within the Einstein radius. |
| **Redshift** | Dimensionless | $z$ | Cosmological redshift. |
| **Coordinates** | Pixels | $\text{px}$ | Relative to image top-left $(0,0)$. |

## 3. Validation Rules & Constraints
The `src/data/dataset.py` loader must enforce the following constraints. Samples failing these checks should be logged as "Broken" and excluded from training.

### 3.1 Integrity Constraints
- **Non-Nullity:** `sample_id`, `lens_id`, `raw_path`, `processed_path`, and `pixel_scale` must be present.
- **Path Existence:** `raw_path` and `processed_path` must point to existing files on disk.
- **Shape Consistency:** All files at `processed_path` must resolve to finite NumPy arrays of shape `(256, 256)` and dtype `float32`.
- **Path Convention:** Paths are relative to the repository root and use `/` separators.

### 3.2 Physical Constraints
- **Redshift Logic:** When both labels are available, $0 < \text{lens\_redshift} < \text{source\_redshift}$.
- **Positive Scale:** $\text{pixel\_scale} > 0$.
- **Coordinate Bounds:** $0 \le \text{center\_x} < \text{width}$ and $0 \le \text{center\_y} < \text{height}$.

## 4. Simulation Alignment
When generating synthetic data in Phase 2, the simulator must output labels using the exact keys and units defined in this document. This ensures that the ML model can be trained on synthetic data and validated on real data without any transformation logic in between.