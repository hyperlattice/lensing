import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.processing.psf import validate_psf


TARGET_FIELDS = {
    "theta_E_arcsec",
    "einstein_radius_estimate_arcsec",
    "einstein_radius_arcsec",
    "mass_enclosed_msun",
    "mass_aperture_arcsec",
    "einstein_radius_method",
    "q",
    "phi_rad",
    "e1",
    "e2",
    "gamma1",
    "gamma2",
}
PRIVILEGED_FIELDS = {
    "mass_model",
    "q",
    "phi_rad",
    "e1",
    "e2",
    "gamma1",
    "gamma2",
    "source_x_arcsec",
    "source_y_arcsec",
    "source_re_arcsec",
    "source_n",
    "simulator_name",
    "simulator_version",
    "parameter_seed",
    "noise_seed",
    "parameters_path",
    "parent_system_id",
    "parent_real_lens_id",
    "split_group_id",
    "source_template_id",
    "background_template_id",
    "lens_center_x_arcsec",
    "lens_center_y_arcsec",
    "source_q",
    "source_phi_rad",
    "source_amp",
    "source_flux",
    "lens_light_q",
    "lens_light_phi_rad",
    "lens_light_n",
    "lens_light_re_arcsec",
    "lens_light_amp",
    "point_source_amp",
    "point_positions_x_arcsec",
    "point_positions_y_arcsec",
    "point_amplitudes",
    "stress_case",
    "image_count_physical",
    "image_count_detectable",
    "lens_equation_max_residual_arcsec",
    "config_hash",
}
MODEL_INPUT_METADATA_FIELDS = {
    "domain",
    "image_unit",
    "pixel_scale",
    "pixel_scale_x",
    "pixel_scale_y",
    "instrument",
    "detector",
    "filter_name",
    "exposure_s",
    "masked_fraction",
    "normalization_method",
    "variance_method",
    "variance_status",
    "variance_unit",
    "noise_correlation_assumption",
    "psf_id",
    "psf_fwhm_arcsec",
    "exposure_time",
    "background_rate",
    "read_noise_electrons",
}
REQUIRED_COLUMNS = {"sample_id", "lens_id", "raw_path", "processed_path", "pixel_scale"}
PRODUCT_FIELDS = {
    "science_path": "science",
    "mask_path": "invalid_mask",
    "coverage_mask_path": "coverage_mask",
    "background_mask_path": "background_mask",
    "variance_path": "variance",
    "expected_variance_path": "expected_variance",
}


class LensDataset:
    """Validated NumPy-compatible dataset with an optional structured sample view."""

    def __init__(
        self,
        manifest_path,
        root=None,
        strict_products=False,
        *,
        split=None,
        domain=None,
        eligible_for=None,
        target_fields=None,
        expected_shape=(256, 256),
    ):
        manifest_path = Path(manifest_path)
        if manifest_path.is_dir():
            manifest_path = manifest_path / "manifest.csv"
        manifest_root = manifest_path.parent
        if split is not None:
            split_path = manifest_path.parent / "splits" / f"{split}.csv"
            if not split_path.exists():
                raise FileNotFoundError(f"Split manifest does not exist: {split_path}")
            manifest_path = split_path
        self.manifest_path = manifest_path
        self.root = manifest_root if root is None else Path(root)
        self.expected_shape = None if expected_shape is None else tuple(int(value) for value in expected_shape)
        if self.expected_shape is not None and (len(self.expected_shape) != 2 or min(self.expected_shape) <= 0):
            raise ValueError("expected_shape must contain two positive dimensions or be None")
        self.metadata = pd.read_csv(self.manifest_path).reset_index(drop=True)
        missing = REQUIRED_COLUMNS.difference(self.metadata.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        if self.metadata["sample_id"].duplicated().any():
            raise ValueError("Manifest contains duplicate sample_id values")
        if domain is not None:
            if "domain" not in self.metadata:
                raise ValueError("Cannot filter by domain: manifest has no domain column")
            self.metadata = self.metadata.loc[self.metadata["domain"] == domain].reset_index(drop=True)
        if eligible_for is not None:
            if "eligible_for" not in self.metadata:
                raise ValueError("Cannot filter by eligibility: manifest has no eligible_for column")
            self.metadata = self.metadata.loc[
                self.metadata["eligible_for"].fillna("").astype(str).str.split(";").map(
                    lambda values: str(eligible_for) in values
                )
            ].reset_index(drop=True)
        self.target_fields = set(TARGET_FIELDS if target_fields is None else target_fields)
        unknown_targets = self.target_fields.difference(TARGET_FIELDS)
        if unknown_targets:
            raise ValueError(f"Unknown target fields: {sorted(unknown_targets)}")
        unavailable_targets = self.target_fields.difference(self.metadata.columns)
        if target_fields is not None and unavailable_targets:
            raise ValueError(f"Target fields are absent from the manifest: {sorted(unavailable_targets)}")
        if strict_products:
            errors = self.validate_all(require_products=True)
            if errors:
                raise ValueError("Invalid dataset manifest:\n" + "\n".join(errors))

    def __len__(self):
        return len(self.metadata)

    def _row(self, index):
        if index < 0:
            index += len(self.metadata)
        if index < 0 or index >= len(self.metadata):
            raise IndexError(index)
        return self.metadata.iloc[index].to_dict()

    def _path(self, value):
        path = Path(str(value))
        return path if path.is_absolute() else self.root / path

    def _load_array(self, row, field, required=False):
        value = row.get(field)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            if required:
                raise ValueError(f"{row['sample_id']}: missing {field}")
            return None
        path = self._path(value)
        if not path.exists():
            if required:
                raise ValueError(f"{row['sample_id']}: missing {field} file {path}")
            return None
        return np.load(path)

    def _validate_image(self, array, sample_id, field, dtype=np.float32):
        if array is None:
            return [f"{sample_id}: {field} is missing"]
        errors = []
        if self.expected_shape is None:
            if array.ndim != 2:
                errors.append(f"{sample_id}: {field} must be two-dimensional, found shape {array.shape}")
        elif array.shape != self.expected_shape:
            errors.append(f"{sample_id}: {field} shape is {array.shape}, expected {self.expected_shape}")
        if dtype is not None and array.dtype != dtype:
            errors.append(f"{sample_id}: {field} dtype is {array.dtype}, expected {dtype}")
        if not np.isfinite(array).all():
            errors.append(f"{sample_id}: {field} contains NaN or infinity")
        return errors

    def validate_all(self, require_products=False):
        """Return row-level product errors without modifying files."""
        errors = []
        for row in self.metadata.to_dict(orient="records"):
            sample_id = row["sample_id"]
            for path_field in ("raw_path", "processed_path"):
                path = self._path(row.get(path_field))
                if not path.exists():
                    errors.append(f"{sample_id}: missing {path_field} file {path}")
            image = self._load_array(row, "processed_path", required=False)
            errors.extend(self._validate_image(image, sample_id, "processed_path"))
            if require_products:
                for field in ("science_path", "mask_path", "coverage_mask_path", "background_mask_path", "variance_path"):
                    if row.get(field) is None or (isinstance(row.get(field), float) and np.isnan(row.get(field))):
                        errors.append(f"{sample_id}: missing required product field {field}")
                        continue
                    array = self._load_array(row, field, required=False)
                    if array is None:
                        errors.append(f"{sample_id}: missing {field} file")
                        continue
                    if field in {"mask_path", "coverage_mask_path", "background_mask_path"}:
                        if array.shape != image.shape or array.dtype != bool:
                            errors.append(f"{sample_id}: {field} must be bool with the image shape {image.shape}")
                        continue
                    errors.extend(self._validate_image(array, sample_id, field))
                variance = self._load_array(row, "variance_path", required=False)
                combined_mask = None
                mask_values = [self._load_array(row, field, required=False) for field in ("mask_path", "coverage_mask_path")]
                if all(value is not None for value in mask_values):
                    combined_mask = np.logical_or.reduce(mask_values)
                if variance is not None and combined_mask is not None:
                    if not np.isfinite(variance[~combined_mask]).all() or not (variance[~combined_mask] > 0).all():
                        errors.append(f"{sample_id}: variance must be finite and positive on valid pixels")
                elif variance is not None and (not np.isfinite(variance).all() or not (variance > 0).all()):
                    errors.append(f"{sample_id}: variance must be finite and positive")
            geometry_value = row.get("geometry_transform_path")
            if require_products and geometry_value is not None and not (isinstance(geometry_value, float) and np.isnan(geometry_value)):
                geometry_path = self._path(geometry_value)
                if not geometry_path.exists():
                    errors.append(f"{sample_id}: missing geometry_transform_path file")
                else:
                    try:
                        geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
                        if tuple(geometry.get("target_shape", [])) != image.shape:
                            errors.append(f"{sample_id}: geometry target_shape does not match image shape")
                    except Exception as exc:
                        errors.append(f"{sample_id}: invalid geometry transform: {exc}")
        return errors

    def __getitem__(self, index):
        row = self._row(index)
        image = self._load_array(row, "processed_path", required=True)
        errors = self._validate_image(image, row["sample_id"], "processed_path")
        if errors:
            raise ValueError("; ".join(errors))
        return image, row

    def get_sample(self, index):
        """Return separated model inputs, targets, provenance metadata, and products."""
        row = self._row(index)
        image = self._load_array(row, "processed_path", required=True)
        errors = self._validate_image(image, row["sample_id"], "processed_path")
        if errors:
            raise ValueError("; ".join(errors))

        products = {"image": image}
        for field, output_name in PRODUCT_FIELDS.items():
            array = self._load_array(row, field, required=False)
            if array is not None:
                if output_name.endswith("mask") or output_name in {"invalid_mask", "coverage_mask", "background_mask"}:
                    if array.shape != image.shape or array.dtype != bool:
                        raise ValueError(f"{row['sample_id']}: {field} must be bool with image shape")
                elif output_name in {"variance", "expected_variance"}:
                    if array.shape != image.shape:
                        raise ValueError(f"{row['sample_id']}: variance shape does not match image")
                else:
                    product_errors = self._validate_image(array, row["sample_id"], output_name)
                    if product_errors:
                        raise ValueError("; ".join(product_errors))
                products[output_name] = array

        variance = products.get("variance")
        masks = [products.get(name) for name in ("invalid_mask", "coverage_mask")]
        if variance is not None:
            combined_mask = np.logical_or.reduce([mask for mask in masks if mask is not None]) if any(mask is not None for mask in masks) else np.zeros(image.shape, dtype=bool)
            if not np.isfinite(variance[~combined_mask]).all() or not (variance[~combined_mask] > 0).all():
                raise ValueError(f"{row['sample_id']}: variance must be finite and positive on valid pixels")

        geometry = None
        geometry_value = row.get("geometry_transform_path")
        if geometry_value is not None and not (isinstance(geometry_value, float) and np.isnan(geometry_value)):
            geometry = json.loads(self._path(geometry_value).read_text(encoding="utf-8"))
            if tuple(geometry.get("target_shape", [])) != tuple(image.shape):
                raise ValueError(f"{row['sample_id']}: geometry target shape does not match image")
        products["geometry"] = geometry

        psf = None
        psf_value = row.get("psf_path")
        if psf_value is not None and not (isinstance(psf_value, float) and np.isnan(psf_value)):
            psf = self._load_array(row, "psf_path", required=True)
            psf_errors = validate_psf(type("PSFRecord", (), {"values": psf, "pixel_scale": float(row.get("pixel_scale", 0.0))})())
            if psf_errors:
                raise ValueError(f"{row['sample_id']}: invalid PSF: {'; '.join(psf_errors)}")
        products["psf"] = psf

        targets = {
            key: row[key]
            for key in self.target_fields
            if key in row and not (isinstance(row[key], float) and np.isnan(row[key]))
        }
        targets["availability"] = {
            key: bool(row.get(key, False))
            for key in ("theta_E_target_available", "mass_target_available")
            if key in row
        }
        # A positive allowlist makes it impossible for a newly added truth or
        # provenance column to silently become a model feature.
        input_metadata = {key: row[key] for key in MODEL_INPUT_METADATA_FIELDS if key in row}
        return {
            "inputs": {"products": products, "metadata": input_metadata},
            "targets": targets,
            "metadata": row,
        }
