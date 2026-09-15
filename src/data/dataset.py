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
}
REQUIRED_COLUMNS = {"sample_id", "lens_id", "raw_path", "processed_path", "pixel_scale"}
PRODUCT_FIELDS = {
    "science_path": "science",
    "mask_path": "invalid_mask",
    "coverage_mask_path": "coverage_mask",
    "background_mask_path": "background_mask",
    "variance_path": "variance",
}


class LensDataset:
    """Validated NumPy-compatible dataset with an optional structured sample view."""

    def __init__(self, manifest_path, root=None, strict_products=False):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent if root is None else Path(root)
        self.metadata = pd.read_csv(self.manifest_path).reset_index(drop=True)
        missing = REQUIRED_COLUMNS.difference(self.metadata.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        if self.metadata["sample_id"].duplicated().any():
            raise ValueError("Manifest contains duplicate sample_id values")
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
        if array.shape != (256, 256):
            errors.append(f"{sample_id}: {field} shape is {array.shape}, expected (256, 256)")
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
                        if array.shape != (256, 256) or array.dtype != bool:
                            errors.append(f"{sample_id}: {field} must be bool with shape (256, 256)")
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
                        if tuple(geometry.get("target_shape", [])) != (256, 256):
                            errors.append(f"{sample_id}: geometry target_shape does not match loader shape")
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
                elif output_name == "variance":
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
            for key in TARGET_FIELDS
            if key in row and not (isinstance(row[key], float) and np.isnan(row[key]))
        }
        targets["availability"] = {
            key: bool(row.get(key, False))
            for key in ("theta_E_target_available", "mass_target_available")
            if key in row
        }
        input_metadata = {
            key: value
            for key, value in row.items()
            if key not in TARGET_FIELDS and key not in PRIVILEGED_FIELDS
        }
        return {
            "inputs": {"products": products, "metadata": input_metadata},
            "targets": targets,
            "metadata": row,
        }
