"""Pure image geometry operations shared by data and mask products."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CropPadTransform:
    """Exact center crop/pad mapping for arrays indexed as ``array[y, x]``."""

    original_shape: tuple[int, int]
    target_shape: tuple[int, int]
    source_y_start: int
    source_y_end: int
    source_x_start: int
    source_x_end: int
    target_y_start: int
    target_y_end: int
    target_x_start: int
    target_x_end: int

    @classmethod
    def from_shapes(cls, original_shape, target_shape):
        if len(original_shape) != 2 or len(target_shape) != 2:
            raise ValueError("Crop/pad shapes must be two-dimensional")
        source_height, source_width = (int(value) for value in original_shape)
        target_height, target_width = (int(value) for value in target_shape)
        if min(source_height, source_width, target_height, target_width) <= 0:
            raise ValueError("Crop/pad shapes must be positive")

        source_y_start = max((source_height - target_height) // 2, 0)
        source_x_start = max((source_width - target_width) // 2, 0)
        source_y_end = min(source_y_start + target_height, source_height)
        source_x_end = min(source_x_start + target_width, source_width)
        target_y_start = max((target_height - source_height) // 2, 0)
        target_x_start = max((target_width - source_width) // 2, 0)
        target_y_end = target_y_start + (source_y_end - source_y_start)
        target_x_end = target_x_start + (source_x_end - source_x_start)
        return cls(
            (source_height, source_width),
            (target_height, target_width),
            source_y_start,
            source_y_end,
            source_x_start,
            source_x_end,
            target_y_start,
            target_y_end,
            target_x_start,
            target_x_end,
        )

    @property
    def output_to_input_offset_xy(self):
        """Return ``(x_offset, y_offset)`` for output-to-input coordinates."""
        return (
            self.source_x_start - self.target_x_start,
            self.source_y_start - self.target_y_start,
        )

    def apply(self, array, pad_value=0):
        array = np.asarray(array)
        if array.ndim != 2 or tuple(array.shape) != self.original_shape:
            raise ValueError(f"Expected array with shape {self.original_shape}, got {array.shape}")
        output = np.full(self.target_shape, pad_value, dtype=array.dtype)
        output[self.target_y_start:self.target_y_end, self.target_x_start:self.target_x_end] = array[
            self.source_y_start:self.source_y_end,
            self.source_x_start:self.source_x_end,
        ]
        return output

    def output_to_input(self, y, x):
        """Map an output pixel-center coordinate to its input coordinate."""
        x_offset, y_offset = self.output_to_input_offset_xy
        return y + y_offset, x + x_offset

    def to_dict(self):
        values = asdict(self)
        values["original_shape"] = list(self.original_shape)
        values["target_shape"] = list(self.target_shape)
        values["array_indexing"] = "image[y, x]"
        values["coordinate_origin"] = "zero_based_pixel_centers"
        values["output_to_input_offset_xy"] = list(self.output_to_input_offset_xy)
        return values


def center_crop_or_pad_with_transform(array, target_shape, pad_value=0):
    """Apply center crop/pad and return ``(output, exact_transform)``."""
    transform = CropPadTransform.from_shapes(np.asarray(array).shape, target_shape)
    return transform.apply(array, pad_value=pad_value), transform


def invalid_pixel_mask(array):
    """Return a boolean mask where ``True`` means invalid input data."""
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError("Invalid-pixel masks require a two-dimensional array")
    return ~np.isfinite(array)


def coverage_mask(array_shape, transform):
    """Return a boolean mask where ``True`` means no measured coverage."""
    source_coverage = np.zeros(tuple(array_shape), dtype=bool)
    return transform.apply(source_coverage, pad_value=True)


def background_exclusion_mask(invalid_mask, coverage, source_mask=None):
    """Combine exclusions used by background estimation without altering data."""
    result = np.asarray(invalid_mask, dtype=bool) | np.asarray(coverage, dtype=bool)
    if source_mask is not None:
        result |= np.asarray(source_mask, dtype=bool)
    return result
