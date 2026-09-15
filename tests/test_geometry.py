import numpy as np

from src.processing.geometry import (
    CropPadTransform,
    background_exclusion_mask,
    center_crop_or_pad_with_transform,
    coverage_mask,
    invalid_pixel_mask,
)


def test_padding_transform_maps_markers_and_masks_padding():
    image = np.arange(4, dtype=np.float32).reshape(2, 2)
    output, transform = center_crop_or_pad_with_transform(image, (4, 4), pad_value=0.0)

    assert output[1:3, 1:3].tolist() == image.tolist()
    assert transform.output_to_input(1, 1) == (0, 0)
    coverage = coverage_mask(image.shape, transform)
    assert coverage.dtype == bool
    assert coverage[0, 0]
    assert not coverage[1, 1]
    assert np.array_equal(background_exclusion_mask(np.zeros((4, 4), dtype=bool), coverage), coverage)


def test_crop_transform_maps_output_to_original_coordinates():
    image = np.arange(25, dtype=np.float32).reshape(5, 5)
    output, transform = center_crop_or_pad_with_transform(image, (3, 3))

    assert output[0, 0] == image[1, 1]
    assert output[-1, -1] == image[3, 3]
    assert transform.output_to_input(0, 0) == (1, 1)
    assert transform.to_dict()["array_indexing"] == "image[y, x]"


def test_invalid_mask_marks_nonfinite_input():
    image = np.array([[1.0, np.nan], [np.inf, 2.0]])
    mask = invalid_pixel_mask(image)
    assert mask.tolist() == [[False, True], [True, False]]
