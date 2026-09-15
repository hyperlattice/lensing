import numpy as np
from astropy.io import fits

from src.processing.preprocess import preprocess_image


def test_background_source_exclusion_does_not_erase_source_from_model_products(tmp_path):
    image = np.zeros((32, 32), dtype=np.float32)
    image[15:19, 15:19] = 10.0
    path = tmp_path / "source.fits"
    fits.PrimaryHDU(image).writeto(path)

    result = preprocess_image(
        path,
        target_shape=(32, 32),
        normalization_method="masked_percentile",
        background_min_valid_pixels=16,
    )

    assert result["background"].source_pixels >= 1
    assert result["background_mask"][16, 16]
    assert not result["invalid_mask"][16, 16]
    assert result["normalized"][16, 16] > 0.0
    assert result["variance"].values[16, 16] > 0.0
