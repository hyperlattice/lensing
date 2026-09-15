import numpy as np
import pandas as pd
from astropy.io import fits

from src.processing.preview_dataset import create_previews


def test_preview_index_and_contact_sheet_preserve_observation_mapping(tmp_path):
    raw_dir = tmp_path / "raw"
    first_dir = raw_dir / "L1" / "FIT" / "NIC2H"
    second_dir = raw_dir / "L2" / "FIT" / "NIC2H"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    first = first_dir / "a.fits"
    second = second_dir / "b.fits"
    fits.PrimaryHDU(np.ones((8, 8), dtype=np.float32)).writeto(first)
    fits.PrimaryHDU(np.ones((8, 8), dtype=np.float32) * 2).writeto(second)
    manifest = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "sample_id": ["observation-a", "observation-b"],
            "raw_path": [first.as_posix(), second.as_posix()],
        }
    ).to_csv(manifest, index=False)

    index = create_previews(raw_dir, tmp_path / "preview", manifest, dpi=50)

    assert len(index) == 2
    assert set(index["status"]) == {"ok"}
    assert index["preview_path"].is_unique
    assert (tmp_path / "preview" / "contact_sheet.png").exists()
    assert index["contact_sheet_path"].nunique() == 1
