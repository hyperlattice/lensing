import numpy as np
import pandas as pd
import pytest
import yaml

from src.modeling import domain_gap_week10 as audit


@pytest.mark.parametrize("all_missing", [True, False])
def test_domain_gap_writes_exclusions_before_rejecting_incomplete_data(tmp_path, all_missing):
    image = tmp_path / "image.npy"
    np.save(image, np.arange(64, dtype=np.float32).reshape(8, 8))
    real = pd.DataFrame(index=range(137), columns=audit.REAL_METADATA_COLUMNS)
    real["sample_id"] = [f"real-{i}" for i in range(137)]
    real["lens_id"] = real["sample_id"]
    real["pixel_scale"] = 0.05
    products = pd.DataFrame(index=range(137), columns=audit.REAL_PRODUCT_COLUMNS)
    products["sample_id"] = real["sample_id"]
    products["processed_path"] = str(tmp_path / "missing.npy") if all_missing else str(image)
    products["original_shape"] = "8x8"
    simulation = pd.DataFrame(index=range(1400), columns=audit.SIMULATION_COLUMNS)
    simulation["sample_id"] = [f"sim-{i}" for i in range(1400)]
    simulation["parent_sample_id"] = [f"parent-{i // 2}" for i in range(1400)]
    simulation["processed_path"] = "missing.npy"
    config = {"output": str(tmp_path / "report"), "real_products_root": str(tmp_path),
              "simulation_release": str(tmp_path)}
    for key, frame in (("real_metadata", real), ("real_processed_metadata", products),
                       ("simulation_train_split", simulation)):
        path = tmp_path / f"{key}.csv"
        frame.to_csv(path, index=False)
        config[key] = str(path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="inspect exclusions.csv"):
        audit.run(config_path)

    exclusions = pd.read_csv(tmp_path / "report" / "exclusions.csv")
    assert len(exclusions) == (1537 if all_missing else 1400)
    assert set(exclusions["domain"]) == ({"castles_real", "synthetic_week8_train"}
                                         if all_missing else {"synthetic_week8_train"})
    assert exclusions["reason"].str.contains("missing.npy", regex=False).all()
