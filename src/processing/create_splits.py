from pathlib import Path

import pandas as pd


MANIFEST_PATH = Path("data/labels/processed_metadata.csv")
OUTPUT_DIR = Path("data/splits")
SPLIT_NAMES = ("train", "validation", "test")


def create_splits(manifest_path=MANIFEST_PATH, seed=42, train_fraction=0.7, validation_fraction=0.15):
    manifest = pd.read_csv(manifest_path)
    lens_ids = manifest["lens_id"].drop_duplicates().sample(frac=1, random_state=seed).tolist()
    train_end = int(len(lens_ids) * train_fraction)
    validation_end = train_end + int(len(lens_ids) * validation_fraction)
    assignments = {
        lens_id: "train" if index < train_end else "validation" if index < validation_end else "test"
        for index, lens_id in enumerate(lens_ids)
    }
    manifest["split"] = manifest["lens_id"].map(assignments)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        manifest.loc[manifest["split"] == split_name].drop(columns="split").to_csv(
            OUTPUT_DIR / f"{split_name}.csv", index=False
        )
    return manifest


if __name__ == "__main__":
    result = create_splits()
    print(result.groupby("split")["lens_id"].nunique().to_string())