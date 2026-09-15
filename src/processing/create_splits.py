from pathlib import Path

import pandas as pd


MANIFEST_PATH = Path("data/labels/processed_metadata.csv")
OUTPUT_DIR = Path("data/splits")
SPLIT_NAMES = ("train", "validation", "test")


def _validate_fractions(train_fraction, validation_fraction):
    if train_fraction < 0 or validation_fraction < 0:
        raise ValueError("Split fractions must be nonnegative")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave room for test")


def _read_existing_assignments(path):
    existing = pd.read_csv(path)
    required = {"lens_id", "split"}
    missing = required.difference(existing.columns)
    if missing:
        raise ValueError(f"Existing split file is missing columns: {sorted(missing)}")
    if not existing["split"].isin(SPLIT_NAMES).all():
        raise ValueError(f"Existing split file contains unknown split names: {path}")

    assignments = existing[["lens_id", "split"]].drop_duplicates()
    if assignments["lens_id"].duplicated().any():
        raise ValueError("Existing assignments place a lens in multiple splits")
    return dict(zip(assignments["lens_id"], assignments["split"]))


def create_splits(
    manifest_path=MANIFEST_PATH,
    output_dir=OUTPUT_DIR,
    seed=42,
    train_fraction=0.7,
    validation_fraction=0.15,
    existing_assignments_path=None,
):
    """Create lens-level splits without writing outside ``output_dir``.

    ``existing_assignments_path`` may point to a manifest containing one
    ``lens_id, split`` row per lens. Existing assignments are preserved and
    only previously unseen lenses are assigned using the configured seed.
    """
    _validate_fractions(train_fraction, validation_fraction)
    manifest = pd.read_csv(manifest_path)
    required = {"sample_id", "lens_id"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if manifest["lens_id"].isna().any():
        raise ValueError("Manifest contains missing lens_id values")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate sample_id values")

    lens_ids = manifest["lens_id"].drop_duplicates().tolist()
    assignments = {}
    if existing_assignments_path is not None:
        assignments.update(_read_existing_assignments(existing_assignments_path))

    unknown_lenses = [lens_id for lens_id in lens_ids if lens_id not in assignments]
    shuffled = pd.Series(unknown_lenses).sample(frac=1, random_state=seed).tolist()

    target_counts = {
        "train": int(len(lens_ids) * train_fraction),
        "validation": int(len(lens_ids) * validation_fraction),
    }
    target_counts["test"] = len(lens_ids) - target_counts["train"] - target_counts["validation"]
    current_counts = {split: sum(value == split for value in assignments.values()) for split in SPLIT_NAMES}

    # Fill split capacities in the historical train/validation/test order.
    # This keeps a seeded rerun compatible with the existing manifests while
    # still allowing explicitly supplied assignments to reserve lenses first.
    unassigned = list(shuffled)
    for split in SPLIT_NAMES:
        capacity = max(target_counts[split] - current_counts[split], 0)
        selected = unassigned[:capacity]
        for lens_id in selected:
            assignments[lens_id] = split
        current_counts[split] += len(selected)
        unassigned = unassigned[len(selected):]

    if unassigned:
        fallback = min(SPLIT_NAMES, key=lambda name: current_counts[name])
        for lens_id in unassigned:
            assignments[lens_id] = fallback
            current_counts[fallback] += 1

    manifest["split"] = manifest["lens_id"].map(assignments)
    if manifest["split"].isna().any():
        raise ValueError("Some lens IDs could not be assigned to a split")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        manifest.loc[manifest["split"] == split_name].drop(columns="split").to_csv(
            output_dir / f"{split_name}.csv", index=False
        )
    manifest[["lens_id", "split"]].drop_duplicates().to_csv(
        output_dir / "assignments.csv", index=False
    )
    return manifest


def main():
    result = create_splits()
    print(result.groupby("split")["lens_id"].nunique().to_string())


if __name__ == "__main__":
    main()
