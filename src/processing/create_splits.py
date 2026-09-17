import hashlib
import json
from pathlib import Path

import pandas as pd


MANIFEST_PATH = Path("data/labels/derived/processed_metadata.csv")
OUTPUT_DIR = Path("data/manifests/actual/castles-development-v1/splits")
SPLIT_NAMES = ("train", "validation", "test")
ANCESTRY_COLUMNS = (
    "split_group_id",
    "parent_system_id",
    "source_template_id",
    "background_template_id",
    "parent_real_lens_id",
)


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


class _DisjointSet:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _usable_group_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    value = str(value).strip()
    if not value or value.lower() in {"unknown", "none", "nan"}:
        return None
    return value


def ancestry_components(manifest, group_columns=ANCESTRY_COLUMNS):
    """Return sample-to-component assignments linked by any ancestry field.

    Rows sharing a parent, source template, background template, or real seed
    lens become one connected component. This also handles transitive links:
    if A shares a source with B and B shares a background with C, all three
    must remain in one split.
    """

    manifest = manifest.reset_index(drop=True)
    if "sample_id" not in manifest:
        raise ValueError("Manifest is missing sample_id")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate sample_id values")
    available = [column for column in group_columns if column in manifest.columns]
    if not available:
        raise ValueError(f"Manifest contains none of the ancestry columns: {list(group_columns)}")

    groups = _DisjointSet(len(manifest))
    token_owner = {}
    for index, row in manifest.iterrows():
        row_has_group = False
        for column in available:
            value = _usable_group_value(row[column])
            if value is None:
                continue
            row_has_group = True
            token = (column, value)
            if token in token_owner:
                groups.union(index, token_owner[token])
            else:
                token_owner[token] = index
        if not row_has_group:
            raise ValueError(f"{row['sample_id']}: no usable ancestry group value")

    members = {}
    for index, sample_id in enumerate(manifest["sample_id"].astype(str)):
        members.setdefault(groups.find(index), []).append(sample_id)
    component_ids = {
        root: "group-" + hashlib.sha256("\n".join(sorted(sample_ids)).encode("utf-8")).hexdigest()[:16]
        for root, sample_ids in members.items()
    }
    return pd.DataFrame(
        {
            "sample_id": manifest["sample_id"].astype(str),
            "effective_group_id": [component_ids[groups.find(index)] for index in range(len(manifest))],
        }
    )


def create_ancestry_splits(
    manifest_path,
    output_dir,
    *,
    seed=4201,
    train_fraction=0.7,
    validation_fraction=0.15,
    group_columns=ANCESTRY_COLUMNS,
):
    """Create transitive ancestry-safe simulation splits and an audit report."""

    _validate_fractions(train_fraction, validation_fraction)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    manifest = pd.read_csv(manifest_path)
    components = ancestry_components(manifest, group_columns=group_columns)
    working = manifest.merge(components, on="sample_id", validate="one_to_one")
    group_ids = working["effective_group_id"].drop_duplicates()
    shuffled = group_ids.sample(frac=1, random_state=int(seed)).tolist()
    train_count = int(len(shuffled) * train_fraction)
    validation_count = int(len(shuffled) * validation_fraction)
    group_splits = {
        group_id: (
            "train"
            if index < train_count
            else "validation"
            if index < train_count + validation_count
            else "test"
        )
        for index, group_id in enumerate(shuffled)
    }
    working["split"] = working["effective_group_id"].map(group_splits)

    leakage = {}
    for column in group_columns:
        if column not in working:
            continue
        usable = working.loc[working[column].map(_usable_group_value).notna(), [column, "split"]]
        crossing = usable.groupby(column)["split"].nunique()
        offenders = crossing[crossing > 1].index.astype(str).tolist()
        if offenders:
            leakage[column] = offenders
    if leakage:
        raise ValueError(f"Ancestry leakage detected: {leakage}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        working.loc[working["split"] == split_name].to_csv(
            output_dir / f"{split_name}.csv", index=False
        )
    assignments = (
        working.groupby(["effective_group_id", "split"], as_index=False)
        .agg(
            sample_count=("sample_id", "size"),
            parent_count=("parent_system_id", "nunique")
            if "parent_system_id" in working
            else ("sample_id", "size"),
        )
    )
    assignments.to_csv(output_dir / "assignments.csv", index=False)
    split_summary = {}
    for split_name in SPLIT_NAMES:
        selected = working[working["split"] == split_name]
        split_summary[split_name] = {
            "observations": int(len(selected)),
            "effective_groups": int(selected["effective_group_id"].nunique()),
            "parents": int(selected["parent_system_id"].nunique())
            if "parent_system_id" in selected
            else int(selected["effective_group_id"].nunique()),
        }
    report = {
        "manifest_path": manifest_path.as_posix(),
        "seed": int(seed),
        "fractions": {
            "train": float(train_fraction),
            "validation": float(validation_fraction),
            "test": float(1.0 - train_fraction - validation_fraction),
        },
        "group_columns": [column for column in group_columns if column in working],
        "sample_count": int(len(working)),
        "effective_group_count": int(working["effective_group_id"].nunique()),
        "split_counts": split_summary,
        "leakage_detected": False,
    }
    (output_dir / "split_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return working, report


def main():
    result = create_splits()
    print(result.groupby("split")["lens_id"].nunique().to_string())


if __name__ == "__main__":
    main()
