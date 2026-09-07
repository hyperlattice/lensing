from pathlib import Path

import numpy as np
import pandas as pd


class LensDataset:
    """Small NumPy-compatible dataset for validated processed lens images."""

    def __init__(self, manifest_path, root=Path(".")):
        self.root = Path(root)
        self.metadata = pd.read_csv(manifest_path).reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        image = np.load(self.root / row["processed_path"])
        if image.shape != (256, 256) or image.dtype != np.float32:
            raise ValueError(f"Invalid image contract for {row['sample_id']}")
        if not np.isfinite(image).all():
            raise ValueError(f"Non-finite image values for {row['sample_id']}")
        return image, row.to_dict()