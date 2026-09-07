# lensing

use pip install -r requirements.txt in venv or local environment

Note: Most data is ignored in repository due to size being too large, contact for data

## Reproduce the dataset pipeline

Run these commands from the repository root after placing the CASTLES FITS files in `data/raw/castles`:

```powershell
python src/processing/build_metadata.py
python src/processing/build_catalog.py
python src/processing/enrich_metadata.py
python src/processing/preprocess.py
python src/testing/validate_dataset.py
python src/processing/create_splits.py
```

The pipeline writes enriched metadata to `data/labels`, normalized `256x256` `float32` arrays to `data/processed/normalized`, and lens-level splits to `data/splits`.