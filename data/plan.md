Directosry:

project/
│
├── data/
│   ├── raw/
│   │   ├── CASTLES/
│   │   ├── simulations/
│   │   └── psfs/
│   │
│   ├── processed/
│   │   ├── images/
│   │   ├── masks/
│   │   ├── normalized/
│   │   └── resized/
│   │
│   ├── labels/
│   │   ├── metadata.csv
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   │
│   └── splits/
│       ├── train/
│       ├── validation/
│       └── test/
│
├── notebooks/
├── src/
├── models/
└── results/

1. Raw data
- Never modify raw data and downloaded files
2. Processed images
- Add all types of processing if needed
3. Data/Labels
- Create csv summarizing each lens (metadata = pd.read_csv("metadata.csv"))
- Label each lens with the parameters (label the values into a csv as well)
4. Train Test Validation Split
- Create csv files listing the images that belong to each split

NOTE: Keep simulated lenses alongside, but add extra files