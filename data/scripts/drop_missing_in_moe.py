from config import OriginalDatasets as Datasets
import pandas as pd

for dataset in Datasets:
    name = f"{dataset}_moe-neutral_descriptors.csv"
    path = f"../moe-neutral_datasets/{name}"
    df = pd.read_csv(path)
    df = df.dropna(axis=1, how="any")
    df.to_csv(path, index=False)
