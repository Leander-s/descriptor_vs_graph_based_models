from config import Datasets
from util import load_dataset
import pandas as pd


def main():
    counts = []
    for dataset in Datasets:
        dataset = load_dataset(dataset)
        if dataset is None:
            counts.append("")
            continue
        counts.append(dataset.count(axis=0).iloc[0])
    pd.DataFrame(
        counts, index=Datasets,
        columns=["count"]).to_csv("sample_counts.csv")


if __name__ == '__main__':
    main()
