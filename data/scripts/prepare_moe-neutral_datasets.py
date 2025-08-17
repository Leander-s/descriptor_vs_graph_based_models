import pandas as pd
from config import OriginalDatasets as Datasets
import sys


def prepare_dataset(dataset):
    print(f"Preparing {dataset}")
    moe_path = f"../moe-neutral_datasets/{dataset}_neutral.csv"
    reference_path = f"../original_datasets/{dataset}.csv"
    result_path = f"../moe-neutral_datasets/{dataset}_moe-" \
        "neutral_descriptors.csv"

    reference_df = pd.read_csv(reference_path)
    moe_df = pd.read_csv(moe_path)
    result = pd.concat(
        [
            reference_df,
            moe_df.iloc[:, 1:]
        ],
        axis=1)
    result.to_csv(result_path, index=False)
    print(f"{dataset} prepared")


def prepare_all():
    for dataset in Datasets:
        prepare_dataset(dataset)


def main():
    dataset = None
    if len(sys.argv) != 1:
        dataset = sys.argv[1]

    if not dataset:
        prepare_all()
        return

    if dataset not in Datasets:
        print(f"Dataset {dataset} not known")
        return

    prepare_dataset(dataset)


if __name__ == '__main__':
    main()
