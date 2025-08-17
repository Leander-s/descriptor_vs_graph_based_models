import pandas as pd
from config import OriginalDatasets as Datasets


def main():
    for dataset in Datasets:
        name = f"{dataset}_moe-neutral_descriptors.csv"
        path = f"../moe-neutral_datasets/{name}"
        df = pd.read_csv(path)

        # This drops all fingerprints, not the final version
        df = df.drop(columns=df.filter(like="FP:", axis=1))
        df.to_csv(path, index=False)


if __name__ == '__main__':
    main()
