import pandas as pd
from config import OriginalDatasets, Multitask_Datasets
from get_all_descriptors import get_all_descriptors
from isolate_tasks import isolate_tasks


def main():
    original_datasets = []
    rdkit_datasets = []
    minimol_datasets = []

    for original_dataset in OriginalDatasets:
        original_datasets.append(
            pd.read_csv(f"../original_datasets/{original_dataset}.csv")
        )
        rdkit_dataset, minimol_dataset = get_all_descriptors(original_dataset)
        rdkit_datasets.append(
            pd.read_csv(
                f"../rdkit_datasets/{original_dataset}-rdkit_descriptors.csv")
        )
        minimol_datasets.append(
            pd.read_csv(
                "../minimol_datasets/"
                f"{original_dataset}-minimol_descriptors.csv")
        )
        if original_dataset in Multitask_Datasets:
            isolate_tasks(original_dataset, -1)


if __name__ == '__main__':
    main()
