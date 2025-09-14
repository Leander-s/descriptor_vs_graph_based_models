import pandas as pd
import os
import shutil
from config import OriginalDatasets, Multitask_Datasets
from get_all_descriptors import get_all_descriptors
from isolate_tasks import isolate_tasks
from minimol import Minimol


def main():
    original_datasets = []
    rdkit_datasets = []
    minimol_datasets = []

    model = Minimol()

    for original_dataset in OriginalDatasets:
        print(f"Preparing data for {original_dataset}")
        original_datasets.append(
            pd.read_csv(f"../original_datasets/{original_dataset}.csv")
        )
        shutil.rmtree("../rdkit_datasets/")
        shutil.rmtree("../minimol_datasets/")
        os.mkdir("../rdkit_datasets")
        os.mkdir("../minimol_datasets")
        rdkit_dataset, minimol_dataset = get_all_descriptors(
            original_dataset, model)
        rdkit_dataset.to_csv(
            f"../rdkit_datasets/{original_dataset}-rdkit_descriptors.csv")
        minimol_dataset.to_csv(
            f"../minimol_datasets/{original_dataset}-minimol_descriptors.csv")
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
