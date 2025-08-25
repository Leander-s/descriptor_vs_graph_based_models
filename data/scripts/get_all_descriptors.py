from get_rdkit_descriptors import get_rdkit_descriptors
from get_minimol_descriptors import get_minimol_descriptors
from config import OriginalDatasets as Datasets
import pandas as pd
from minimol import Minimol


def get_all_descriptors(dataset) -> (pd.DataFrame, pd.DataFrame):
    model = Minimol()
    try:
        time_df = pd.read_csv("descriptor_compute_times.csv")
    except FileNotFoundError:
        cols = ["rdkit_time", "minimol_time", "dataset"]
        time_df = pd.DataFrame(columns=cols)
    path = f"../original_datasets/{dataset}.csv"
    rdkit_df, rdkit_time = get_rdkit_descriptors(path)
    minimol_df, minimol_time = get_minimol_descriptors(model, path)
    new_row = [
        rdkit_time,
        minimol_time,
        dataset
    ]

    index_list = time_df.index[time_df["dataset"] == dataset].tolist()
    if len(index_list) == 0:
        time_df.loc[len(time_df)] = new_row
    else:
        time_df.loc[index_list[0]] = new_row
    time_df.to_csv("descriptor_compute_times.csv", index=False)

    return (rdkit_df, minimol_df)


def main():
    for dataset in Datasets:
        rdkit_df, minimol_df = get_all_descriptors(dataset)
        minimol_df.to_csv(
            f"../minimol_datasets/{dataset}-minimol_descriptors.csv",
            index=False)

        rdkit_df.to_csv(
            f"../rdkit_datasets/{dataset}-rdkit_descriptors.csv",
            index=False)


if __name__ == '__main__':
    main()
