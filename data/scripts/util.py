import os
import pandas as pd


def load_dataset(name, descriptors=""):
    folder = ""
    match descriptors:
        case "rdkit":
            folder = "rdkit_datasets"
        case "minimol":
            folder = "minimol_datasets"
        case "moe":
            folder = "moe-neutral_datasets"
        case _:
            folder = "original_datasets"
    descriptors_suffix = ""
    if descriptors != "":
        descriptors_suffix = f"_{descriptors}_descriptors"
    path = f"../{folder}/{name}{descriptors_suffix}.csv"
    if not os.path.isfile(path):
        folder = f"additional_datasets/{descriptors}"
        if descriptors == "":
            folder = "additional_datasets/original"
        path = f"../{folder}/{name}{descriptors_suffix}.csv"
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        return None
    return pd.read_csv(path)
