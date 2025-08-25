import pandas as pd
from config import Multitask_Datasets


def isolate_task(df, task, max_tasks):
    tasks_last_ind = max_tasks
    to_keep = df.columns[[0, task]].append(df.columns[tasks_last_ind + 1:])
    return df[to_keep]


def isolate_tasks(dataset_name, amount):
    original_path = f"../original_datasets/{dataset_name}.csv"
    rdkit_path = f"../rdkit_datasets/{dataset_name}-rdkit_descriptors.csv"
    minimol_path = (
        f"../minimol_datasets/"
        f"{dataset_name}-minimol_descriptors.csv"
    )
    moe_neutral_path = (
        f"../moe-neutral_datasets/"
        f"{dataset_name}-moe-neutral_descriptors.csv"
    )
    original_frame = pd.read_csv(original_path)
    try:
        rdkit_frame = pd.read_csv(rdkit_path)
    except Exception:
        rdkit_frame = None
    try:
        minimol_frame = pd.read_csv(minimol_path)
    except Exception:
        minimol_frame = None
    try:
        moe_neutral_frame = pd.read_csv(moe_neutral_path)
    except Exception:
        moe_neutral_frame = None

    max_tasks = len(original_frame.columns) - 2

    if amount == -1:
        amount = max_tasks

    for i in range(1, amount + 1):
        new_frame = isolate_task(original_frame, i, max_tasks)
        path = f"../additional_datasets/original/{dataset_name}{i}.csv"
        new_frame.to_csv(path, index=False)

        if rdkit_frame:
            new_rdkit_frame = isolate_task(rdkit_frame, i, max_tasks)
            new_rdkit_path = f"../additional_datasets/rdkit/{
                dataset_name}{i}_rdkit_descriptors.csv"
            new_rdkit_frame.to_csv(new_rdkit_path, index=False)
        if minimol_frame:
            new_minimol_frame = isolate_task(minimol_frame, i, max_tasks)
            new_minimol_path = f"../additional_datasets/minimol/{
                dataset_name}{i}_minimol_descriptors.csv"
            new_minimol_frame.to_csv(new_minimol_path, index=False)
        if moe_neutral_frame:
            new_moe_neutral_frame = isolate_task(
                moe_neutral_frame, i, max_tasks
            )
            new_moe_neutral_path = f"../additional_datasets/moe-neutral/{
                dataset_name}{i}_moe-neutral_descriptors.csv"
            new_moe_neutral_frame.to_csv(new_moe_neutral_path, index=False)


def main():
    for dataset in Multitask_Datasets:
        isolate_tasks(dataset, -1)


if __name__ == '__main__':
    main()
