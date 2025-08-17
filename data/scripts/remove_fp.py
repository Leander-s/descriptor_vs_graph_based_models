import pandas as pd
import sys


def remove_fp(dataset):
    # read original csv
    path = f"../moe-neutral_datasets/{dataset}_moe-neutral_descriptors.csv"
    try:
        df = pd.read_csv(path)
    except Exception:
        raise Exception("Not a valid dataset")

    # save original as backup
    fp_path = "../moe-neutral_datasets/"\
        f"{dataset}_moe-neutral_descriptors_fp.csv"
    df.to_csv(fp_path, index=False)

    # find all fp columns
    fps = []
    for column in df.columns:
        if "FP:" in column:
            fps.append(column)

    # drop fp columns and save at old path
    df = df.drop(columns=fps)
    df.to_csv(path, index=False)


def main():
    # Argument given?
    if len(sys.argv) < 2:
        raise Exception("Need to give a dataset as argument")
    dataset = sys.argv[1]
    remove_fp(dataset)


if __name__ == '__main__':
    main()
