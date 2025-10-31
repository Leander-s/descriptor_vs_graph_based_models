from lib.config import Datasets, Models
import os


def get_missing(folder):
    directory = os.fsencode(folder)

    all_files = []

    for dataset in Datasets:
        for model in Models:
            all_files.append(f"{dataset}_{model}_results.csv")

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if not filename.endswith(".csv"):
            continue
        if filename not in all_files:
            print(f"{filename} not in list")
            continue

        all_files.remove(filename)

    return all_files


def get_missing_on(missing_files, x):
    missing_on = []
    for file in missing_files:
        if x in file:
            missing_on.append(file)
    return missing_on


def main():
    missing = {}
    all_missing = get_missing(".")
    for model in Models:
        missing[model] = get_missing_on(all_missing, model)
        print(f"{model} has {len(missing[model])} missing results")
    print(80*"-")
    for dataset in Datasets:
        missing[dataset] = get_missing_on(all_missing, dataset)
        print(f"{dataset} has {len(missing[dataset])} missing results")


if __name__ == '__main__':
    main()
