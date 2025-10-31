import pandas as pd
import os


def get_times(datasets, models):
    data = []
    for dataset in datasets:
        row = []
        for model in models:
            path = f"times/{dataset}_{model}_time.csv"
            if not os.path.isfile(path):
                row.append("na")
                continue
            time = pd.read_csv(path).iloc[0,0]
            row.append(str(round(time, 3)))
        data.append(row)
    return data


def main():
    # plot times cd
    pass


if __name__ == '__main__':
    main()
