import numpy as np
import pandas as pd
from lib.config import Models, Datasets
from lib.demsar import demsar


def rank_dataset(dataset, results):
    score = {}
    row_labels = results.index
    column_labels = results.columns
    dataset_idx = list(row_labels).index(dataset)
    np_results = results.to_numpy()
    for model_idx, model in enumerate(column_labels):
        result = np_results[dataset_idx, model_idx]
        if result == "na":
            score[model] = 0
            continue
        score[model] = float(result)

    return score


def rank_data(datasets, models):
    scores = {}
    datasets_in_use = []
    results = pd.read_csv("summarized_results.csv", index_col=0)
    for dataset in datasets:
        scores[dataset] = rank_dataset(dataset, results)
        # if everything has the same rank, there are no results for this
        # dataset, therefore we ignore it
        if any(value != "na" for value in scores[dataset].values()):
            datasets_in_use.append(dataset)

    data = []
    for dataset in datasets_in_use:
        data.append(np.array([scores[dataset][model] for model in models]))

    return demsar(np.array(data))


def rank_results(results: pd.DataFrame):
    scores = {}
    rows = results.index
    columns = results.columns
    used_rows = []
    for row in rows:
        scores[row] = rank_dataset(row, results)
        if any(value != "na" for value in scores[row].values()):
            used_rows.append(row)
    data = []
    for row in used_rows:
        data.append(np.array([scores[row][column] for column in columns]))

    return demsar(np.array(data))


def main():
    results = rank_data(Datasets, Models)

    for i, model in enumerate(Models):
        print(f"{model} : {results['average_ranks'][i]}")
    print(results["iman_davenport"])


if __name__ == '__main__':
    main()
