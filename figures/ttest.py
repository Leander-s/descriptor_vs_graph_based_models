from scipy.stats import t
from math import sqrt
from statistics import stdev
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from lib.config import (Multitask_Datasets, Datasets,
                        Regression_Datasets, Models, Graph_Based_Models)

REGRESSION_METRIC = "r2"
CLASSIFICATION_METRIC_DBM = "auc_roc"
CLASSIFICATION_METRIC_GNN = "roc_auc" if CLASSIFICATION_METRIC_DBM == "auc_roc" else "roc_prc"
ALPHA = 0.05


class DatasetTestResult:
    def __init__(self):
        self.data = np.array([])
        self.labels = []


def corrected_dependent_ttest(
        data1, data2, n_training_samples, n_test_samples, alpha
):

    n = len(data1)
    differences = [(data1[i]-data2[i]) for i in range(n)]
    sd = stdev(differences)
    # degrees of freedom
    df = n - 1
    # calculate the critical value
    cv = t.ppf(1.0 - alpha, df)
    if sd == 0:
        return 0, df, cv, 1
    divisor = 1 / n * sum(differences)
    test_training_ratio = n_test_samples / n_training_samples
    denominator = sqrt(1 / n + test_training_ratio) * sd
    t_stat = divisor / denominator
    # calculate the p-value
    p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
    # return everything
    return t_stat, df, cv, p


def test_model(dataset, model, reg):
    results = []
    model_path = f"./{dataset}_{model}_results.csv"
    try:
        model_data = pd.read_csv(model_path)
    except FileNotFoundError:
        return np.array(results)

    if reg:
        model_data = model_data[REGRESSION_METRIC].values
    else:
        metric = CLASSIFICATION_METRIC_GNN if model in Graph_Based_Models else CLASSIFICATION_METRIC_DBM
        model_data = model_data[metric].values

    original_dataset_path = f"../../../data/original_datasets/{dataset}.csv"
    additional_dataset_path = f"../../../data/additional_datasets/original/{
        dataset}.csv"
    try:
        dataset_samples = pd.read_csv(original_dataset_path)
    except FileNotFoundError:
        dataset_samples = pd.read_csv(additional_dataset_path)

    n_train = int(len(dataset_samples) * 0.8)
    n_test = int(n_train/8)

    for other_model in Models:
        other_path = f"./{dataset}_{other_model}_results.csv"
        try:
            other_data = pd.read_csv(other_path)
        except FileNotFoundError:
            continue
        if reg:
            other_data = other_data[REGRESSION_METRIC].values
        else:
            metric = CLASSIFICATION_METRIC_GNN if other_model in Graph_Based_Models else CLASSIFICATION_METRIC_DBM
            other_data = other_data[metric].values

        tstat, df, cv, p = corrected_dependent_ttest(
            model_data, other_data, n_train, n_test, ALPHA
        )
        if model_data.mean() < other_data.mean():
            p *= -1
        results.append(p)
    return np.array(results)


def test_dataset(dataset) -> DatasetTestResult:
    '''
    dataset is the name of the dataset to test
    '''

    results = DatasetTestResult()
    data = []
    reg = True if dataset in Regression_Datasets else False
    for model in Models:
        model_results: np.array = test_model(dataset, model, reg)
        if model_results.size == 0:
            continue
        data.append(model_results)
        results.labels.append(model)
    results.data = np.array(data)
    return results


def plot_results(dataset, results: DatasetTestResult):
    '''
    dataset: name of the dataset to plot
    results: DatasetTestResult from test_dataset
    '''
    fig, ax = plt.subplots(figsize=(20, 20))
    color_matrix = np.where(results.data < 0, -1, 1)
    color_matrix = np.where(abs(results.data) >= ALPHA, 0, color_matrix)
    _ = ax.imshow(color_matrix, cmap=ListedColormap(
        ['red', 'white', 'green']), vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(results.labels)))
    ax.set_yticks(np.arange(len(results.labels)))
    ax.set_xticklabels(results.labels)
    ax.set_yticklabels(results.labels)

    plt.xticks(rotation=45)

    for i in range(len(results.labels)):
        for j in range(len(results.labels)):
            ax.text(
                j, i, f"{abs(results.data[i, j]):.2f}",
                ha="center", va="center", color="black"
            )

    path = f"./ttest_plots/{dataset}.pdf"
    plt.savefig(path)
    plt.close()


def main():
    for dataset in Datasets:
        if dataset in Multitask_Datasets:
            continue
        results: DatasetTestResult = test_dataset(dataset)
        df = pd.DataFrame(
            results.data, index=results.labels, columns=results.labels
        )

        plot_results(dataset, results)
        path = f"./ttests/{dataset}.csv"
        df.to_csv(path)


if __name__ == '__main__':
    main()
