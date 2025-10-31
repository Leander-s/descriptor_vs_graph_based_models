from lib.config import Datasets, Models, Regression_Datasets
from lib.plotting import get_average_and_std_r2, get_average_and_std_auc_roc
import os
import pandas as pd


def get_file_name(dataset, model) -> str:
    return f"{dataset}_{model}_results.csv"


def list_to_csv(
        data: list[list], index: list[str], columns: list[str], filename: str
):
    df = pd.DataFrame(data, index=index, columns=columns)

    df.to_csv(filename)


def load_data(datasets, models) -> list[list[float]]:
    '''
    Puts data for datasets and models into a 2D python list
    Rows: Datasets
    Columns: Models
    '''
    data = []
    errors = []
    for dataset in datasets:
        dataset_data = []
        dataset_errors = []
        for model in models:
            file = get_file_name(dataset, model)
            if not os.path.isfile(file):
                dataset_data.append("na")
                dataset_errors.append("na")
                continue
            if dataset in Regression_Datasets:
                score, std = get_average_and_std_r2(file)
            else:
                score, std = get_average_and_std_auc_roc(file)
            dataset_data.append(round(score, 3))
            dataset_errors.append(round(std, 3))
        data.append(dataset_data)
        errors.append(dataset_errors)
    return data, errors


def summarize_data(datasets, models):
    data, errors = load_data(datasets, models)
    list_to_csv(data, datasets, models, "summarized_results.csv")
    list_to_csv(errors, datasets, models, "summarized_errors.csv")


def summarize_times(datasets, models):
    times = []
    for dataset in datasets:
        dataset_times = []
        for model in models:
            time_path = f"./times/{dataset}_{model}_time.csv"
            if not os.path.isfile(time_path):
                dataset_times.append('na')
                continue
            df = pd.read_csv(time_path)
            time = df['time'].iloc[0]
            dataset_times.append(round(time, 3))
        times.append(dataset_times)
    list_to_csv(times, datasets, models, "summarized_times.csv")


def main():
    summarize_data(Datasets, Models)
    summarize_times(Datasets, Models)


if __name__ == '__main__':
    main()
