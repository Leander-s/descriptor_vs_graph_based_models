import pandas as pd
from matplotlib.lines import Line2D
import re
from lib.config import (
    Datasets, Models, ClassificationTask, TrueSingleTask,
    SingleTask, Multitask_Datasets, Regression_Datasets
)
from rank import rank_results
import matplotlib.pyplot as plt
from matplotlib.table import Table
import numpy as np


def get_file_name(dataset, model) -> str:
    return f"{dataset}_{model}_results.csv"


def find_best_rank(s: list[float], i: list[int]):
    best = i[0]
    for index in i:
        if s[index] < s[best]:
            best = index
    return best


def find_best_three_ranks(s: list[float], i: list[int]):
    top_three = sorted(i[:3], key=lambda ind: s[ind])
    for index in i[3:]:
        if (s[index] < s[top_three[2]]):
            top_three[2] = index
            top_three = sorted(top_three, key=lambda ind: s[ind])
    return top_three


def dataframe_to_list(df: pd.DataFrame) -> list[list[float]]:
    rows = df.index
    cols = df.columns
    data = []
    for row in rows:
        row_data = []
        for col in cols:
            if df.at[row, col] == "na":
                row_data.append("na")
                continue
            row_data.append(
                df.at[row, col]
            )
        data.append(row_data)
    return data


def load_times(datasets, models) -> list[list[float]]:
    '''
    Puts times for datasets and models into a 2D python list
    Rows: Datasets
    Columns: Models
    '''
    times = pd.read_csv("./summarized_times.csv", index_col=0)
    data = []
    for dataset in datasets:
        dataset_data = []
        for model in models:
            if times.at[dataset, model] == "na":
                dataset_data.append("na")
                continue
            dataset_data.append(
                times.at[dataset, model]
            )
        data.append(dataset_data)
    return data


def load_data(datasets, models) -> list[list[float]]:
    '''
    Puts data for datasets and models into a 2D python list
    Rows: Datasets
    Columns: Models
    '''
    results = pd.read_csv("./summarized_results.csv", index_col=0)
    errors = pd.read_csv("./summarized_errors.csv", index_col=0)
    data = []
    for dataset in datasets:
        dataset_data = []
        for model in models:
            if results.at[dataset, model] == "na":
                dataset_data.append("na")
                continue
            dataset_data.append(
                f"{results.at[dataset, model]} ± {errors.at[dataset, model]}"
            )
        data.append(dataset_data)
    return data


def get_results(data):
    results = []
    errors = []
    for row in data:
        row = np.array(row)
        row_values = []
        row_errors = []
        for number in row:
            if "na" in number:
                row_values.append(np.nan)
                row_errors.append(np.nan)
                continue
            row_values.append(float(
                re.match(r"([-+]?[\d.]+)", number).group(1)
            ))
            row_errors.append(float(
                re.match(r"([-+]?[\d.]+\s*±\s*([\d.]+))", number).group(2)
            ))

        results.append(row_values)
        errors.append(row_errors)
    return results, errors


def rank_list(in_list: list) -> list:
    return np.argsort(in_list)


def create_table(data, results, datasets, models, name):
    clean_name = ""
    name_split = name.split(" ")
    for part in name_split:
        clean_name += part
    ranking_data = rank_results(results)

    results = results.to_numpy()

    mask = results == 'na'
    out = np.empty(results.shape, dtype=float)
    out[mask] = np.nan
    out[~mask] = results[~mask].astype(float)
    results = out

    data.insert(0, ranking_data['average_ranks'])

    row_color_modifier = [0, 50]
    row_colors = [3 * [1.0], 3 * [(255 - row_color_modifier[1])/255]]
    cell_colors = [
        [row_colors[i % 2]] * (len(models)) for i in range(len(datasets) + 1)
    ]

    rank_row_ranks = rank_list(data[0])
    cell_colors[0][rank_row_ranks[0]] = (212/255, 175/255, 55/255)
    cell_colors[0][rank_row_ranks[1]] = (165/255, 169/255, 180/255)
    cell_colors[0][rank_row_ranks[2]] = (191/255, 137/255, 112/255)

    for i in range(1, len(datasets) + 1):
        row = np.array(results[i-1])
        row_ranks = rank_list(np.nan_to_num(row, nan=0.0))[::-1]

        cell_colors[i][row_ranks[0]] = (212/255, 175/255, 55/255)
        cell_colors[i][row_ranks[1]] = (165/255, 169/255, 180/255)
        cell_colors[i][row_ranks[2]] = (191/255, 137/255, 112/255)

    data = np.array(data)

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(len(models)/2, len(datasets)/3), dpi=300)
    ax.axis('off')

    table: Table = ax.table(
        cellText=data,
        colLabels=models,
        cellColours=cell_colors,
        rowLabels=["Ranks"] + datasets,
        loc="center",
        cellLoc="center"
    )

    for i, model in enumerate(models):
        if "rdkit" in model:
            table[0, i].set_facecolor("pink")
        elif "minimol" in model:
            table[0, i].set_facecolor("lightgreen")
        elif "wwl" in model:
            table[0, i].set_facecolor("violet")
        elif "moe" in model:
            table[0, i].set_facecolor("white")
        else:
            table[0, i].set_facecolor("lightblue")

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    table.auto_set_column_width(col=list(range(len(models))))

    legend_labels = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='gold', markersize=10, label='First'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='silver', markersize=10, label='Second'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='sandybrown', markersize=10, label='Third'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='violet', markersize=10, label='WWL'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='lightblue', markersize=10, label='GNN'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='pink', markersize=10, label='RDKit'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='lightgreen', markersize=10, label='Minimol'),
    ]

    ax.legend(handles=legend_labels, loc='upper left',
              bbox_to_anchor=(-1, 0), title="Legend", frameon=False)

    if ranking_data['iman_davenport']['p_value'] < 0.05:
        title = f"{name} overview, ranks are significant with alpha=0.05"
    else:
        title = f"{name} overview, ranks are not significant with alpha=0.05"

    plt.title(title, fontsize=20, fontweight='bold', pad=30)
    plt.tight_layout(pad=3)
    plt.savefig(f"tables/{name}.pdf", format='pdf', bbox_inches='tight')


def remove_na(results, datasets, models):
    '''
    results: results from load_data(datasets, models)
    datasets: all datasets
    models: all models
    '''
    trimmed_datasets = []
    trimmed_results = []
    for dataset, row in zip(datasets, results):
        if "na" not in row:
            trimmed_datasets.append(dataset)
            trimmed_results.append(row)
    return trimmed_results, trimmed_datasets


def create_table_from(datasets, models, name):
    results = pd.read_csv("./summarized_results.csv", index_col=0)
    data = load_data(datasets, models)
    create_table(data, results, datasets, models, name)


def main():
    create_table_from(Datasets, Models, "Complete")

    results = load_data(Datasets, Models)
    trimmed_results, trimmed_datasets = remove_na(
        results, Datasets, Models)
    if (len(trimmed_datasets) > 2):
        create_table_from(
            trimmed_datasets, Models, "Data available"
        )

    create_table_from(ClassificationTask, Models, "Classification")

    create_table_from(Regression_Datasets, Models, "Regression")

    create_table_from(TrueSingleTask, Models, "True Single Task")

    multitask_datasets = Multitask_Datasets.copy()
    True_Models = SingleTask + multitask_datasets
    true_models = True_Models.copy()
    true_models.remove("clintox1")
    true_models.remove("clintox2")
    create_table_from(true_models, Models, "Original")
    multitask_datasets.remove("clintox")
    True_plus_clintox = SingleTask + multitask_datasets
    create_table_from(True_plus_clintox, Models, "Original but Clintox split")

    descriptor_means = pd.read_csv(
        "./descriptor_means.csv", index_col=0, dtype=str)
    descriptors = ["RDKit", "MiniMol", "MOE"]
    create_table(dataframe_to_list(descriptor_means), descriptor_means,
                 Datasets, descriptors, "Ranking by mean descriptor results")


if __name__ == '__main__':
    main()
