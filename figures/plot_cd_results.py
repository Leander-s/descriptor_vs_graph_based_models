from create_table import load_data, load_times, remove_na
from lib.config import (
    Datasets, Models, ClassificationTask, Regression_Datasets, SingleTask, Multitask_Datasets
)
from lib.demsar import demsar
import numpy as np
import matplotlib.pyplot as plt

BASELINE = -0.05


def get_symbols(models):
    symbols = []
    colors = []
    for model in models:
        if "svm" in model:
            symbols.append("o")
        elif "rf" in model:
            symbols.append("s")
        elif "xgb" in model:
            symbols.append("D")
        elif "dnn" in model:
            symbols.append("^")
        elif "gcn" in model:
            symbols.append(".")
        elif "gat" in model:
            symbols.append("*")
        elif "mpnn" in model:
            symbols.append("d")
        else:
            symbols.append("p")

        if "rdkit" in model:
            colors.append("tab:red")
        elif "minimol" in model:
            colors.append("tab:green")
        elif "moe" in model:
            colors.append("black")
        elif "wwl" in model:
            colors.append("violet")
        else:
            colors.append("tab:blue")
    return symbols, colors


def prepare_plot(name: str) -> (plt.Figure, plt.Axes):
    fig, ax = plt.subplots(figsize=(18, 6))

    ax.axhline(y=BASELINE, color='black', linewidth=1, zorder=1)
    # Customize the plot
    ax.set_ylim(-0.1, 0.1)  # Adjust vertical space around points
    ax.set_yticks([])  # Remove y-axis ticks

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis='x', length=0, labelsize=25)

    # Add labels and title
    ax.set_xlabel('Ranks', fontsize=25)
    ax.set_title(name, fontsize=25)

    # Add a grid for better readability
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    ax.invert_xaxis()
    return fig, ax


def plot_models(ax, symbols, colors, models, results):
    for point, symbol, color, model in zip(results['average_ranks'], symbols, colors, models):
        ax.scatter(point, BASELINE, marker=symbol, c=color,
                   label=model, s=700, clip_on=False, zorder=5
                   )


def draw_cd_line(ax, start, end, height):
    ax.plot(
        [start, end],
        [height, height],
        color='black',
        linewidth=2
    )
    ax.plot(
        [start, start],
        [height - 0.001, height + 0.001],
        color='black',
        linewidth=2
    )
    ax.plot(
        [end, end],
        [height - 0.001, height + 0.001],
        color='black',
        linewidth=2
    )


def plot_cd(ax, results, bonferroni_results):
    i = 0
    last_j = 0
    j = 1
    lines = 0
    last_line_j = last_j
    sig = False
    while (True):
        if sig:
            break
        if i == len(results):
            break
        if j == len(results) + 1:
            i += 1
            j = i + 1
            continue
        index_i = results[list(results.keys())[i]]
        if j < len(results):
            index_j = results[list(results.keys())[j]]
            a = min(index_i, index_j)
            b = max(index_i, index_j)
            if a == b:
                break
        else:
            sig = True

        if (sig or bonferroni_results[(a, b)]['significant']) and i != j-1:
            if j != last_line_j or sig:
                draw_cd_line(
                    ax,
                    list(results.keys())[i],
                    list(results.keys())[j-1],
                    BASELINE-0.015 - (lines) * 0.003
                )
                if i >= last_line_j:
                    lines = 0
                lines += 1
            last_line_j = j
            i += 1
            j = i + 1
            continue

        j += 1


def create_plot(datasets, models, name: str):
    used_models = models.copy()

    data = load_data(datasets, used_models)
    data, used_datasets = remove_na(data, datasets, used_models)
    data_values = np.array([[float(entry.split(" ")[0])
                           for entry in row] for row in data])
    demsar_results = demsar(data_values)

    rank_list = demsar_results["average_ranks"]
    rank_dict = dict(sorted(
        {value: index for index, value in enumerate(rank_list)}.items()))

    symbols, colors = get_symbols(used_models)

    fig, ax = prepare_plot(f'Critical Difference Diagram for {name}')

    plot_models(ax, symbols, colors, used_models, demsar_results)

    plot_cd(ax, rank_dict, demsar_results['bonferroni'])

    ax.legend(fontsize="22", loc="upper right", ncol=4)
    plt.tight_layout()
    plt.savefig(f"tables/{name}_cd_plot.pdf", format='pdf')
    plt.savefig(f"tables/{name}_cd_plot.png", format='png')


def create_time_plot(datasets, models, name: str):
    used_models = models.copy()

    data = load_times(datasets, used_models)
    data, used_datasets = remove_na(data, datasets, used_models)
    # time data is negated because lower is better here
    data = -np.array(data, dtype=float)
    demsar_results = demsar(data)

    rank_list = demsar_results["average_ranks"]
    rank_dict = dict(sorted(
        {value: index for index, value in enumerate(rank_list)}.items()))

    symbols, colors = get_symbols(used_models)

    fig, ax = prepare_plot(f'Critical Difference Diagram for {name} times')

    plot_models(ax, symbols, colors, used_models, demsar_results)

    plot_cd(ax, rank_dict, demsar_results['bonferroni'])

    ax.legend(fontsize="22", loc="upper right", ncol=4)
    plt.tight_layout()
    plt.savefig(f"tables/{name}_cd_plot_times.pdf", format='pdf')
    plt.savefig(f"tables/{name}_cd_plot_times.png", format='png')


def main():
    create_plot(Datasets, Models, "Complete Data")
    create_time_plot(Datasets, Models, "Complete Data")
    create_plot(ClassificationTask, Models, "Classification Data")
    create_time_plot(ClassificationTask, Models, "Classification Data")
    create_plot(Regression_Datasets, Models, "Regression Data")
    create_time_plot(Regression_Datasets, Models, "Regression Data")
    create_plot(SingleTask + Multitask_Datasets, Models, "Original")


if __name__ == '__main__':
    main()
