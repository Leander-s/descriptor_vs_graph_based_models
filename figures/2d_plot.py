from create_table import load_data, remove_na
from lib.config import Datasets, Models
from lib.demsar import demsar
from get_times import get_times
import matplotlib.pyplot as plt
import numpy as np


def create_plot():
    used_models = Models.copy()

    data = load_data(Datasets, used_models)
    data, used_datasets = remove_na(data, Datasets, used_models)

    np_data = np.array([[float(cell.split(" ")[0])
                       for cell in row] for row in data])
    ranks = demsar(np_data)['average_ranks']

    times = get_times(used_datasets, used_models)
    np_times = np.array([[float(time) for time in row] for row in times])
    max_time = np.max(np_times)
    time_ranks = demsar(max_time - np_times)['average_ranks']

    time_differences = demsar(max_time - np_times)['bonferroni']

    result_string = ""

    for i in range(len(used_models)):
        for j in range(i+1, len(used_models)):
            comparism = f"{used_models[i]}, {used_models[j]} : {time_differences[(i, j)]['significant']}\n"
            result_string += comparism

    f = open("time_comparism.txt", "w")
    f.write(result_string)
    f.close()

    x = ranks
    y = time_ranks

    symbols = []
    colors = []
    for model in used_models:
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

    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 11))
    for xi, yi, sym, model, color in zip(x, y, symbols, used_models, colors):
        ax.scatter(xi, yi, marker=sym, color=color, s=800, label=model)

    # Add axis labels
    ax.set_xlabel('Rank', fontsize=45)
    ax.set_ylabel('Time', fontsize=45)

    # Add annotations on the axes or elsewhere
    ax.annotate('Resource axis label', xy=(2, 0), xytext=(1.5, -1),
                arrowprops=dict(facecolor='black', arrowstyle='->'))
    ax.annotate('Performance axis label', xy=(0, 2), xytext=(-1, 1.5),
                arrowprops=dict(facecolor='black', arrowstyle='->'))

    # Set grid and show the plot
    ax.grid(True, linestyle='--', alpha=0.5)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.legend(fontsize="24", loc="center left", bbox_to_anchor=(1.0, 0.5))

    plt.xticks(fontsize="35")
    plt.yticks(fontsize="35")
    plt.savefig("tables/2d_plot.pdf", format='pdf')
    plt.savefig("tables/2d_plot.png", format='png')


def main():
    # Define the data points and their respective symbols
    create_plot()


if __name__ == '__main__':
    main()
