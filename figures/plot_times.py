import pandas as pd
import numpy as np
from lib.config import Datasets, Models
import matplotlib.pyplot as plt


def get_colors(models):
    colors = []
    for model in models:
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
    return colors


def plot_times(datasets, models):
    times_df = pd.read_csv("./summarized_times.csv", index_col=0)
    times_df = times_df.loc[datasets]
    all_model_times = []
    for model in models:
        model_times = [
            time if time != "na" else np.nan for time in
            times_df[model].to_numpy()
        ]
        all_model_times.append(
            np.nanmean(np.array(model_times, dtype=np.float32)))
    all_model_times = np.array(all_model_times)
    colors = get_colors(models)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.bar(models, all_model_times, color=colors)
    plt.xlabel('Mean Time', fontsize=25)
    plt.ylabel('Model', fontsize=25)
    plt.title('Mean Times of Models', fontsize=25)
    plt.xticks(rotation=60, fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("tables/times_plot.pdf", format='pdf')


def main():
    used_models = Models.copy()

    plot_times(Datasets, used_models)


if __name__ == '__main__':
    main()
