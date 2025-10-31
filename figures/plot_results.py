from ttest import DatasetTestResult, test_dataset
from lib.plotting import get_average_and_std_r2, get_average_and_std_auc_roc
import matplotlib.pyplot as plt
from lib.config import Models, Regression_Datasets, Datasets


def plot_dataset(dataset):
    '''
    dataset: name of the dataset to plot
    '''
    reg = dataset in Regression_Datasets
    symbols = []
    colors = []
    means = []
    used_models = []
    for model in Models:
        path = f"{dataset}_{model}_results.csv"
        if reg:
            mean, std = get_average_and_std_r2(path)
        else:
            mean, std = get_average_and_std_auc_roc(path)

        if mean is not None:
            means.append(mean)
            used_models.append(model)

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
            colors.append("red")
        elif "minimol" in model:
            colors.append("green")
        elif "moe" in model:
            colors.append("black")
        elif "wwl" in model:
            colors.append("violet")
        else:
            colors.append("blue")

    fig, ax = plt.subplots(figsize=(12, 10))

    ax.axhline(y=0, color='black', linewidth=1, zorder=1)

    for point, symbol, color, model in zip(means, symbols, colors, used_models):
        ax.scatter(point, 0, marker=symbol, c='None',
                   label=model, s=100, clip_on=False, zorder=5, edgecolors=color
                   )

    # Customize the plot
    ax.set_ylim(-0.1, 0.1)  # Adjust vertical space around points
    ax.set_yticks([])  # Remove y-axis ticks

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis='x', length=0)

    # Add labels and title
    ax.set_xlabel('Mean scores')
    ax.set_title(f'Plot for {dataset}')
    ax.legend()

    # Add a grid for better readability
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"plots/{dataset}_plot.pdf", format='pdf')
    plt.close()


def main():
    for dataset in Datasets:
        plot_dataset(dataset)


if __name__ == '__main__':
    main()
