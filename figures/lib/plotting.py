import csv
import matplotlib.pyplot as plt
import math


def get_average_and_std_auc_roc(file_path):
    try:
        auc_roc_values = []

        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            for row in reader:
                if row.get('set') == 'test' or row.get('set') == 'te':
                    auc_roc_value = row.get('auc_roc')
                    if not auc_roc_value:
                        auc_roc_value = row.get('roc_auc')
                    if auc_roc_value:
                        auc_roc_values.append(float(auc_roc_value))

        if auc_roc_values:
            average = sum(auc_roc_values) / len(auc_roc_values)
            variance = sum((x - average) **
                           2 for x in auc_roc_values) / len(auc_roc_values)
            std_dev = math.sqrt(variance)
            return average, std_dev
        else:
            print(f"No 'test' set or 'auc_roc' values found in the file {file_path}.")
            return None, None
    except FileNotFoundError:
        print(f"file {file_path} was not found.")
        return None, None


def plot_auc_roc(file_list):
    results = []
    for file_path in file_list:
        average_auc_roc, std_dev = get_average_and_std_auc_roc(file_path)
        if average_auc_roc is not None:
            results.append((file_path, average_auc_roc, std_dev))

    # Plot the results
    if results:
        dataset_label = results[0][0].split("_")[0].upper()
        names = [file[0].split("_")[1] for file in results]
        descriptors = [file[0].split("_")[2] if '.' not in file[0].split("_")[2] else 'gnn' for file in results]
        auc_roc_values = [file[1] for file in results]
        std_devs = [file[2] for file in results]

        colors = ['orange' if 'rdkit' in descriptor else 'yellow' if 'minimol' in descriptor else 'lightblue' for descriptor in descriptors]

        names = [name + '-rdkit' if 'rdkit' in descriptor else name + '-minimol' if 'minimol' in descriptor else name for name, descriptor in zip(names,descriptors)]

        plt.figure(figsize=(10, 6))
        bars = plt.barh(names, auc_roc_values, color=colors)
        plt.xlabel('Average AUC ROC')
        plt.title(results[0][0].split("_")[0].upper() + ' AUC_ROC')

        for bar, (avg, std) in zip(bars, zip(auc_roc_values, std_devs)):
            width = bar.get_width()
            plt.text(width / 2, bar.get_y() + bar.get_height() / 2,
                     f"{avg:.3f} ± {std:.3f}", ha='left', va='center', color='black')

        plt.tight_layout()
        plt.savefig(f'plots/{dataset_label}_{descriptors[0]}_{descriptors[-1]}.png')
    else:
        print("No valid AUC ROC values found in any files.")


def get_average_and_std_r2(file_path):
    try:
        r2_values = []

        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            for row in reader:
                if row.get('set') == 'test' or row.get('set') == 'te':
                    r2_value = row.get('r2')
                    if r2_value:
                        r2_values.append(float(r2_value))

        if r2_values:
            average = sum(r2_values) / len(r2_values)
            variance = sum((x - average) **
                           2 for x in r2_values) / len(r2_values)
            std_dev = math.sqrt(variance)
            return average, std_dev
        else:
            print(f"No 'test' set or 'r2' values found in the file {file_path}.")
            return None, None
    except FileNotFoundError:
        print(f"file {file_path} was not found.")
        return None, None


def plot_r2(file_list):
    results = []
    for file_path in file_list:
        average_r2, std_dev = get_average_and_std_r2(file_path)
        if average_r2 is not None:
            results.append((file_path, average_r2, std_dev))

    # Plot the results
    if results:
        dataset_label = results[0][0].split("_")[0].upper()
        names = [file[0].split("_")[1] for file in results]
        descriptors = [file[0].split("_")[2] if '.' not in file[0].split("_")[2] else 'gnn' for file in results]
        r2_values = [file[1] for file in results]
        std_devs = [file[2] for file in results]

        colors = ['orange' if 'rdkit' in descriptor else 'yellow' if 'minimol' in descriptor else 'lightblue' for descriptor in descriptors]

        names = [name + '-rdkit' if 'rdkit' in descriptor else name + '-minimol' if 'minimol' in descriptor else name for name, descriptor in zip(names,descriptors)]

        plt.figure(figsize=(10, 6))
        bars = plt.barh(names, r2_values, color=colors)
        plt.xlabel('Average R2')
        plt.title(dataset_label + ' R2')

        for bar, (avg, std) in zip(bars, zip(r2_values, std_devs)):
            width = bar.get_width()
            plt.text(width / 2, bar.get_y() + bar.get_height() / 2,
                     f"{avg:.3f} ± {std:.3f}", ha='left', va='center', color='black')

        plt.tight_layout()
        plt.savefig(f'plots/{dataset_label}_{descriptors[0]}_{descriptors[-1]}.png')
    else:
        print("No valid R2 values found in any files.")
