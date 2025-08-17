import sys
import pandas as pd
from util import (DescriptorBasedModels,
                  GraphBasedModels,
                  Descriptors,
                  RunParameters,
                  RegressionSets)
from scripts.svm import run_svm
from scripts.svm_wwl import run_svm_wwl
from scripts.rf import run_rf
from scripts.xgb import run_xgb
from scripts.dnn_torch import run_dnn
from scripts.gcn import run_gcn
from scripts.gat import run_gat
from scripts.mpnn import run_mpnn
from scripts.attentivefp import run_attentivefp
from scripts.ensemble import run_ensemble
import os


'''
Arguments to give:
    1. dataset_name
    2. model_name
    3. descriptors
'''


def interpret_args(argv):
    if (len(argv) < 3):
        raise Exception("Not enough arguments")

    dataset_name: str = argv[1]
    model: str = argv[2]

    descriptors: str = None
    if (len(argv) == 4):
        if argv[3] not in Descriptors:
            raise Exception(f"Invalid descriptors : {argv[3]}")
        descriptors: str = argv[3]

    # Check if model is known
    if model not in DescriptorBasedModels and model not in GraphBasedModels:
        raise Exception(f"Invalid model : {model}")

    # Check if dataset is known and set it to graph dataset
    dataset_path = f"../data/original_datasets/{dataset_name}.csv"
    if not os.path.isfile(dataset_path):
        dataset_path = "../data/additional_datasets/original/"\
            f"{dataset_name}.csv"

    if not os.path.isfile(dataset_path):
        raise Exception(f"Dataset {dataset_name} could not be found")

    # Set dataset to descriptors if dbm
    if model in DescriptorBasedModels:
        if not descriptors:
            raise Exception("Need to specify descriptors")
        dataset_path = f"../data/{descriptors}_datasets/"\
            f"{dataset_name}_{descriptors}_descriptors.csv"
        if not os.path.isfile(dataset_path):
            dataset_path = f"../data/additional_datasets/{descriptors}/"\
                f"{dataset_name}_{descriptors}_descriptors.csv"

    if not os.path.isfile(dataset_path):
        raise Exception("No dataset with "
                        f"{descriptors} descriptors found for {dataset_name}")

    dataset = pd.read_csv(dataset_path)
    mode = 'reg' if dataset_name in RegressionSets else 'cla'
    return dataset, model, dataset_name, descriptors, mode


def main():
    dataset, model, dataset_name, descriptors, mode = interpret_args(sys.argv)

    parameters: RunParameters = RunParameters(
        dataset_name,
        dataset,
        mode
    )

    match model:
        case "svm":
            result, time = run_svm(parameters)
        case "svm-wwl":
            result, time = run_svm_wwl(parameters)
        case "rf":
            result, time = run_rf(parameters)
        case "xgb":
            result, time = run_xgb(parameters)
        case "dnn":
            result, time = run_dnn(parameters)
        case "gcn":
            result, time = run_gcn(parameters)
        case "gat":
            result, time = run_gat(parameters)
        case "mpnn":
            result, time = run_mpnn(parameters)
        case "attentivefp":
            result, time = run_attentivefp(parameters)
        case "ensemble":
            result, time = run_ensemble(parameters)
        case _:
            result, time = None, 0

    if descriptors:
        result_path = "./stat_res/mogon_results/"\
            f"{dataset_name}_{model}_{descriptors}_results.csv"
        time_path = "./stat_res/mogon_results/times/"\
            f"{dataset_name}_{model}_{descriptors}_time.csv"

    else:
        result_path = "./stat_res/mogon_results/"\
            f"{dataset_name}_{model}_results.csv"
        time_path = "./stat_res/mogon_results/times/"\
            f"{dataset_name}_{model}_time.csv"

    result.to_csv(result_path, index=False)
    time_data = {'time': [time]}
    tf = pd.DataFrame(time_data)
    tf.to_csv(time_path, index=False)


if __name__ == '__main__':
    main()
