import pandas as pd

DescriptorBasedModels = ["svm", "rf", "xgb", "dnn"]
GraphBasedModels = ["gcn", "gat", "mpnn", "attentivefp", "svm-wwl"]

Descriptors = ["rdkit", "minimol", "moe-neutral"]

RegressionSets = ["freesolv", "esol", "lipop"]


class RunParameters:
    '''
    label: str = <dataset name>
    dataset: pd.DataFrame = <dataset read from csv>
    task_type: str = <'reg' or 'cla'>
    hyperOptIters: int = <hyper-parameter-optimization iterations>
    evalIters: int = <evaluation iterations>
    '''

    def __init__(
            self, label: str, dataset: pd.DataFrame, task_type: str,
            hyperOptIters: int = 30, evalIters: int = 15
    ):
        self.task_type = task_type
        self.label = label
        self.dataset = dataset
        self.hyperOptIters = hyperOptIters
        self.evalIters = evalIters
