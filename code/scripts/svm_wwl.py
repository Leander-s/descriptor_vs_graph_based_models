from util import RunParameters
import time
import warnings
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, precision_recall_curve, auc,
    mean_squared_error, r2_score, mean_absolute_error)
from sklearn.model_selection import train_test_split
from sklearn import svm
import multiprocessing
from task_dict import tasks_dic
from wwl import wwl
from rdkit import Chem
import igraph as ig


def smiles_to_igraph(smiles):
    if type(smiles) is not str:
        print(f"Smiles were not string : {smiles}")
        exit(1)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # num_atoms = mol.GetNumAtoms() chat gippity wrote this, doesn't get used..
    edges = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
             for bond in mol.GetBonds()]

    g = ig.Graph(edges=edges, directed=False)
    g.vs["label"] = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    return g


def standardize(col):
    return (col - np.mean(col)) / np.std(col)


# the metrics for classification
def statistical(y_true, y_pred, y_pro):
    c_mat = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = list(c_mat.flatten())
    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    acc = (tp + tn) / (tn + fp + fn + tp)
    mcc = ((tp * tn - fp * fn) /
           np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8))
    auc_prc = auc(precision_recall_curve(y_true, y_pro, pos_label=1)[1],
                  precision_recall_curve(y_true, y_pro, pos_label=1)[0])
    auc_roc = roc_auc_score(y_true, y_pro)
    return tn, fp, fn, tp, se, sp, acc, mcc, auc_prc, auc_roc


def all_one_zeros(data):
    if (len(np.unique(data)) == 2):
        flag = False
    else:
        flag = True
    return flag


def hyper_runing(task_args):
    (subtask, kernel_matrix, dataset, dataset_label, task_type, OPT_ITERS,
     space_) = task_args
    sub_dataset = dataset

    # get the attentivefp data splits
    data_tr = sub_dataset.index[sub_dataset['group'] == 'train']
    data_va = sub_dataset.index[sub_dataset['group'] == 'valid']
    data_te = sub_dataset.index[sub_dataset['group'] == 'test']

    # prepare data for training
    # training set
    data_tr_x = kernel_matrix[np.ix_(data_tr, data_tr)]
    data_tr_y = dataset.iloc[data_tr][subtask].to_numpy().reshape(-1, 1)
    data_tr_na = data_tr_y[subtask == np.nan]

    np.delete(data_tr_y, data_tr_na, axis=0)
    np.delete(data_tr_x, data_tr_na, axis=0)
    np.delete(data_tr_x, data_tr_na, axis=1)

    # validation set
    data_va_y = dataset.iloc[data_va][subtask].values.reshape(-1, 1)
    data_va_x = kernel_matrix[np.ix_(data_va, data_tr)]
    data_va_na = data_va_y[subtask == np.nan]

    np.delete(data_va_y, data_va_na, axis=0)
    np.delete(data_va_x, data_va_na, axis=0)
    np.delete(data_va_x, data_va_na, axis=1)

    # test set
    data_te_y = dataset.iloc[data_te][subtask].values.reshape(-1, 1)
    data_te_x = kernel_matrix[np.ix_(data_te, data_tr)]
    data_te_na = data_te_y[subtask == np.nan]

    np.delete(data_te_y, data_te_na, axis=0)
    np.delete(data_te_x, data_te_na, axis=0)
    np.delete(data_te_x, data_te_na, axis=1)

    def hyper_opt(args):
        if task_type == "cla":
            model = svm.SVC(
                **args, kernel='precomputed', random_state=1,
                probability=True, class_weight='balanced',
                cache_size=2000, max_iter=10000)
        else:
            model = svm.SVR(
                **args, kernel='precomputed',
                cache_size=2000, max_iter=10000)
        model.fit(data_tr_x, data_tr_y)
        val_preds = model.predict_proba(
            data_va_x) if task_type == 'cla' else model.predict(data_va_x)
        if task_type == "cla":
            loss = 1 - roc_auc_score(data_va_y, val_preds[:, 1])
        else:
            np.sqrt(mean_squared_error(data_va_y, val_preds))
        return {'loss': loss, 'status': STATUS_OK}

    # start hyper-parameters optimization
    trials = Trials()
    best_results = fmin(hyper_opt, space_, algo=tpe.suggest,
                        max_evals=OPT_ITERS, trials=trials,
                        show_progressbar=False)
    print('the best hyper-parameters for ' + dataset_label +
          ' ' + subtask + ' are:  ', best_results)
    if task_type == "cla":
        best_model = svm.SVC(
            C=best_results['C'], gamma=best_results['gamma'],
            kernel='precomputed', random_state=1, probability=True,
            class_weight='balanced', cache_size=2000, max_iter=10000)
    else:
        best_model = svm.SVR(
            C=best_results['C'], gamma=best_results['gamma'],
            kernel='precomputed', cache_size=2000, max_iter=10000)
    best_model.fit(data_tr_x, data_tr_y)
    num_of_compounds = len(sub_dataset)

    if task_type == 'cla':
        # training set
        tr_pred = best_model.predict_proba(data_tr_x)
        tr_results = [
            dataset_label, subtask, 'tr', 1, num_of_compounds,
            data_tr_y[data_tr_y == 1].shape[0],
            data_tr_y[data_tr_y == 0].shape[0],
            data_tr_y[data_tr_y == 0].shape[0] /
            data_tr_y[data_tr_y == 1].shape[0],
            best_results['C'],
            best_results['gamma']]
        tr_results.extend(statistical(
            data_tr_y, np.argmax(tr_pred, axis=1), tr_pred[:, 1]))

        # validation set
        va_pred = best_model.predict_proba(data_va_x)
        va_results = [
            dataset_label, subtask, 'va', 1, num_of_compounds,
            data_va_y[data_va_y == 1].shape[0],
            data_va_y[data_va_y == 0].shape[0],
            data_va_y[data_va_y == 0].shape[0] /
            data_va_y[data_va_y == 1].shape[0],
            best_results['C'],
            best_results['gamma']]
        va_results.extend(statistical(
            data_va_y, np.argmax(va_pred, axis=1), va_pred[:, 1]))

        # test set
        te_pred = best_model.predict_proba(data_te_x)
        te_results = [
            dataset_label, subtask, 'te', 1, num_of_compounds,
            data_te_y[data_te_y == 1].shape[0],
            data_te_y[data_te_y == 0].shape[0],
            data_te_y[data_te_y == 0].shape[0] /
            data_te_y[data_te_y == 1].shape[0],
            best_results['C'],
            best_results['gamma']]
        te_results.extend(statistical(
            data_te_y, np.argmax(te_pred, axis=1), te_pred[:, 1]))
    else:
        # training set
        tr_pred = best_model.predict(data_tr_x)
        tr_results = [dataset_label, subtask, 'tr', 1, num_of_compounds,
                      best_results['C'],
                      best_results['gamma'],
                      np.sqrt(mean_squared_error(data_tr_y, tr_pred)
                              ), r2_score(data_tr_y, tr_pred),
                      mean_absolute_error(data_tr_y, tr_pred)]

        # validation set
        va_pred = best_model.predict(data_va_x)
        va_results = [dataset_label, subtask, 'va', 1, num_of_compounds,
                      best_results['C'],
                      best_results['gamma'],
                      np.sqrt(mean_squared_error(data_va_y, va_pred)
                              ), r2_score(data_va_y, va_pred),
                      mean_absolute_error(data_va_y, va_pred)]

        # test set
        te_pred = best_model.predict(data_te_x)
        te_results = [dataset_label, subtask, 'te', 1, num_of_compounds,
                      best_results['C'],
                      best_results['gamma'],
                      np.sqrt(mean_squared_error(data_te_y, te_pred)
                              ), r2_score(data_te_y, te_pred),
                      mean_absolute_error(data_te_y, te_pred)]
    return tr_results, va_results, te_results


def best_model_runing(args):
    (split, task_type, sub_dataset, kernel_matrix, subtask, best_hyper,
     dataset_label) = args
    seed = split
    n_samples = sub_dataset.shape[0]
    indices = np.arange(n_samples)
    print(f"Split {split} running")
    if task_type == 'cla':
        while True:
            training_indices, data_te = train_test_split(
                indices, test_size=0.1, random_state=seed)
            # the training set was further splited into the training set and
            # validation set
            data_tr, data_va = train_test_split(
                training_indices, test_size=0.1, random_state=seed)
            if (
                    all_one_zeros(sub_dataset.iloc[data_tr][subtask]) or
                    all_one_zeros(sub_dataset.iloc[data_va][subtask]) or
                    all_one_zeros(sub_dataset.iloc[data_te][subtask])
            ):
                print(
                    "\ninvalid random seed {} due to one class presented in "
                    "the {} splitted sets..."
                    .format(
                        seed, subtask))
                print("Changing to another random seed...\n")
                seed = np.random.randint(50, 999999)
            else:
                print("random seed used in repetition {} is {}"
                      .format(split, seed))
                break
    else:
        training_indices, data_te = train_test_split(
            indices, test_size=0.1, random_state=seed)
        # the training set was further splited into the training set and
        # validation set
        data_tr, data_va = train_test_split(
            training_indices, test_size=0.1, random_state=seed)

    # prepare data for training
    # training set
    data_tr_x = kernel_matrix[np.ix_(data_tr, data_tr)]
    data_tr_y = sub_dataset.iloc[data_tr][subtask].to_numpy().reshape(-1, 1)
    data_tr_na = data_tr_y.index[subtask == np.nan]

    np.delete(data_tr_y, data_tr_na, axis=0)
    np.delete(data_tr_x, data_tr_na, axis=0)
    np.delete(data_tr_x, data_tr_na, axis=1)

    # validation set
    data_va_y = sub_dataset.iloc[data_va][subtask].values.reshape(-1, 1)
    data_va_x = kernel_matrix[np.ix_(data_va, data_tr)]
    data_va_na = data_va_y[subtask == np.nan]

    np.delete(data_va_y, data_va_na, axis=0)
    np.delete(data_va_x, data_va_na, axis=0)
    np.delete(data_va_x, data_va_na, axis=1)

    # test set
    data_te_y = sub_dataset.iloc[data_te][subtask].values.reshape(-1, 1)
    data_te_x = kernel_matrix[np.ix_(data_te, data_tr)]
    data_te_na = data_te_y[subtask == np.nan]

    np.delete(data_te_y, data_te_na, axis=0)
    np.delete(data_te_x, data_te_na, axis=0)
    np.delete(data_te_x, data_te_na, axis=1)

    if task_type == "cla":
        model = svm.SVC(
            C=best_hyper[best_hyper.subtask == subtask].iloc[0,]['C'],
            gamma=best_hyper[best_hyper.subtask == subtask].iloc[0,]['gamma'],
            kernel='precomputed', random_state=1, probability=True,
            class_weight='balanced', cache_size=2000, max_iter=10000)
    else:
        model = svm.SVR(
            C=best_hyper[best_hyper.subtask == subtask].iloc[0,]['C'],
            gamma=best_hyper[best_hyper.subtask == subtask].iloc[0,]['gamma'],
            kernel='precomputed', cache_size=2000, max_iter=10000)

    model.fit(data_tr_x, data_tr_y)
    num_of_compounds = sub_dataset.shape[0]
    if task_type == 'cla':
        # training set
        tr_pred = model.predict_proba(data_tr_x)
        tr_results = [
            split, dataset_label, subtask, 'tr', 1, num_of_compounds,
            np.array(data_tr_y)[data_tr_y == 1].shape[0],
            np.array(data_tr_y)[data_tr_y == 0].shape[0],
            np.array(data_tr_y)[data_tr_y == 0].shape[0] /
            np.array(data_tr_y)[data_tr_y == 1].shape[0]]
        tr_results.extend(statistical(
            data_tr_y, np.argmax(tr_pred, axis=1), tr_pred[:, 1]))

        # validation set
        va_pred = model.predict_proba(data_va_x)
        va_results = [
            split, dataset_label, subtask, 'va', 1, num_of_compounds,
            np.array(data_va_y)[data_va_y == 1].shape[0],
            np.array(data_va_y)[data_va_y == 0].shape[0],
            np.array(data_va_y)[data_va_y == 0].shape[0] /
            np.array(data_va_y)[data_va_y == 1].shape[0]]
        va_results.extend(statistical(
            data_va_y, np.argmax(va_pred, axis=1), va_pred[:, 1]))

        # test set
        te_pred = model.predict_proba(data_te_x)
        te_results = [
            split, dataset_label, subtask, 'te', 1, num_of_compounds,
            np.array(data_te_y)[data_te_y == 1].shape[0],
            np.array(data_te_y)[data_te_y == 0].shape[0],
            np.array(data_te_y)[data_te_y == 0].shape[0] /
            np.array(data_te_y)[data_te_y == 1].shape[0]]
        te_results.extend(statistical(
            data_te_y, np.argmax(te_pred, axis=1), te_pred[:, 1]))
    else:
        # training set
        tr_pred = model.predict(data_tr_x)
        tr_results = [split, dataset_label, subtask, 'tr', 1, num_of_compounds,
                      np.sqrt(mean_squared_error(data_tr_y, tr_pred)
                              ), r2_score(data_tr_y, tr_pred),
                      mean_absolute_error(data_tr_y, tr_pred)]

        # validation set
        va_pred = model.predict(data_va_x)
        va_results = [split, dataset_label, subtask, 'va', 1, num_of_compounds,
                      np.sqrt(mean_squared_error(data_va_y, va_pred)
                              ), r2_score(data_va_y, va_pred),
                      mean_absolute_error(data_va_y, va_pred)]

        # test set
        te_pred = model.predict(data_te_x)
        te_results = [split, dataset_label, subtask, 'te', 1, num_of_compounds,
                      np.sqrt(mean_squared_error(data_te_y, te_pred)
                              ), r2_score(data_te_y, te_pred),
                      mean_absolute_error(data_te_y, te_pred)]
    return tr_results, va_results, te_results


def run_svm_wwl(
        parameters: RunParameters
) -> (pd.DataFrame, float):
    start = time.time()
    warnings.filterwarnings("ignore")
    # feature_selection = False
    task_type = parameters.task_type  # 'reg' or 'cla'
    dataset_label = parameters.label
    tasks = tasks_dic[dataset_label]
    OPT_ITERS = parameters.hyperOptIters
    repetitions = parameters.evalIters
    dataset = parameters.dataset
    num_pools = 28
    space_ = {
        'C': hp.uniform('C', 0.01, 100),
        'gamma': hp.uniform('gamma', 0, 0.2)}

    pd_res = []

    data = [smiles_to_igraph(s) for s in dataset.iloc[:, 0].values]

    kernel_matrix = wwl(data)

    task_args = [
        (task, kernel_matrix, dataset, dataset_label, task_type, OPT_ITERS,
         space_) for task in tasks
    ]

    pool = multiprocessing.Pool(num_pools)
    res = pool.map(hyper_runing, task_args)
    pool.close()
    pool.join()
    for item in res:
        for i in range(3):
            pd_res.append(item[i])
    if task_type == 'cla':
        best_hyper = pd.DataFrame(
            pd_res, columns=[
                'dataset', 'subtask', 'set', 'num_of_retained_feature',
                'num_of_compounds', 'postives', 'negtives',
                'negtives/postives', 'C', 'gamma', 'tn', 'fp', 'fn', 'tp',
                'se', 'sp', 'acc', 'mcc', 'auc_prc', 'auc_roc'])
    else:
        best_hyper = pd.DataFrame(
            pd_res, columns=[
                'dataset', 'subtask', 'set', 'num_of_retained_feature',
                'num_of_compounds', 'C', 'gamma', 'rmse', 'r2', 'mae'])

    best_hyper.to_csv(
        f"./stat_res/hyperopts/{dataset_label}"
        "_moe_pubsub_svm_hyperopt_info.csv", index=0)

    if task_type == 'cla':
        print("train", best_hyper[best_hyper['set'] == 'tr']['auc_roc'].mean(),
              best_hyper[best_hyper['set'] == 'tr']['auc_prc'].mean())
        print("valid", best_hyper[best_hyper['set'] == 'va']['auc_roc'].mean(),
              best_hyper[best_hyper['set'] == 'va']['auc_prc'].mean())
        print("test", best_hyper[best_hyper['set'] == 'te']['auc_roc'].mean(),
              best_hyper[best_hyper['set'] == 'te']['auc_prc'].mean())
    else:
        print("train", best_hyper[best_hyper['set'] == 'tr']['rmse'].mean(),
              best_hyper[best_hyper['set'] == 'tr']['r2'].mean(),
              best_hyper[best_hyper['set'] == 'tr']['mae'].mean())
        print("valid", best_hyper[best_hyper['set'] == 'va']['rmse'].mean(),
              best_hyper[best_hyper['set'] == 'va']['r2'].mean(),
              best_hyper[best_hyper['set'] == 'va']['mae'].mean())
        print("test", best_hyper[best_hyper['set'] == 'te']['rmse'].mean(),
              best_hyper[best_hyper['set'] == 'te']['r2'].mean(),
              best_hyper[best_hyper['set'] == 'te']['mae'].mean())

    # 50 repetitions based on thr best hypers
    dataset.drop(columns=['group'], inplace=True)
    pd_res = []

    for subtask in tasks:
        # for split in range(1, splits+1):
        args = [(
            split, task_type, dataset, kernel_matrix, subtask, best_hyper,
            dataset_label) for split in range(1, repetitions+1)]
        pool = multiprocessing.Pool(num_pools)
        res = pool.map(best_model_runing, args)
        pool.close()
        pool.join()
        for item in res:
            for i in range(3):
                pd_res.append(item[i])
    if task_type == 'cla':
        stat_res = pd.DataFrame(
            pd_res, columns=[
                'split', 'dataset', 'subtask', 'set',
                'num_of_retained_feature', 'num_of_compounds', 'postives',
                'negtives', 'negtives/postives', 'tn', 'fp', 'fn', 'tp',
                'se', 'sp', 'acc', 'mcc', 'auc_prc', 'auc_roc'])
    else:
        stat_res = pd.DataFrame(
            pd_res, columns=[
                'split', 'dataset', 'subtask', 'set',
                'num_of_retained_feature', 'num_of_compounds', 'rmse',
                'r2', 'mae'])
    if len(tasks) == 1:
        args = {
            'data_label': dataset_label,
            'metric': 'auc_roc' if task_type == 'cla' else 'rmse',
            'model': 'SVM'}
        print("{}_{}: the mean {} for the training set is {:.3f} with std "
              "{:.3f}"
              .format(
                  args['data_label'], args['model'], args['metric'],
                  np.mean(stat_res[stat_res['set'] == 'tr'][args['metric']]),
                  np.std(stat_res[stat_res['set'] == 'tr'][args['metric']])))
        print(
            "{}_{}: the mean {} for the validation set is {:.3f} with std "
            "{:.3f}"
            .format(
                args['data_label'], args['model'], args['metric'],
                np.mean(stat_res[stat_res['set'] == 'va'][args['metric']]),
                np.std(stat_res[stat_res['set'] == 'va'][args['metric']])))
        print("{}_{}: the mean {} for the test set is {:.3f} with std {:.3f}"
              .format(
                  args['data_label'], args['model'], args['metric'],
                  np.mean(stat_res[stat_res['set'] == 'te'][args['metric']]),
                  np.std(stat_res[stat_res['set'] == 'te'][args['metric']])))
# multi-tasks
    else:
        args = {
            'data_label': dataset_label,
            'metric': 'auc_roc' if dataset_label != 'muv' else 'auc_prc',
            'model': 'SVM'}
        tr_acc = np.zeros(repetitions)
        va_acc = np.zeros(repetitions)
        te_acc = np.zeros(repetitions)
        for subtask in tasks:
            tr = stat_res[stat_res['set'] == 'tr']
            tr_acc = tr_acc + tr[tr['subtask'] ==
                                 subtask][args['metric']].values

            va = stat_res[stat_res['set'] == 'va']
            va_acc = va_acc + va[va['subtask'] ==
                                 subtask][args['metric']].values

            te = stat_res[stat_res['set'] == 'te']
            te_acc = te_acc + te[te['subtask'] ==
                                 subtask][args['metric']].values
        tr_acc = tr_acc / len(tasks)
        va_acc = va_acc / len(tasks)
        te_acc = te_acc / len(tasks)
        print("{}_{}: the mean {} for the training set is {:.3f} with std "
              "{:.3f}"
              .format(
                  args['data_label'], args['model'], args['metric'],
                  np.mean(tr_acc), np.std(tr_acc)))
        print(
            "{}_{}: the mean {} for the validation set is {:.3f} with std "
            "{:.3f}"
            .format(
                args['data_label'], args['model'], args['metric'],
                np.mean(va_acc), np.std(va_acc)))
        print("{}_{}: the mean {} for the test set is {:.3f} with std {:.3f}"
              .format(
                  args['data_label'], args['model'], args['metric'],
                  np.mean(te_acc), np.std(te_acc)))
    end = time.time()  # get the end time
    elapsed = end-start
    print("the elapsed time is:", (end - start), "S")

    return stat_res, elapsed
