from util import RunParameters
import time
import warnings
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, precision_recall_curve, auc,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.model_selection import train_test_split
from sklearn import svm
import multiprocessing
from task_dict import tasks_dic


def standardize(col):
    return (col - np.mean(col)) / np.std(col)


# the metrics for classification
def statistical(y_true, y_pred, y_pro):
    c_mat = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tn, fp, fn, tp = list(c_mat.flatten())
    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    acc = (tp + tn) / (tn + fp + fn + tp)
    mcc = (
        (tp * tn - fp * fn) /
        np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
    )
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
    subtask, dataset, dataset_label, task_type, OPT_ITERS, space_ = task_args
    cols = [subtask]
    cols.extend(dataset.columns[(len(tasks_dic[dataset_label]) + 1):])
    sub_dataset = dataset[cols]

    # detect the na in the subtask (y cloumn)
    rm_index = sub_dataset[subtask][sub_dataset[subtask].isnull()].index
    sub_dataset.drop(index=rm_index, inplace=True)

    # remove the features with na
    if dataset_label != 'hiv':
        sub_dataset = sub_dataset.dropna(axis=1)
    else:
        sub_dataset = sub_dataset.dropna(axis=0)
    # *******************
    # demension reduction
    # *******************
    # Removing features with low variance
    # threshold = 0.05
    data_fea_var = sub_dataset.iloc[:, 2:].var()
    del_fea1 = list(data_fea_var[data_fea_var <= 0.05].index)
    sub_dataset.drop(columns=del_fea1, inplace=True)

    # pair correlations
    # threshold = 0.95
    data_fea_corr = sub_dataset.iloc[:, 2:].corr()
    del_fea2_col = []
    del_fea2_ind = []
    length = data_fea_corr.shape[1]
    for i in range(length):
        for j in range(i + 1, length):
            if abs(data_fea_corr.iloc[i, j]) >= 0.95:
                del_fea2_col.append(data_fea_corr.columns[i])
                del_fea2_ind.append(data_fea_corr.index[j])
    sub_dataset.drop(columns=del_fea2_ind, inplace=True)

    # standardize the features
    cols_ = list(sub_dataset.columns)[2:]
    sub_dataset[cols_] = sub_dataset[cols_].apply(standardize, axis=0)

    # get the attentivefp data splits
    data_tr = sub_dataset[sub_dataset['group'] == 'train']
    data_va = sub_dataset[sub_dataset['group'] == 'valid']
    data_te = sub_dataset[sub_dataset['group'] == 'test']

    # prepare data for training
    # training set
    data_tr_y = data_tr[subtask].values.reshape(-1, 1)
    data_tr_x = np.array(data_tr.iloc[:, 2:].values)

    # validation set
    data_va_y = data_va[subtask].values.reshape(-1, 1)
    data_va_x = np.array(data_va.iloc[:, 2:].values)

    # test set
    data_te_y = data_te[subtask].values.reshape(-1, 1)
    data_te_x = np.array(data_te.iloc[:, 2:].values)

    num_fea = data_tr_x.shape[1]
    print('the num of retained features for the ' +
          dataset_label + ' ' + subtask + ' is:', num_fea)

    def hyper_opt(args):
        if task_type == "cla":
            model = svm.SVC(
                **args, kernel='rbf', random_state=1,
                probability=True, class_weight='balanced',
                cache_size=2000, max_iter=10000
            )
        else:
            model = svm.SVR(
                **args, kernel='rbf', cache_size=2000, max_iter=10000
            )
        model.fit(data_tr_x, data_tr_y)
        val_preds = model.predict_proba(
            data_va_x) if task_type == 'cla' else model.predict(data_va_x)
        if task_type == "cla":
            loss = 1 - roc_auc_score(data_va_y, val_preds[:, 1])
        else:
            loss = np.sqrt(mean_squared_error(data_va_y, val_preds))
        return {'loss': loss, 'status': STATUS_OK}

    # start hyper-parameters optimization
    trials = Trials()
    best_results = fmin(
        hyper_opt, space_, algo=tpe.suggest, max_evals=OPT_ITERS,
        trials=trials, show_progressbar=False
    )
    print("the best hyper-parameters for " + dataset_label + " " + subtask +
          " are:  ", best_results
          )
    if task_type == "cla":
        best_model = svm.SVC(
            C=best_results['C'], gamma=best_results['gamma'], kernel="rbf",
            random_state=1, probability=True, class_weight="balanced",
            cache_size=2000, max_iter=10000
        )
    else:
        best_model = svm.SVR(
            C=best_results['C'], gamma=best_results['gamma'], kernel="rbf",
            cache_size=2000, max_iter=10000
        )
    best_model.fit(data_tr_x, data_tr_y)
    num_of_compounds = len(sub_dataset)

    if task_type == 'cla':
        # training set
        tr_pred = best_model.predict_proba(data_tr_x)
        tr_results = [
            dataset_label, subtask, "tr", num_fea, num_of_compounds,
            data_tr_y[data_tr_y == 1].shape[0],
            data_tr_y[data_tr_y == 0].shape[0],
            data_tr_y[data_tr_y == 0].shape[0] /
            data_tr_y[data_tr_y == 1].shape[0],
            best_results['C'],
            best_results['gamma']
        ]
        tr_results.extend(statistical(
            data_tr_y, np.argmax(tr_pred, axis=1), tr_pred[:, 1]))

        # validation set
        va_pred = best_model.predict_proba(data_va_x)
        va_results = [
            dataset_label, subtask, "va", num_fea, num_of_compounds,
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
            dataset_label, subtask, "te", num_fea, num_of_compounds,
            data_te_y[data_te_y == 1].shape[0],
            data_te_y[data_te_y == 0].shape[0],
            data_te_y[data_te_y == 0].shape[0] /
            data_te_y[data_te_y == 1].shape[0],
            best_results['C'],
            best_results['gamma']
        ]
        te_results.extend(statistical(
            data_te_y, np.argmax(te_pred, axis=1), te_pred[:, 1]))
    else:
        # training set
        tr_pred = best_model.predict(data_tr_x)
        tr_results = [dataset_label, subtask, 'tr', num_fea, num_of_compounds,
                      best_results['C'],
                      best_results['gamma'],
                      np.sqrt(mean_squared_error(data_tr_y, tr_pred)
                              ), r2_score(data_tr_y, tr_pred),
                      mean_absolute_error(data_tr_y, tr_pred)]

        # validation set
        va_pred = best_model.predict(data_va_x)
        va_results = [dataset_label, subtask, 'va', num_fea, num_of_compounds,
                      best_results['C'],
                      best_results['gamma'],
                      np.sqrt(mean_squared_error(data_va_y, va_pred)
                              ), r2_score(data_va_y, va_pred),
                      mean_absolute_error(data_va_y, va_pred)]

        # test set
        te_pred = best_model.predict(data_te_x)
        te_results = [dataset_label, subtask, 'te', num_fea, num_of_compounds,
                      best_results['C'],
                      best_results['gamma'],
                      np.sqrt(mean_squared_error(data_te_y, te_pred)
                              ), r2_score(data_te_y, te_pred),
                      mean_absolute_error(data_te_y, te_pred)]
    return tr_results, va_results, te_results


def best_model_runing(args):
    split, task_type, sub_dataset, subtask, best_hyper, dataset_label = args
    seed = split
    print(f"Split {split} running")
    if task_type == 'cla':
        while True:
            training_data, data_te = train_test_split(
                sub_dataset, test_size=0.1, random_state=seed)
            # the training set was further splited into the training set and
            # validation set
            data_tr, data_va = train_test_split(
                training_data, test_size=0.1, random_state=seed)
            if (
                    all_one_zeros(data_tr[subtask]) or
                    all_one_zeros(data_va[subtask]) or
                    all_one_zeros(data_te[subtask])
            ):
                print("\ninvalid random seed {} due to one class presented in"
                      " the {} splitted sets..."
                      .format(seed, subtask))
                print("Changing to another random seed...\n")
                seed = np.random.randint(50, 999999)
            else:
                print("random seed used in repetition {} is {}"
                      .format(split, seed))
                break
    else:
        training_data, data_te = train_test_split(
            sub_dataset, test_size=0.1, random_state=seed)
        # the training set was further splited into the training set and
        # validation set
        data_tr, data_va = train_test_split(
            training_data, test_size=0.1, random_state=seed)

    # prepare data for training
    # training set
    data_tr_y = data_tr[subtask].values.reshape(-1, 1)
    data_tr_x = np.array(data_tr.iloc[:, 1:].values)

    # validation set
    data_va_y = data_va[subtask].values.reshape(-1, 1)
    data_va_x = np.array(data_va.iloc[:, 1:].values)

    # test set
    data_te_y = data_te[subtask].values.reshape(-1, 1)
    data_te_x = np.array(data_te.iloc[:, 1:].values)

    num_fea = data_tr_x.shape[1]
    if task_type == "cla":
        model = svm.SVC(
            C=best_hyper[best_hyper.subtask == subtask].iloc[0,]['C'],
            gamma=best_hyper[
                best_hyper.subtask == subtask].iloc[0,]['gamma'],
            kernel="rbf", random_state=1, probability=True,
            class_weight="balanced", cache_size=2000, max_iter=10000
        )
    else:
        model = svm.SVR(
            C=best_hyper[best_hyper.subtask == subtask].iloc[0,]['C'],
            gamma=best_hyper[
                best_hyper.subtask == subtask].iloc[0,]['gamma'],
            kernel="rbf", cache_size=2000, max_iter=10000)

    model.fit(data_tr_x, data_tr_y)
    num_of_compounds = sub_dataset.shape[0]
    if task_type == 'cla':
        # training set
        tr_pred = model.predict_proba(data_tr_x)
        tr_results = [
            split, dataset_label, subtask, 'tr', num_fea, num_of_compounds,
            data_tr_y[data_tr_y == 1].shape[0],
            data_tr_y[data_tr_y == 0].shape[0],
            data_tr_y[data_tr_y == 0].shape[0] /
            data_tr_y[data_tr_y == 1].shape[0]]
        tr_results.extend(statistical(
            data_tr_y, np.argmax(tr_pred, axis=1), tr_pred[:, 1]))

        # validation set
        va_pred = model.predict_proba(data_va_x)
        va_results = [
            split, dataset_label, subtask, 'va', num_fea, num_of_compounds,
            data_va_y[data_va_y == 1].shape[0],
            data_va_y[data_va_y == 0].shape[0],
            data_va_y[data_va_y == 0].shape[0] /
            data_va_y[data_va_y == 1].shape[0]]
        va_results.extend(statistical(
            data_va_y, np.argmax(va_pred, axis=1), va_pred[:, 1]))

        # test set
        te_pred = model.predict_proba(data_te_x)
        te_results = [
            split, dataset_label, subtask, 'te', num_fea, num_of_compounds,
            data_te_y[data_te_y == 1].shape[0],
            data_te_y[data_te_y == 0].shape[0],
            data_te_y[data_te_y == 0].shape[0] /
            data_te_y[data_te_y == 1].shape[0]]
        te_results.extend(statistical(
            data_te_y, np.argmax(te_pred, axis=1), te_pred[:, 1]))
    else:
        # training set
        tr_pred = model.predict(data_tr_x)
        tr_results = [
            split, dataset_label, subtask, 'tr', num_fea, num_of_compounds,
            np.sqrt(mean_squared_error(data_tr_y, tr_pred)),
            r2_score(data_tr_y, tr_pred),
            mean_absolute_error(data_tr_y, tr_pred)]

        # validation set
        va_pred = model.predict(data_va_x)
        va_results = [split, dataset_label, subtask, 'va', num_fea,
                      num_of_compounds,
                      np.sqrt(mean_squared_error(data_va_y, va_pred)),
                      r2_score(data_va_y, va_pred),
                      mean_absolute_error(data_va_y, va_pred)]

        # test set
        te_pred = model.predict(data_te_x)
        te_results = [
            split, dataset_label, subtask, 'te', num_fea, num_of_compounds,
            np.sqrt(mean_squared_error(data_te_y, te_pred)),
            r2_score(data_te_y, te_pred),
            mean_absolute_error(data_te_y, te_pred)]
    return tr_results, va_results, te_results


def run_svm(
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
    num_pools = 28
    space_ = {
        'C': hp.uniform('C', 0.01, 100),
        'gamma': hp.uniform('gamma', 0, 0.2)
    }
    dataset = parameters.dataset
    pd_res = []

    task_args = [
        (task, dataset, dataset_label, task_type, OPT_ITERS, space_)
        for task in tasks
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
                'dataset', 'subtask', 'set',
                'num_of_retained_feature',
                'num_of_compounds', 'postives',
                'negtives', 'negtives/postives',
                'C', 'gamma',
                'tn', 'fp', 'fn', 'tp', 'se', 'sp',
                'acc', 'mcc', 'auc_prc', 'auc_roc']
        )
    else:
        best_hyper = pd.DataFrame(
            pd_res, columns=[
                'dataset', 'subtask', 'set',
                'num_of_retained_feature',
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
        cols = [subtask]
        cols.extend(dataset.columns[(len(tasks) + 1):])
        # cols.extend(dataset.columns[(617+1):])
        sub_dataset = dataset[cols]

        # detect the NA in the subtask (y cloumn)
        rm_index = sub_dataset[subtask][sub_dataset[subtask].isnull()].index
        sub_dataset.drop(index=rm_index, inplace=True)

        # remove the features with na
        if dataset_label != 'hiv':
            sub_dataset = sub_dataset.dropna(axis=1)
        else:
            sub_dataset = sub_dataset.dropna(axis=0)

        # *******************
        # demension reduction
        # *******************
        # Removing features with low variance
        # threshold = 0.05
        data_fea_var = sub_dataset.iloc[:, 1:].var()
        del_fea1 = list(data_fea_var[data_fea_var <= 0.05].index)
        sub_dataset.drop(columns=del_fea1, inplace=True)

        # pair correlations
        # threshold = 0.95
        data_fea_corr = sub_dataset.iloc[:, 1:].corr()
        del_fea2_col = []
        del_fea2_ind = []
        length = data_fea_corr.shape[1]
        for i in range(length):
            for j in range(i + 1, length):
                if abs(data_fea_corr.iloc[i, j]) >= 0.95:
                    del_fea2_col.append(data_fea_corr.columns[i])
                    del_fea2_ind.append(data_fea_corr.index[j])
        sub_dataset.drop(columns=del_fea2_ind, inplace=True)

        # standardize the features
        cols_ = list(sub_dataset.columns)[1:]
        sub_dataset[cols_] = sub_dataset[cols_].apply(standardize, axis=0)

        # for split in range(1, splits+1):
        args = [
            (split, task_type, sub_dataset, subtask, best_hyper,
             dataset_label)
            for split in range(1, repetitions+1)]
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
                'num_of_retained_feature',
                'num_of_compounds', 'postives',
                'negtives', 'negtives/postives',
                'tn', 'fp', 'fn', 'tp', 'se', 'sp',
                'acc', 'mcc', 'auc_prc', 'auc_roc'])
    else:
        stat_res = pd.DataFrame(
            pd_res, columns=[
                'split', 'dataset', 'subtask', 'set',
                'num_of_retained_feature',
                'num_of_compounds', 'rmse', 'r2', 'mae'])
# single tasks
    if len(tasks) == 1:
        args = {
            'data_label': dataset_label,
            'metric': 'auc_roc' if task_type == 'cla' else 'rmse',
            'model': 'SVM'}
        print("{}_{}: the mean {} for the training set is {:.3f} with "
              "std {:.3f}"
              .format(
                  args['data_label'], args['model'], args['metric'],
                  np.mean(stat_res[stat_res['set'] == 'tr'][args['metric']]),
                  np.std(stat_res[stat_res['set'] == 'tr'][args['metric']])))
        print(
            "{}_{}: the mean {} for the validation set is {:.3f} with "
            "std {:.3f}"
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
            'model': 'SVM'
        }
        tr_acc = np.zeros(repetitions)
        va_acc = np.zeros(repetitions)
        te_acc = np.zeros(repetitions)
        for subtask in tasks:
            tr = stat_res[stat_res['set'] == 'tr']
            tr_acc = (
                tr_acc + tr[tr['subtask'] == subtask][args['metric']]
                .values
            )

            va = stat_res[stat_res['set'] == 'va']
            va_acc = (
                va_acc + va[va['subtask'] == subtask][args['metric']]
                .values
            )

            te = stat_res[stat_res['set'] == 'te']
            te_acc = (
                te_acc + te[te['subtask'] == subtask][args['metric']]
                .values
            )
        tr_acc = tr_acc / len(tasks)
        va_acc = va_acc / len(tasks)
        te_acc = te_acc / len(tasks)
        print("{}_{}: the mean {} for the training set is {:.3f} with std "
              "{:.3f}".format(
                  args['data_label'], args['model'], args['metric'],
                  np.mean(tr_acc), np.std(tr_acc))
              )
        print("{}_{}: the mean {} for the validation set is {:.3f} with std "
              "{:.3f}".format(
                  args['data_label'], args['model'],
                  args['metric'], np.mean(va_acc), np.std(va_acc))
              )
        print("{}_{}: the mean {} for the test set is {:.3f} with std {:.3f}"
              .format(args['data_label'], args['model'], args['metric'],
                      np.mean(te_acc), np.std(te_acc))
              )
    end = time.time()  # get the end time
    elapsed = end-start
    print("the elapsed time is:", (end - start), "S")

    return stat_res, elapsed
