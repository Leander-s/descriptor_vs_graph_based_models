from task_dict import tasks_dic
import warnings
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import pandas as pd
from dnn_torch_utils import (
    Meter, MyDataset, EarlyStopping, MyDNN, collate_fn, set_random_seed
)
from hyperopt import fmin, tpe, hp, Trials
import copy
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, MSELoss
import gc
import time
from util import RunParameters


def run_a_train_epoch(model, data_loader, loss_func, optimizer, args):
    model.train()

    train_metric = Meter()  # for each epoch
    for batch_id, batch_data in enumerate(data_loader):
        Xs, Ys, masks = batch_data

        # transfer the data to device(cpu or cuda)
        Xs, Ys, masks = Xs.to(args['device']), Ys.to(
            args['device']), masks.to(args['device'])

        outputs = model(Xs)
        loss = (loss_func(outputs, Ys) * (masks != 0).float()).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs.cpu()
        Ys.cpu()
        masks.cpu()
        loss.cpu()
#        torch.cuda.empty_cache()

        train_metric.update(outputs, Ys, masks)
    if args['reg']:
        rmse_score = np.mean(train_metric.compute_metric(
            args['metric']))  # in case of multi-tasks
        # in case of multi-tasks
        mae_score = np.mean(train_metric.compute_metric('mae'))
        r2_score = np.mean(train_metric.compute_metric('r2')
                           )  # in case of multi-tasks
        return {'rmse': rmse_score, 'mae': mae_score, 'r2': r2_score}
    else:
        roc_score = np.mean(train_metric.compute_metric(
            args['metric']))  # in case of multi-tasks
        prc_score = np.mean(train_metric.compute_metric(
            'prc_auc'))  # in case of multi-tasks
        # mcc = np.mean(train_metric.compute_metric('mcc'))
        # accuracy = np.mean(train_metric.compute_metric('accuracy'))
        return {
            'roc_auc': roc_score,
            'prc_auc': prc_score,
            # 'mcc': mcc,
            # 'accuracy': accuracy
        }


def run_an_eval_epoch(model, data_loader, args):
    model.eval()

    eval_metric = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            Xs, Ys, masks = batch_data
            # transfer the data to device(cpu or cuda)
            Xs, Ys, masks = Xs.to(args['device']), Ys.to(
                args['device']), masks.to(args['device'])

            outputs = model(Xs)

            outputs.cpu()
            Ys.cpu()
            masks.cpu()
#            torch.cuda.empty_cache()
            eval_metric.update(outputs, Ys, masks)
    if args['reg']:
        rmse_score = np.mean(eval_metric.compute_metric(
            args['metric']))  # in case of multi-tasks
        # in case of multi-tasks
        mae_score = np.mean(eval_metric.compute_metric('mae'))
        r2_score = np.mean(eval_metric.compute_metric('r2')
                           )  # in case of multi-tasks
        return {'rmse': rmse_score, 'mae': mae_score, 'r2': r2_score}
    else:
        roc_score = np.mean(eval_metric.compute_metric(
            args['metric']))  # in case of multi-tasks
        prc_score = np.mean(eval_metric.compute_metric(
            'prc_auc'))  # in case of multi-tasks
        # mcc = np.mean(eval_metric.compute_metric('mcc'))
        # accuracy = np.mean(eval_metric.compute_metric('accuracy'))
        return {
            'roc_auc': roc_score,
            'prc_auc': prc_score,
            # 'mcc': mcc,
            # 'accuracy': accuracy
        }


def get_pos_weight(Ys):
    Ys = torch.tensor(np.nan_to_num(Ys), dtype=torch.float32)
    num_pos = torch.sum(Ys, dim=0)
    num_indices = torch.tensor(len(Ys))
    return (num_indices - num_pos) / num_pos


def standardize(col):
    return (col - np.mean(col)) / np.std(col)


def all_one_zeros(series):
    if (len(series.dropna().unique()) == 2):
        flag = False
    else:
        flag = True
    return flag


def run_dnn(parameters: RunParameters) -> (pd.DataFrame, float):
    start = time.time()
    warnings.filterwarnings('ignore')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    set_random_seed(seed=43)

    hyper_paras_space = {
        'l2': hp.uniform('l2', 0, 0.01),
        'dropout': hp.uniform('dropout', 0, 0.5),
        'hidden_unit1': hp.choice('hidden_unit1', [64, 128, 256, 512]),
        'hidden_unit2': hp.choice('hidden_unit2', [64, 128, 256, 512]),
        'hidden_unit3': hp.choice('hidden_unit3', [64, 128, 256, 512])}

    task_type = parameters.task_type  # 'cla' or 'reg'
    reg = True if task_type == 'reg' else False
    epochs = 300  # training epoch
    data_label = parameters.label
    batch_size = 128
    patience = 50
    opt_iters = parameters.hyperOptIters
    repetitions = parameters.evalIters
    if torch.cuda.is_available():
        id = torch.cuda.current_device()
        device = torch.device(f"cuda:{id}")
    else:
        raise Exception("No GPU")

    args = {'device': device, 'metric': 'rmse' if reg else 'roc_auc',
            'epochs': epochs, 'patience': patience, 'task': data_label,
            'reg': reg}

# preprocess data
    dataset_all = parameters.dataset
    tasks = tasks_dic[data_label]
    cols = copy.deepcopy(tasks)
    cols.extend(dataset_all.columns[len(tasks) + 1:])
    dataset = dataset_all[cols]
    x_cols = dataset_all.columns[len(tasks) + 1:].drop(['group'])

# remove the features with na
    if data_label != 'hiv':
        rm_cols1 = (
            dataset[x_cols].isnull()
            .any()[dataset[x_cols].isnull().any() is True]
            .index
        )
        dataset.drop(columns=rm_cols1, inplace=True)
    else:
        rm_indx1 = (
            dataset[x_cols].isnull().T
            .any()[dataset[x_cols].isnull().T.any() is True].index
        )
        dataset.drop(index=rm_indx1, inplace=True)
    x_cols = dataset.columns.drop(tasks).drop(['group'])

# Removing features with low variance
# threshold = 0.05
    data_fea_var = dataset[x_cols].var()
    del_fea1 = list(data_fea_var[data_fea_var <= 0.05].index)
    dataset.drop(columns=del_fea1, inplace=True)
    x_cols = dataset.columns.drop(tasks).drop(['group'])

# pair correlations
# threshold = 0.95
    data_fea_corr = dataset[x_cols].corr()
    del_fea2_col = []
    del_fea2_ind = []
    length = data_fea_corr.shape[1]
    for i in range(length):
        for j in range(i + 1, length):
            if abs(data_fea_corr.iloc[i, j]) >= 0.95:
                del_fea2_col.append(data_fea_corr.columns[i])
                del_fea2_ind.append(data_fea_corr.index[j])
    dataset.drop(columns=del_fea2_ind, inplace=True)
# standardize the features
    cols_ = dataset.columns[len(tasks) + 1:]
    print('the retained features for %s is %d' % (args['task'], len(cols_)))
    dataset[cols_] = dataset[cols_].apply(standardize, axis=0)

    data_tr = dataset[dataset.group == 'train']
    data_va = dataset[dataset.group == 'valid']
    data_te = dataset[dataset.group == 'test']
# training set
    data_tr_y = data_tr[tasks].values.reshape(-1, len(tasks))
    data_tr_x = data_tr.iloc[:, len(tasks) + 1:].values

# test set
    data_te_y = data_te[tasks].values.reshape(-1, len(tasks))
    data_te_x = data_te.iloc[:, len(tasks) + 1:].values

# validation set
    data_va_y = data_va[tasks].values.reshape(-1, len(tasks))
    data_va_x = data_va.iloc[:, len(tasks) + 1:].values

# dataloader
    train_dataset = MyDataset(data_tr_x, data_tr_y)
    validation_dataset = MyDataset(data_va_x, data_va_y)
    test_dataset = MyDataset(data_te_x, data_te_y)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn)
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=True, collate_fn=collate_fn)
    inputs = data_tr_x.shape[1]
    if not reg:
        pos_weights = get_pos_weight(dataset[tasks].values)

    def hyper_opt(hyper_paras):
        hidden_units = [
            hyper_paras['hidden_unit1'],
            hyper_paras['hidden_unit2'],
            hyper_paras['hidden_unit3']
        ]
        my_model = MyDNN(
            inputs=inputs, hideen_units=hidden_units,
            dp_ratio=hyper_paras['dropout'], outputs=len(tasks), reg=reg)
        optimizer = torch.optim.Adadelta(
            my_model.parameters(), weight_decay=hyper_paras['l2'])
        file_name = './model_save/%s_%.4f_%d_%d_%d_%.4f_early_stop.pth' % (
            args['task'], hyper_paras['dropout'],
            hyper_paras['hidden_unit1'],
            hyper_paras['hidden_unit2'],
            hyper_paras['hidden_unit3'],
            hyper_paras['l2'])
        if reg:
            loss_func = MSELoss(reduction='none')
            stopper = EarlyStopping(
                mode='lower', patience=patience, filename=file_name)
        else:
            loss_func = BCEWithLogitsLoss(
                reduction='none', pos_weight=pos_weights.to(args['device']))
            stopper = EarlyStopping(
                mode='higher', patience=patience, filename=file_name)
        my_model.to(device)
        for i in range(epochs):
            # training
            run_a_train_epoch(
                my_model, train_loader, loss_func, optimizer, args
            )

            # early stopping
            val_scores = run_an_eval_epoch(my_model, validation_loader, args)

            early_stop = stopper.step(val_scores[args['metric']], my_model)

            if early_stop:
                break
        stopper.load_checkpoint(my_model)
        val_scores = run_an_eval_epoch(my_model, validation_loader, args)
        feedback = val_scores[args['metric']] if reg else (
            1 - val_scores[args['metric']])

        my_model.cpu()
#    torch.cuda.empty_cache()
        gc.collect()
        return feedback


# start hyper-parameters optimization
    trials = Trials()  # 通过Trials捕获信息
    print('******hyper-parameter optimization is starting now******')
    opt_res = fmin(hyper_opt, hyper_paras_space, algo=tpe.suggest,
                   max_evals=opt_iters, trials=trials)

# hyper-parameters optimization is over
    print('******hyper-parameter optimization is over******')
    print('the best hyper-parameters settings for ' +
          args['task'] + ' are:  ', opt_res)

# construct the model based on the optimal hyper-parameters
    hidden_unit1_ls = [64, 128, 256, 512]
    hidden_unit2_ls = [64, 128, 256, 512]
    hidden_unit3_ls = [64, 128, 256, 512]
    opt_hidden_units = [
        hidden_unit1_ls[opt_res['hidden_unit1']],
        hidden_unit2_ls[opt_res['hidden_unit2']],
        hidden_unit3_ls[opt_res['hidden_unit3']]
    ]
    best_model = MyDNN(
        inputs=inputs, hideen_units=opt_hidden_units, outputs=len(tasks),
        dp_ratio=opt_res['dropout'], reg=reg)
    best_file_name = './model_save/%s_%.4f_%d_%d_%d_%.4f_early_stop.pth' % (
        args['task'], opt_res['dropout'],
        hidden_unit1_ls[opt_res['hidden_unit1']],
        hidden_unit1_ls[opt_res['hidden_unit2']],
        hidden_unit1_ls[opt_res['hidden_unit3']],
        opt_res['l2'])

    best_model.load_state_dict(torch.load(
        best_file_name, map_location=device)['model_state_dict'])
    best_model.to(device)
    tr_scores = run_an_eval_epoch(best_model, train_loader, args)
    val_scores = run_an_eval_epoch(best_model, validation_loader, args)
    te_scores = run_an_eval_epoch(best_model, test_loader, args)

    print('training set:', tr_scores)
    print('validation set:', val_scores)
    print('test set:', te_scores)

# 50 repetitions based on the best model
    tr_res = []
    val_res = []
    te_res = []
    if data_label != 'muv' and data_label != 'toxcast':
        dataset.drop(columns=['group'], inplace=True)
    else:
        file = data_label + '_norepeat_moe_pubsubfp.csv'
        # repreprocess data
        dataset = pd.read_csv(file)
        dataset.drop(columns=['cano_smiles'], inplace=True)

        # remove the features with na
        x_cols = dataset.columns.drop(tasks)
        rm_cols1 = dataset[x_cols].isnull().any(
        )[dataset[x_cols].isnull().any() is True].index
        dataset.drop(columns=rm_cols1, inplace=True)

        # Removing features with low variance
        # threshold = 0.05
        x_cols = dataset.columns.drop(tasks)
        data_fea_var = dataset[x_cols].var()
        del_fea1 = list(data_fea_var[data_fea_var <= 0.05].index)
        dataset.drop(columns=del_fea1, inplace=True)

        # pair correlations
        # threshold = 0.95
        x_cols = dataset.columns.drop(tasks)
        data_fea_corr = dataset[x_cols].corr()
        del_fea2_col = []
        del_fea2_ind = []
        length = data_fea_corr.shape[1]
        for i in range(length):
            for j in range(i + 1, length):
                if abs(data_fea_corr.iloc[i, j]) >= 0.95:
                    del_fea2_col.append(data_fea_corr.columns[i])
                    del_fea2_ind.append(data_fea_corr.index[j])
        dataset.drop(columns=del_fea2_ind, inplace=True)

        # standardize the features
        x_cols = dataset.columns.drop(tasks)
        print('the retained features for noreaptead %s is %d' %
              (args['task'], len(x_cols)))
        dataset[x_cols] = dataset[x_cols].apply(standardize, axis=0)

    for split in range(1, repetitions + 1):
        # splitting the data set for classification
        if not reg:
            seed = split
            while True:
                training_data, data_te = train_test_split(
                    dataset, test_size=0.1, random_state=seed)
                # the training set was further splited into the training set
                # and validation set
                data_tr, data_va = train_test_split(
                    training_data, test_size=0.1, random_state=seed)
                if np.any(data_tr[tasks].apply(all_one_zeros)) or \
                        np.any(data_va[tasks].apply(all_one_zeros)) or \
                        np.any(data_te[tasks].apply(all_one_zeros)):
                    print("\ninvalid random seed {} due to one class "
                          "presented in the splitted {} sets..."
                          .format(seed, data_label))
                    print("Changing to another random seed...\n")
                    seed = np.random.randint(50, 999999)
                else:
                    print("random seed used in repetition {} is {}"
                          .format(split, seed))
                    break
        else:
            training_data, data_te = train_test_split(
                dataset, test_size=0.1, random_state=split)
            # the training set was further splited into the training set and
            # validation set
            data_tr, data_va = train_test_split(
                training_data, test_size=0.1, random_state=split)
        # prepare data for training
        # training set
        data_tr_y = data_tr[tasks].values.reshape(-1, len(tasks))
        data_tr_x = data_tr.iloc[:, len(tasks):].values

        # test set
        data_te_y = data_te[tasks].values.reshape(-1, len(tasks))
        data_te_x = data_te.iloc[:, len(tasks):].values

        # validation set
        data_va_y = data_va[tasks].values.reshape(-1, len(tasks))
        data_va_x = data_va.iloc[:, len(tasks):].values

        # dataloader
        train_dataset = MyDataset(data_tr_x, data_tr_y)
        validation_dataset = MyDataset(data_va_x, data_va_y)
        test_dataset = MyDataset(data_te_x, data_te_y)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn)
        validation_loader = DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn)

        best_model = MyDNN(
            inputs=inputs, hideen_units=opt_hidden_units,
            outputs=len(tasks), dp_ratio=opt_res['dropout'], reg=reg)

        best_optimizer = torch.optim.Adadelta(
            best_model.parameters(), weight_decay=opt_res['l2'])
        file_name = './model_save/%s_%.4f_%d_%d_%d_%.4f_early_stop_%d.pth' % (
            args['task'], opt_res['dropout'],
            hidden_unit1_ls[opt_res['hidden_unit1']],
            hidden_unit1_ls[opt_res['hidden_unit2']],
            hidden_unit1_ls[opt_res['hidden_unit3']],
            opt_res['l2'], split)
        if reg:
            loss_func = MSELoss(reduction='none')
            stopper = EarlyStopping(
                mode='lower', patience=patience, filename=file_name)
        else:
            loss_func = BCEWithLogitsLoss(
                reduction='none', pos_weight=pos_weights.to(args['device']))
            stopper = EarlyStopping(
                mode='higher', patience=patience, filename=file_name)
        best_model.to(device)

        for j in range(epochs):
            # training
            st = time.time()
            run_a_train_epoch(best_model, train_loader,
                              loss_func, best_optimizer, args)
            end = time.time()
            # early stopping
            train_scores = run_an_eval_epoch(best_model, train_loader, args)
            val_scores = run_an_eval_epoch(best_model, validation_loader, args)
            early_stop = stopper.step(val_scores[args['metric']], best_model)
            if early_stop:
                break
            print(
                "task:{} repetition {:d}/{:d} epoch {:d}/{:d}, training {} "
                "{:.3f}, validation {} {:.3f}, time:{:.3f}S"
                .format(
                    args['task'], split, repetitions, j + 1, epochs,
                    args['metric'], train_scores[args['metric']],
                    args['metric'], val_scores[args['metric']], end - st))
        stopper.load_checkpoint(best_model)
        tr_scores = run_an_eval_epoch(best_model, train_loader, args)
        val_scores = run_an_eval_epoch(best_model, validation_loader, args)
        te_scores = run_an_eval_epoch(best_model, test_loader, args)
        tr_res.append(tr_scores)
        val_res.append(val_scores)
        te_res.append(te_scores)
    if reg:
        cols = ['rmse', 'mae', 'r2']
    else:
        # cols = ['roc_auc', 'prc_auc', 'mcc', 'accuracy']
        cols = ['roc_auc', 'prc_auc']
    tr = [list(item.values()) for item in tr_res]
    val = [list(item.values()) for item in val_res]
    te = [list(item.values()) for item in te_res]
    tr_pd = pd.DataFrame(tr, columns=cols)
    tr_pd['split'] = range(1, repetitions + 1)
    tr_pd['set'] = 'train'
    val_pd = pd.DataFrame(val, columns=cols)
    val_pd['split'] = range(1, repetitions + 1)
    val_pd['set'] = 'validation'
    te_pd = pd.DataFrame(te, columns=cols)
    te_pd['split'] = range(1, repetitions + 1)
    te_pd['set'] = 'test'
    sta_pd = pd.concat([tr_pd, val_pd, te_pd], ignore_index=True)

    print("training mean:", np.mean(tr, axis=0),
          "training std:", np.std(tr, axis=0))
    print("validation mean:", np.mean(val, axis=0),
          "validation std:", np.std(val, axis=0))
    print("testing mean:", np.mean(te, axis=0),
          "test std:", np.std(te, axis=0))
    end = time.time()
    elapsed = end-start
    print("total elapsed time is", end-start, "S")
    return sta_pd, elapsed
