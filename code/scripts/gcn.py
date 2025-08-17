import time
import gc
from dgl.data.utils import Subset
import pandas as pd
from dgllife.data import MoleculeCSVDataset, csv_dataset
from dgllife.utils import smiles_to_bigraph
from gnn_utils import (
    AttentiveFPBondFeaturizer, AttentiveFPAtomFeaturizer,
    collate_molgraphs, EarlyStopping, set_random_seed, Meter)
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader
import torch
from dgllife.model.model_zoo import GCNPredictor
import numpy as np
from sklearn.model_selection import train_test_split
from dgl import backend as F
from hyperopt import fmin, tpe, hp, Trials
from task_dict import tasks_dic
from util import RunParameters


def run_a_train_epoch(model, data_loader, loss_func, optimizer, args):
    model.train()
    train_metric = Meter()  # for each epoch
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        atom_feats = bg.ndata.pop('h')
        bond_feats = bg.edata.pop('e')

        # transfer the data to device(cpu or cuda)
        labels, masks, atom_feats, bond_feats = (
            labels.to(args['device']),
            masks.to(args['device']),
            atom_feats.to(args['device']),
            bond_feats.to(args['device'])
        )

        if args['model'] in ['gcn', 'gat']:
            outputs = model(bg.to(args['device']), atom_feats)
        else:
            outputs = model(bg.to(args['device']), atom_feats, bond_feats)
        loss = (loss_func(outputs, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs.cpu()
        labels.cpu()
        masks.cpu()
        atom_feats.cpu()
        bond_feats.cpu()
        loss.cpu()
        torch.cuda.empty_cache()

        train_metric.update(outputs, labels, masks)

    if args['metric'] == 'rmse':
        rmse_score = np.mean(train_metric.compute_metric(
            args['metric']))  # in case of multi-tasks
        # in case of multi-tasks
        mae_score = np.mean(train_metric.compute_metric('mae'))
        r2_score = np.mean(train_metric.compute_metric('r2'))
        # in case of multi-tasks
        return {'rmse': rmse_score, 'mae': mae_score, 'r2': r2_score}
    else:
        roc_score = np.mean(train_metric.compute_metric(
            args['metric']))  # in case of multi-tasks
        prc_score = np.mean(train_metric.compute_metric('prc_auc'))
        # mcc = np.mean(train_metric.compute_metric('mcc'))
        # accuracy = np.mean(train_metric.compute_metric('accuracy'))
        # in case of multi-tasks
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
            smiles, bg, labels, masks = batch_data
            atom_feats = bg.ndata.pop('h')
            bond_feats = bg.edata.pop('e')

            # transfer the data to device(cpu or cuda)
            labels = labels.to(args['device'])
            masks = masks.to(args['device'])
            atom_feats = atom_feats.to(args['device'])
            bond_feats = bond_feats.to(args['device'])

            if args['model'] in ['gcn', 'gat']:
                outputs = model(bg.to(args['device']), atom_feats)
            else:
                outputs = model(bg.to(args['device']), atom_feats, bond_feats)

            outputs.cpu()
            labels.cpu()
            masks.cpu()
            atom_feats.cpu()
            bond_feats.cpu()
            # loss.cpu()
            torch.cuda.empty_cache()
            eval_metric.update(outputs, labels, masks)
    if args['metric'] == 'rmse':
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


def get_pos_weight(my_dataset):
    num_pos = F.sum(my_dataset.labels, dim=0)
    num_indices = F.tensor(len(my_dataset.labels))
    return (num_indices - num_pos) / num_pos


def all_one_zeros(series):
    if (len(series.dropna().unique()) == 2):
        flag = False
    else:
        flag = True
    return flag


def run_gcn(parameters: RunParameters) -> (pd.DataFrame, float):
    start = time.time()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    set_random_seed(seed=43)
    task_type = parameters.task_type  # 'cla' or 'reg'
#   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
# if device == 'cuda':
#    torch.cuda.set_device(eval(gpu_id))  # gpu device id
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        raise Exception("No GPU")
    dataset = parameters.label
    my_df = parameters.dataset
    AtomFeaturizer = AttentiveFPAtomFeaturizer
    BondFeaturizer = AttentiveFPBondFeaturizer
    epochs = 300
    batch_size = 128*5
    patience = 50
    opt_iters = parameters.hyperOptIters
    repetitions = parameters.evalIters
    num_workers = 0
    args = {
        'device': device,
        'task': dataset,
        'metric': 'roc_auc' if task_type == 'cla' else 'rmse',
        'model': 'gcn'
    }
    tasks = tasks_dic[args['task']]
    hyper_space = dict(
        l2=hp.choice('l2', [0, 10 ** -8, 10 ** -6, 10 ** -4]),
        lr=hp.choice('lr', [10 ** -2.5, 10 ** -3.5, 10 ** -1.5]),
        gcn_hidden_feats=hp.choice(
            'gcn_hidden_feats',
            [[128, 128], [256, 256], [128, 64], [256, 128]]
        ),
        classifier_hidden_feats=hp.choice(
            'classifier_hidden_feats', [128, 64, 256])
    )

# get the df and generate graph,
# attention for graph generation for some bad smiles
# my_df.iloc[:, 0:-1], except with 'group'

    my_dataset: MoleculeCSVDataset = csv_dataset.MoleculeCSVDataset(
        my_df.iloc[:, 0:-1],
        smiles_to_bigraph,
        AtomFeaturizer,
        BondFeaturizer,
        'cano_smiles',
        f'../../../model_cache/{dataset}.bin'
    )

    if task_type == 'cla':
        pos_weight = get_pos_weight(my_dataset)
    else:
        pos_weight = None

# get the training, validation, and test sets
    tr_indx, val_indx, te_indx = (
        my_df[my_df.group == 'train'].index,
        my_df[my_df.group == 'valid'].index,
        my_df[my_df.group == 'test'].index
    )
    train_loader = DataLoader(
        Subset(my_dataset, tr_indx), batch_size=batch_size, shuffle=True,
        collate_fn=collate_molgraphs, num_workers=num_workers)
    val_loader = DataLoader(
        Subset(my_dataset, val_indx), batch_size=batch_size, shuffle=True,
        collate_fn=collate_molgraphs, num_workers=num_workers)
    test_loader = DataLoader(
        Subset(my_dataset, te_indx), batch_size=batch_size, shuffle=True,
        collate_fn=collate_molgraphs, num_workers=num_workers)

    def hyper_opt(hyper_paras):
        # get the model instance
        my_model = GCNPredictor(
            in_feats=AtomFeaturizer.feat_size('h'),
            hidden_feats=hyper_paras['gcn_hidden_feats'],
            n_tasks=len(tasks),
            predictor_hidden_feats=hyper_paras['classifier_hidden_feats']
        )
        model_file_name = './saved_model/%s_%s_%s_%.6f_%s_%s.pth' % (
            args['model'], args['task'],
            hyper_paras['l2'], hyper_paras['lr'],
            hyper_paras['gcn_hidden_feats'],
            hyper_paras['classifier_hidden_feats'])

        optimizer = torch.optim.Adam(
            my_model.parameters(),
            lr=hyper_paras['lr'],
            weight_decay=hyper_paras['l2']
        )

        if task_type == 'reg':
            loss_func = MSELoss(reduction='none')
            stopper = EarlyStopping(
                mode='lower', patience=patience, filename=model_file_name)
        else:
            loss_func = BCEWithLogitsLoss(
                reduction='none', pos_weight=pos_weight.to(args['device']))
            stopper = EarlyStopping(
                mode='higher', patience=patience, filename=model_file_name)
        my_model.to(args['device'])

        for j in range(epochs):
            # training
            run_a_train_epoch(
                my_model, train_loader, loss_func, optimizer, args
            )

            # early stopping
            val_scores = run_an_eval_epoch(my_model, val_loader, args)
            early_stop = stopper.step(val_scores[args['metric']], my_model)

            if early_stop:
                break
        stopper.load_checkpoint(my_model)
        tr_scores = run_an_eval_epoch(my_model, train_loader, args)
        val_scores = run_an_eval_epoch(my_model, val_loader, args)
        te_scores = run_an_eval_epoch(my_model, test_loader, args)
        print({'train': tr_scores, 'valid': val_scores, 'test': te_scores})
        feedback = val_scores[args['metric']] if task_type == 'reg' else (
            1 - val_scores[args['metric']])
        my_model.cpu()
        torch.cuda.empty_cache()
        gc.collect()
        return feedback


# start hyper-parameters optimization
    print('******hyper-parameter optimization is starting now******')
    trials = Trials()
    opt_res = fmin(hyper_opt, hyper_space, algo=tpe.suggest,
                   max_evals=opt_iters, trials=trials)

# hyper-parameters optimization is over
    print('******hyper-parameter optimization is over******')
    print('the best hyper-parameters settings for ' +
          args['task'] + ' gcn are:  ', opt_res)

# construct the model based on the optimal hyper-parameters
    l2_ls = [0, 10 ** -8, 10 ** -6, 10 ** -4]
    lr_ls = [10 ** -2.5, 10 ** -3.5, 10 ** -1.5]
    hidden_feats_ls = [(128, 128), (256, 256), (128, 64), (256, 128)]
    classifier_hidden_feats_ls = [128, 64, 256]

# 50 repetitions based on the best model
    tr_res = []
    val_res = []
    te_res = []
# regenerate the graphs
    if args['task'] == 'muv' or args['task'] == 'toxcast':
        file_name = './dataset/' + args['task'] + '_new.csv'
        my_df = pd.read_csv(file_name)
        my_dataset = csv_dataset.MoleculeCSVDataset(
            my_df, smiles_to_bigraph, AtomFeaturizer, BondFeaturizer,
            'cano_smiles', file_name.replace('.csv', '.bin'))
    else:
        my_df.drop(columns=['group'], inplace=True)

    for split in range(1, repetitions + 1):
        # splitting the data set for classification
        if args['metric'] == 'roc_auc':
            seed = split
            while True:
                training_data, data_te = train_test_split(
                    my_df, test_size=0.1, random_state=seed)
                # the training set was further splitted into the training set
                # and validation set
                data_tr, data_va = train_test_split(
                    training_data, test_size=0.1, random_state=seed)
                if np.any(data_tr[tasks].apply(all_one_zeros)) or \
                        np.any(data_va[tasks].apply(all_one_zeros)) or \
                        np.any(data_te[tasks].apply(all_one_zeros)):
                    print("\ninvalid random seed {} due to one class "
                          "presented in the splitted {} sets..."
                          .format(seed, args['task']))
                    print("Changing to another random seed...\n")
                    seed = np.random.randint(50, 999999)
                else:
                    print("random seed used in repetition {} is {}"
                          .format(split, seed))
                    break
        else:
            training_data, data_te = train_test_split(
                my_df, test_size=0.1, random_state=split)
            # the training set was further splitted into the training set
            # and validation set
            data_tr, data_va = train_test_split(
                training_data, test_size=0.1, random_state=split)
        tr_indx, val_indx, te_indx = (
            data_tr.index, data_va.index, data_te.index
        )
        train_loader = DataLoader(
            Subset(my_dataset, tr_indx), batch_size=batch_size,
            shuffle=True, collate_fn=collate_molgraphs,
            num_workers=num_workers)
        val_loader = DataLoader(
            Subset(my_dataset, val_indx), batch_size=batch_size,
            shuffle=True, collate_fn=collate_molgraphs,
            num_workers=num_workers)
        test_loader = DataLoader(
            Subset(my_dataset, te_indx), batch_size=batch_size,
            shuffle=True, collate_fn=collate_molgraphs,
            num_workers=num_workers)
        best_model_file = "./saved_model/%s_%s_bst_%s.pth" % (
            "gcn", args['task'], split
        )

        best_model = GCNPredictor(
            in_feats=AtomFeaturizer.feat_size('h'),
            hidden_feats=hidden_feats_ls[opt_res['gcn_hidden_feats']],
            n_tasks=len(tasks),
            predictor_hidden_feats=classifier_hidden_feats_ls[
                opt_res['classifier_hidden_feats']]
        )
        best_optimizer = torch.optim.Adam(
            best_model.parameters(), lr=lr_ls[opt_res['lr']],
            weight_decay=l2_ls[opt_res['l2']])
        if task_type == 'reg':
            loss_func = MSELoss(reduction='none')
            stopper = EarlyStopping(
                mode='lower', patience=patience, filename=best_model_file)
        else:
            loss_func = BCEWithLogitsLoss(
                reduction='none', pos_weight=pos_weight.to(args['device']))
            stopper = EarlyStopping(
                mode='higher', patience=patience, filename=best_model_file)
        best_model.to(device)

        for j in range(epochs):
            # training
            st = time.time()
            run_a_train_epoch(best_model, train_loader,
                              loss_func, best_optimizer, args)
            end = time.time()
            # early stopping
            train_scores = run_an_eval_epoch(
                best_model, train_loader, args)
            val_scores = run_an_eval_epoch(
                best_model, val_loader, args)
            early_stop = stopper.step(
                val_scores[args['metric']], best_model)
            if early_stop:
                break
            print("task:{} repetition {:d}/{:d} epoch {:d}/{:d}, training {} "
                  "{:.3f}, validation {} {:.3f}, time:{:.3f}S"
                  .format(
                      args['task'], split, repetitions, j + 1, epochs,
                      args['metric'], train_scores[args['metric']],
                      args['metric'], val_scores[args['metric']], end - st))
        stopper.load_checkpoint(best_model)
        tr_scores = run_an_eval_epoch(best_model, train_loader, args)
        val_scores = run_an_eval_epoch(best_model, val_loader, args)
        te_scores = run_an_eval_epoch(best_model, test_loader, args)
        tr_res.append(tr_scores)
        val_res.append(val_scores)
        te_res.append(te_scores)

    if task_type == 'reg':
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
    sta_pd['model'] = 'gcn'
    sta_pd['dataset'] = args['task']

    print("training mean:", np.mean(tr, axis=0),
          "training std:", np.std(tr, axis=0))
    print("validation mean:", np.mean(val, axis=0),
          "validation std:", np.std(val, axis=0))
    print("testing mean:", np.mean(te, axis=0),
          "test std:", np.std(te, axis=0))

    end = time.time()
    elapsed = end-start
    print("the total elapsed time is", end - start, "S")
    return sta_pd, elapsed
