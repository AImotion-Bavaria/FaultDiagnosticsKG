from datetime import datetime
import os
from pathlib import Path
import logging

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import yaml

import utils_data


def write_to_results(config, path_results=None):
    '''
    Write results from config file to csv.

    Args:
        config: Config containing results of model run
        path_results: Path to csv. If None is given a new csv is created.
    '''
    path_results = Path(path_results) / 'results_raw.csv'
    if not os.path.exists(path_results):
        create_new_results_df(config['path']['model_dir'])
    fault_mapping = config['data'].pop('fault_mapping')
    res_temp = pd.json_normalize(config)
    res_temp['fault_mapping'] = [fault_mapping]
    res = pd.read_csv(path_results)
    res = pd.concat([res, res_temp], axis=0).reset_index(drop=True)
    path = config['path']['model_dir']
    logging.info("Write to results to %s/results_raw.csv.", path)
    res.to_csv(path_results, index=False)


def create_new_results_df(path):
    '''
    Create new results dataframe.

    Args:
        path: Path to saved models with results saved in config files.
    '''
    res = pd.DataFrame()
    path = Path(path)
    path_res = path / 'results_raw.csv'
    paths = [path / 'Embedding',
             path / 'DirectClassification']
    for model_type in paths:
        if os.path.exists(model_type):
            for mod in os.listdir(model_type):
                cl_f = model_type / mod / 'config_loss.yaml'
                with open(cl_f, encoding='utf8') as stream:
                    config = yaml.safe_load(stream)
                fault_mapping = config['data'].pop('fault_mapping')
                cl_t = model_type / mod / 'DomainTransfer_FE_time' / \
                    'config_transfer.yaml'
                if os.path.exists(cl_t):
                    with open(cl_t, encoding='utf8') as stream:
                        config_transfer = yaml.safe_load(stream)
                    config['transfer'] = config_transfer
                res_temp = pd.json_normalize(config)
                res_temp['fault_mapping'] = [fault_mapping]
                res = pd.concat([res, res_temp], axis=0).reset_index(drop=True)

    res.to_csv(path_res, index=False)


def calc_accuracy(y_hat, y):
    '''
    Calc accuaracy.

    Args:
        y_hat: 1d torch tensor of predicted labels
        y: 1d torch tensor of true labels

    Returns:
        Accuracy
    '''
    return float(
        (torch.argmax(y_hat, axis=1) == y).sum() / len(y)
    )


def get_accuracy(clf, x_emb, y, device, return_y_pred=False):
    '''
    Get accuracy for different classifiers.

    Args:
        clf: Classifier
        x_emb: Embedding of x, input to clf
        y: True labels
        device: Device on which model will be evaluated
        return_y_pred: Return predicted labels

    Returns:
        acc: Accuracy
        y_pred: Predicted labels (if retur_y_pred=True)
    '''
    dataloader = utils_data.create_dataloader(
        x_emb, y, shuffle=False
    )
    acc = 0
    y_pred = []
    for x_batch, y_batch in dataloader:
        batch_size = len(y_batch)
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred_batch = clf(x_batch.to(device))
        y_pred.append(y_pred_batch.cpu())

        # Calculate overall accuracy
        acc += calc_accuracy(
            y_pred_batch, y_batch
        ) * batch_size

    ds_size = len(dataloader.dataset)
    acc /= ds_size

    if not return_y_pred:
        return acc
    return acc, torch.cat(y_pred)


def evaluate_model_oc(
        x_emb_source, x_emb_target, y_source, y_target, config, path
        ):
    '''
    Evaluate model on changing operating conditions.

    Args:
        x_emb_source: X embedding on source domain
        x_emb_target: X embedding on target domain
        y_source: labels of source domain
        y_target: labels of target domain
        config: Config dictionary
        path: Path where results are saved
    '''
    x_emb_source_2d = get_2d_embedding(x_emb_source, perp=50)
    x_emb_target_2d = get_2d_embedding(x_emb_target, perp=50)

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(21, 7))
    fig.suptitle(config['name'])

    ax0.set_title('Loss')
    plot_loss(config, ax=ax0)

    train_score = f"(Clf-Score: {config['clf_test_score']:.3f})"
    ax1.set_title(f'Source OC: T-SNE 2d embedding {train_score}')
    plot_2d_embedding(
        x_emb_source_2d,
        y_source,
        mapping=config['data']['fault_mapping']['data_target'],
        ax=ax1
    )

    test_score = f"(Clf-Score: {config['oc_acc']:.3f})"
    ax2.set_title(f'Target OC: T-SNE 2d embedding {test_score}')
    plot_2d_embedding(
        x_emb_target_2d,
        y_target,
        mapping=config['data']['fault_mapping']['data_target'],
        ax=ax2
    )

    fig.savefig(path / 'results.png')


def evaluate_model_bearing(x_emb_train, x_emb_test,
                           y_train, y_test, config, path):
    '''
    Evaluate model on different bearing.

    Args:
        x_emb_train: X embedding of training data
        x_emb_test: X embedding of test data
        y_train: Training labels
        y_test: Test labels
        config: Config dictionary
        path: Path where results are saved
    '''
    x_emb_train_2d = get_2d_embedding(x_emb_train, perp=50)
    x_emb_test_2d = get_2d_embedding(x_emb_test, perp=50)

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(21, 7))
    fig.suptitle(config['name'])

    ax0.set_title('Loss')
    plot_loss(config, ax=ax0)

    train_score = f"(Clf-Score: {config['clf_train_score']:.3f})"
    ax1.set_title(f'Train: T-SNE 2d embedding {train_score}')
    plot_2d_embedding(
        x_emb_train_2d,
        y_train,
        mapping=config['data']['fault_mapping']['data_target'],
        ax=ax1
    )

    test_score = f"(Clf-Score: {config['clf_test_score']:.3f})"
    ax2.set_title(f'Test: T-SNE 2d embedding {test_score}')
    plot_2d_embedding(
        x_emb_test_2d,
        y_test,
        mapping=config['data']['fault_mapping']['data_target'],
        ax=ax2
    )

    fig.savefig(path / 'results.png')


def save_config(config, path):
    '''
    Save config dictionary.

    Args:
        config: Config dictionary to be saved
        path: Path where config is saved
    '''
    with open(path / 'config_loss.yaml', 'w', encoding='utf8') as out:
        yaml.dump(config, out)
    logging.info("Results saved in %s.", path)


def get_save_path(config):
    '''
    Create new folder for model information and get path.

    Args:
        config: Configuration dictionary

    Returns:
        Complete path to folder
        Name of folder
    '''
    time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    if config["model"]["loss_fn"] == "CrossEntropyLoss":
        m_dir = Path(config['path']['model_dir']) / "DirectClassification"
    else:
        m_dir = Path(config['path']['model_dir']) / "Embedding"
    try:
        load = f'{config["data"]["oc_target"]}_'
    except KeyError:
        load = ''
    m_folder = (
        f'{config["data"]["source_domain"]}_'
        f'{load}'
        f'{config["seed"]}_'
        f'{config["model"]["loss_fn"]}_'
        f'{config["model"]["distance"]}_'
        f'{config["model"]["mining_func"]}_'
        f'{config["kg"]["kg_trainer"]}_'
        f'{time}'
    )
    Path(m_dir / m_folder).mkdir(exist_ok=True, parents=True)
    return (Path(m_dir / m_folder), m_folder)


def get_2d_embedding(x_emb, perp=50):
    '''
    Get 2d embedding with T-SNE.

    Args:
        x_emb: X embedding
        perp: Perplexity parameter for T-SNE

    Returns:
        x_emb_2d: 2d embedding of x_emb
    '''
    if x_emb.shape[1] > 50:
        x_emb = PCA(n_components=50).fit_transform(x_emb)
    x_emb_2d = TSNE(
        n_components=2,
        learning_rate='auto',
        init='random',
        perplexity=perp).fit_transform(x_emb)
    return x_emb_2d


def plot_2d_embedding(X, y, mapping, ax=None):
    '''
    Plot the 2d embedding of different bearing conditions.

    Args:
        X: 2d vector
        y: labels to X
        mapping: Mapping of numbers to labels
        ax: Matplotlib ax object if available
    '''
    labels = [k for i in set(y) for k in mapping if i == mapping[k]]
    color_mapping = get_color_mapping()
    if ax is None:
        _, ax = plt.subplots()
    for i, fault_id in enumerate(set(y)):
        color = color_mapping[labels[i]]
        ax.scatter(
            X[y == fault_id, 0],
            X[y == fault_id, 1],
            label=labels[i],
            color=color,
            s=0.5
        )
    ax.legend(markerscale=15)


def plot_loss(config_loss, ax=None):
    '''
    Plot loss of network training.

    Args:
        config_loss: Config dictionary that includes training loss
        ax: Matplotlib ax object if available
    '''
    if ax is None:
        _, ax = plt.subplots()
    ax.set_title('Loss')
    ax.plot(config_loss['train_loss'], label='train')
    ax.plot(config_loss['test_loss'], label='test')
    ax.legend()


def get_color_mapping():
    '''
    Color mapping dictionary for different bearing conditions.
    '''
    return {
        'B': 'red',
        'IR': 'green',
        'OR': 'orange',
        'N': 'blue',
        'Ball': 'red',
        'InnerRace': 'green',
        'OuterRace': 'orange',
        'Normal': 'blue',
        'SmallBall': 'tomato',
        'MediumBall': 'red',
        'LargeBall': 'darkred',
        'SmallInnerRace': 'lime',
        'MediumInnerRace': 'green',
        'LargeInnerRace': 'darkgreen',
        'SmallOuterRace': 'moccasin',
        'MediumOuterRace': 'orange',
        'LargeOuterRace': 'darkorange'
    }


def summarize_load_results(path):
    '''
    Summarize the results from results_raw.csv of the operating condition
    transfer experiments and save to results_mean.csv and results_std.csv.

    Args:
        path: Path to results_raw.csv
    '''
    path = Path(path)

    res = pd.read_csv(path)

    group_col = [
        'data.source_domain',
        'data.oc_target',
        'model.loss_fn',
        'model.distance',
        'model.mining_func',
        'kg.kg_trainer'
    ]
    acc_col = [
        col
        for col in res.columns
        if (('acc' in col) or ('score' in col)) and ('train' not in col)
    ]

    res_mean = res.loc[
        :, group_col + acc_col
    ].groupby(group_col, dropna=False).mean().reset_index()
    res_std = res.loc[
        :, group_col + acc_col
    ].groupby(group_col, dropna=False).std().reset_index()

    res_mean[group_col] = res_mean[group_col].fillna('-')
    res_std[group_col] = res_std[group_col].fillna('-')

    res_mean[['clf_test_score', 'oc_acc']] = np.round(
        res_mean[['clf_test_score', 'oc_acc']] * 100, decimals=1
    )
    res_std[['clf_test_score', 'oc_acc']] = np.round(
        res_std[['clf_test_score', 'oc_acc']] * 100, decimals=1
    )

    res_mean.to_csv(path.parent / 'results_mean.csv', index=False)
    res_std.to_csv(path.parent / 'results_std.csv', index=False)

    return res_mean, res_std


def summarize_bearing_results(path, shots):
    '''
    Summarize the results from results_raw.csv of the bearing transfer
    experiments and save to results_mean.csv and results_std.csv.

    Args:
        path: Path to results_raw.csv
        shots: List of shots used for few shot learning
    '''
    path = Path(path)
    res = pd.read_csv(path)

    res['test_acc_0_mean'] = np.nansum(
        res[['transfer.results.CWRU_FE.test_acc',
            'transfer.results.CWRU_DE.test_acc']],
        axis=1
    )

    test_acc_shot_col = ['test_acc_0_mean']
    for shot in shots:
        col_mean = f'test_acc_{shot}_mean'
        col_std = f'test_acc_{shot}_std'
        res[col_mean] = np.nanmean(
            res[res.columns[res.columns.str.endswith(f'.{shot}.test_acc')]],
            axis=1
        )
        res[col_std] = np.nanstd(
            res[res.columns[res.columns.str.endswith(f'.{shot}.test_acc')]],
            axis=1
        )
        test_acc_shot_col.extend([col_mean, col_std])

    group_col = [
        'data.source_domain',
        'model.loss_fn',
        'model.distance',
        'model.mining_func',
        'kg.kg_trainer'
    ]

    res_mean = res.loc[
        :,
        group_col
        + ['clf_test_score']
        + [col for col in test_acc_shot_col if col.endswith('mean')]
    ].groupby(group_col, dropna=False).mean().reset_index()

    res_std = res.loc[
        :,
        group_col + [col for col in test_acc_shot_col if col.endswith('std')]
    ].groupby(
        group_col, dropna=False
    ).agg(lambda x: np.sqrt(np.sum(np.square(x)) / len(x))).reset_index()

    res_std[['test_acc_source_std', 'test_acc_0_std']] = res.loc[
        :, group_col + ['test_acc_0_mean', 'clf_test_score']
    ].groupby(
        group_col,
        dropna=False
    ).std().reset_index()[['test_acc_0_mean', 'clf_test_score']]

    res_mean[group_col] = res_mean[group_col].fillna('-')
    res_std[group_col] = res_std[group_col].fillna('-')

    cols_mean = list(res_mean.columns[res_mean.columns.str.endswith('mean')])
    cols_std = list(res_std.columns[res_std.columns.str.endswith('std')])
    res_mean[cols_mean] = np.round(res_mean[cols_mean] * 100, 1)
    res_std[cols_std] = np.round(res_std[cols_std] * 100, 1)

    res_mean.to_csv(path.parent / 'results_mean.csv', index=False)
    res_std.to_csv(path.parent / 'results_std.csv', index=False)

    return res_mean, res_std
