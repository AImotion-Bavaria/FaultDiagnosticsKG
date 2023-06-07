from pathlib import Path
import re
import logging

from scipy.io import loadmat
from scipy.signal import decimate

import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset

import utils
import utils_kg


def cwru_load(path, dom, sample_rate, segment_length):
    '''
    Load faulty and healthy CWRU data for specific bearing into dataframe.
    Signals are downsampled to sample_rate and split into signals of size
    segment_length.

    Args:
        path: Path to faulty data
        dom: Bearing (CWRU_DE or CWRU_FE)
        sample_rate: Sample rate to which the original signal is downsampled
        segment_length: Length into which the orignal signals are segmented

    Returns:
        df_cwru: Dataframe with segmented signals
    '''
    if dom == 'CWRU_DE':
        measure_type = 'DE_time'
    elif dom == 'CWRU_FE':
        measure_type = 'FE_time'
    path = Path(path)
    path_data = path.parent
    df_fault = cwru_to_df(
        path,
        segment_length,
        measure_type,
        sample_rate
    )
    df_normal = cwru_to_df(
        path_data / 'Normal',
        segment_length,
        measure_type,
        sample_rate
    )
    df_cwru = pd.concat([df_fault, df_normal], axis=0).reset_index(drop=True)
    return df_cwru


def cwru_load_raw_data(path, measure_type, force_reload=False):
    '''
    Load raw CWRU data and return dictionary from matlab files.

    Args:
        path: Path to data
        measure_type: Which sensor is used (FE_time or DE_time)
        force_reload: Force reload from source

    Returns:
        data: Dictionary with raw CWRU data
    '''
    dom = path.name
    file_name = f'cwru_{dom}_{measure_type}_data.pkl'
    if (path / file_name).is_file() and not force_reload:
        logging.info('Load %s.', file_name)
        data = utils.load_pickle(path / file_name)
        return data
    logging.info('Load CWRU data from source.')
    all_files = path.glob('*.mat')
    data = {}
    for i in all_files:
        name = i.name
        # data[i] = loadmat(path / i)
        data[name] = loadmat(i)
        k = [j for j in data[name].keys() if j.endswith(measure_type)][0]
        data[name] = data[name][k]
        data[name] = data[name].reshape(data[i].shape[0])
    utils.save_pickle(data, path / file_name)
    return data


def cwru_to_df(path, segment_length, measure_type, sample_rate=12):
    '''
    Transform CWRU dictionary to dataframe and split signals
    in segments of specified length.

    Args:
        path: Path to CWRU_xx data
        segment_length: Length into which signals are segmented
        measure_type: Which sensor is used (FE_time or DE_time)
        sample_rate: Sample rate which to downsample Normal signal to (kHz)

    Returns:
        df_cwru: Dataframe with CWRU_xx data
    '''
    is_normal = str(path).endswith('Normal')
    name_mapping = get_name_mapping()
    size_mapping = get_size_mapping()

    path = Path(path)
    data = cwru_load_raw_data(path, measure_type)
    dic_cwru = {}
    idx = 0
    if is_normal:
        if 48000 != sample_rate:
            down_sample = 48 // sample_rate
            for k in list(data.keys()):
                # Downsample from 48kHz to sample_rate
                data[k] = decimate(data[k], q=down_sample)
    for k, data_k in data.items():
        n_sample_points = len(data_k)
        n_segments = n_sample_points // segment_length
        k_pos = name_mapping[re.search(r'([A-Z]*)', k)[1]]
        if is_normal:
            k_error_size = ''
        else:
            k_error_size = re.search(r"([0-9]{3})", k)[1]
            # Exclude 028 bearing faults:
            if k_error_size == '028':
                continue
            k_error_size = size_mapping[k_error_size]
        k_rpm = re.search(r"(\d+)(?!.*\d)", k)[1]
        k_pos_err = f"{k_error_size}{k_pos}"
        k_labels = [k, k_pos, k_error_size, k_rpm, k_pos_err]
        for segment in range(n_segments):
            segment_data = data_k[
                segment_length * segment:segment_length*(segment+1)
            ]
            seg_out = k_labels + list(segment_data)
            dic_cwru[idx] = seg_out
            idx += 1
    df_cwru = pd.DataFrame(dic_cwru).T
    label_cols = ["FileName", "Position", "Size", "RPM",
                  "PositionSize"]
    for i, col in enumerate(label_cols):
        df_cwru.rename(columns={df_cwru.columns[i]: col}, inplace=True)

    return df_cwru


def create_mappings(target, ent_to_index, data, kg=False):
    '''
    Create mapping between label names and numerical representation.
    Map this to KG embedding if applicable.

    Args:
        target: Name of target column
        ent_to_index: Mapping dictionary from entities to index
        kg: Whether or not a KG is used for training
    '''
    class_label_mapping = {}
    target_ls = sorted(list(set(data[target])))
    class_label_mapping['data_target'] = {
        lab: i
        for i, lab in enumerate(target_ls)
    }
    if kg:
        class_label_mapping = utils_kg.match_kg_data_labels(
            class_label_mapping, ent_to_index
        )
    return class_label_mapping


def get_name_mapping():
    '''
    Name mapping.
    '''
    return {'B': 'Ball',
            'IR': 'InnerRace',
            'OR': 'OuterRace',
            'N': 'Normal'}


def get_size_mapping():
    '''
    Size mapping.
    '''
    return {'007': 'Small',
            '014': 'Medium',
            '021': 'Large'}


def split_oc(data, oc_target):
    '''
    Split data according to operating condition.

    Args:
        data: CWRU bearing dataframe
        oc_target: Target operating condition

    Returns:
        source: CWRU bearing dataframe form source domain
        target: CWRU bearing dataframe form target domain
    '''
    source = data.loc[data.RPM != oc_target]
    target = data.loc[data.RPM == oc_target]
    return source, target


def create_target_columns(data, target, data_target_mapping):
    '''
    Create target column with label numbers.

    Args:
        data: CWRU bearing dataframe
        target: Target column name
        data_target_mapping: Mapping dictionary from label names
            to label numbers

    Returns
        data: CWRU bearing dataframe with target column
    '''
    data['y'] = data[target].map(
        data_target_mapping
    )
    return data


def get_x_y(data):
    '''
    Get features (X) and labels (y) from CWRU bearing dataframe.

    Args:
        data: CWRU bearing dataframe

    Returns:
        X: Features
        y: Labels
    '''
    sig_cols = [col for col in data.columns if isinstance(col, int)]
    X = data.loc[:, sig_cols].astype(float).values
    y = data['y'].values
    return X, y


def create_tensor_dataset(X, y, y_emb=None):
    '''
    Create torch tensor dataset.

    Args:
        X: Feature matrix
        y: Label vector
        y_emb: Embedding of label vector

    Returns:
        TensorDataset
    '''
    data_shape = (-1, 1, X.shape[1])
    if y_emb is not None:
        return TensorDataset(
            torch.tensor(X).reshape(data_shape),
            torch.tensor(y),
            y_emb
        )
    return TensorDataset(
        torch.tensor(X).reshape(data_shape),
        torch.tensor(y)
    )


def create_dataloader(x, y, shuffle=True):
    '''
    Create dataloader for x and y.

    Args:
        x: Features
        y: Labels
        shuffle: Shuffle dataloader

    Returns:
        DataLoader
    '''
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)
    return DataLoader(
        dataset=TensorDataset(x, y),
        batch_size=64,
        shuffle=shuffle
    )
