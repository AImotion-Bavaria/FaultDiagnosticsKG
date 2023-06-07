import pickle
import os
import random

import torch
import numpy as np


def prune_config(config):
    '''
    Prune configuration file for saving.

    Args:
        config: Configuration dictionary
    '''
    if not config['kg']['kg_trainer']:
        config.pop('kg', None)
    if config['model']['loss_fn'] == 'CrossEntropyLoss':
        config.pop('kg', None)
        config['model'].pop('distance', None)
        config['model'].pop('reducer', None)
        config['model'].pop('mining_func', None)
    return config


def adjust_config(config):
    '''
    Adjust configuration for training.

    Args:
        config: Configuration dictionary
    '''
    if config['model']['loss_fn'] == 'CrossEntropyLoss':
        config['kg']['kg_trainer'] = False
    return config


def save_pickle(obj, filename):
    '''
    Save object as pickle.
    '''
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle)


def load_pickle(filename):
    '''
    Load pickled file.
    '''
    with open(filename, 'rb') as handle:
        file = pickle.load(handle)
    return file


def set_seed(seed=42):
    '''
    Sets the seed of the entire notebook so results are the same every
    time we run. This is for REPRODUCIBILITY.
    '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
