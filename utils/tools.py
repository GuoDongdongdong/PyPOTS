import os
import random
from typing import Union

import torch
import numpy as np
import pandas as pd
from pygrinder import mcar

# pypots logger
from pypots.utils.logging import logger_creator, logger


def fix_random_seed(random_seed : int) :
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logger(saving_dir: str, file_name: str = 'logfile') :
    format = "[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s"
    logger_creator.set_logging_format(format)
    logger_creator.set_saving_path(saving_dir, file_name)
    logger_creator.set_level('debug')

"""
    If our original dataset has no missing values, we need to simulate the situation where the dataset has missing values. Note that we will never know these masked data and cannot use them for model testing.
    There are various types of missing data in the original data set. Here we use completely random missing(MCAR).
    TODO: What impact will other situations have on filling if the original missing type of the dataset is not completely random missing?
"""
def mask_original_dataset(dir_name:str, file_name:str, missing_rate: float, targets: list[str]):
    file_path = os.path.join(dir_name, file_name)
    data = pd.read_csv(file_path)

    for col in targets:
        missing_col = col + '_missing'
        data[missing_col] = mcar(np.array(data[col]), p=missing_rate)

    file_name, file_suffix = file_name.split('.')
    data.to_csv(f'{file_name}_{int(missing_rate * 100)}per_missing.{file_suffix}', index=False)

"""
    mask all non-nan values in data with the probability of p
"""
def mcar(data : np.ndarray, p : float):
    res_data = data.copy().reshape(-1)
    observed_mask = (1 - np.isnan(data)).reshape(-1)
    observed_index = np.where(observed_mask)[0]
    artifical_missing_index = np.random.choice(observed_index,
                                               int(len(observed_index) * p),
                                               replace=False)
    res_data[artifical_missing_index] = np.nan
    res_data = res_data.reshape(data.shape)
    return res_data

def calc_mae(x: Union[np.ndarray | torch.Tensor], y: Union[np.ndarray | torch.Tensor], mask:Union[np.ndarray | torch.Tensor]) -> float:
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if isinstance(y, torch.Tensor):
        y = y.numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)
    return np.sum(np.abs(x - y) * mask) / np.sum(mask)

def calc_rmse(x: Union[np.ndarray | torch.Tensor], y: Union[np.ndarray | torch.Tensor], mask:Union[np.ndarray | torch.Tensor]) -> float:
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if isinstance(y, torch.Tensor):
        y = y.numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)
    mse = np.sum(((x - y) ** 2) * mask)
    return np.sqrt(mse / np.sum(mask))