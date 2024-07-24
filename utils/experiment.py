from typing import Union
import numpy as np
import pandas as pd
from pypots.imputation import CSDI, SAITS, LOCF, BRITS, Transformer, USGAN
import torch
from utils.tools import mcar
from sklearn.preprocessing import StandardScaler


class Experiment:
    def __init__(self, conf: dict):
        self.model_dict = {
            'SAITS'       : SAITS,
            'BRITS'       : BRITS,
            'Transformer' : Transformer,
            'USGAN'       : USGAN,
            'LOCF'        : LOCF,
            'CSDI'        : CSDI
        }
        self.conf   = conf
        model_name  = conf['base']['model']
        self.args   = conf[model_name]
        self.model  = self.model_dict[model_name](**self.args)
        self.scaler = StandardScaler()
        self.train_set, self.validation_set, self.test_set = self._get_dataset()

    # pypots need inputs shape like [samples, n_steps, D], so we happy them happy.
    def _set_shape(self, data:np.ndarray):
        assert data.ndim == 2, 'input shape should be [L, D]'
        L, D = data.shape
        n_steps = self.conf['default']['n_steps']
        new_length = L // n_steps * n_steps
        data = data[:new_length, :]
        data = data.reshape(-1, n_steps, D)
        return data

    def _get_dataset(self):
        dataset_path = self.conf['base']['dataset_path']
        raw_data = pd.read_csv(dataset_path)
        # split dataset to train validation test
        train_ratio      = self.conf['base']['train_ratio']
        validation_ratio = self.conf['base']['validation_ratio']
        total_len        = len(raw_data)
        train_len        = int(total_len * train_ratio)
        validation_len   = int(total_len * validation_ratio)
        data_border_l    = [0, train_len, train_len + validation_len]
        data_border_r    = [train_len, train_len + validation_len, total_len]
        target = self.conf['base']['target']
        # DataFrame [L, D]
        data = raw_data[target]
        # ndarry [L, D]
        train_data = data[data_border_l[0]:data_border_r[0]].values
        validation_data = data[data_border_l[1]:data_border_r[1]].values
        test_data = data[data_border_l[2]:data_border_r[2]].values
        # normalization
        train_data = self.scaler.fit_transform(train_data)
        validation_data = self.scaler.transform(validation_data)
        test_data = self.scaler.transform(test_data)
        # ndarray [samples, n_steps, D]
        train_data = self._set_shape(train_data)
        validation_data = self._set_shape(validation_data)
        test_data = self._set_shape(test_data)

        missing_rate = self.conf['base']['missing_rate']
        # dataset for CSDI could be like:
        '''
            train_dataset = {
                'X' # original dataset
                'for_pattern_mask'
                'time_points'
            }
            validataion_dataset = {
                'X_ori' # orginal dataset
                'X'     # X_ori with artifical missing
                'for_pattern_mask'
                'time_points'
            }
        '''
        train_set = {
            'X' : train_data
        }
        validation_set = {
            'X_ori' : validation_data,
            'X' : mcar(validation_data, missing_rate)
        }
        test_set = {
            'X_ori' : test_data,
            'X' : mcar(test_data, missing_rate)
        }
        return train_set, validation_set, test_set

    def _inverse_ndarry(self, data : np.ndarray):
        assert data.ndim == 2, f"innverse need data's shape like [length, features], but got {data.shape}"
        return self.scaler.inverse_transform(data)

    def _inverse_tensor(self, data : torch.Tensor) :
        device = data.device
        res = self._inverse_ndarry(data.cpu().numpy())
        return torch.from_numpy(res).to(device)

    def inverse(self, data : Union[np.ndarray | torch.Tensor]) :
        if isinstance(data, np.ndarray):
            return self._inverse_ndarry(data)
        if isinstance(data, torch.Tensor):
            return self._inverse_tensor(data)
        raise TypeError(f'only support ndarry and Tensor, but got {type(data)}.')

    def fit(self):
        self.model.fit(train_set=self.train_set, val_set=self.validation_set)

    def impute(self) -> np.ndarray:
        imputation = self.model.impute(test_set=self.test_set)
        pass

    def load(self, path: str):
        self.model.load(path)