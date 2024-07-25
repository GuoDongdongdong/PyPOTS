import os
from typing import Union

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pypots.imputation import CSDI, SAITS, LOCF, BRITS, Transformer, USGAN

from utils.tools import logger, mcar, calc_mae, calc_rmse


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
    # def _set_shape(self, data:np.ndarray):
    #     assert data.ndim == 2, 'input shape should be [L, D]'
    #     L, D = data.shape
    #     n_steps = self.conf['default']['n_steps']
    #     new_length = L // n_steps * n_steps
    #     data = data[:new_length, :]
    #     data = data.reshape(-1, n_steps, D)
    #     return data

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
        # for result visualization
        self.visual_data = raw_data[data_border_l[2]:data_border_r[2]].values
        # normalization
        train_data = self.scaler.fit_transform(train_data)
        validation_data = self.scaler.transform(validation_data)
        test_data = self.scaler.transform(test_data)
        '''
        pypots need inputs shape like [samples, n_steps, D], but it cause the training process to see fewer batch. it is harmful to training, so we modified BaseDataset DatasetForCSDI class in pypots.
        '''

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

    def save_csv(self, imputation : Union[np.ndarray | torch.Tensor], path):
        if isinstance(imputation, torch.Tensor):
            imputation = imputation.numpy()
        assert imputation.ndim == 2, f'imputation shape shoule be like [L, D], but got {imputation.shape}'
        L, D = imputation.shape
        df = pd.DataFrame(self.visual_data[: L])
        df['imputation'] = imputation
        df.to_csv(os.path.join(path, 'result.csv'), index=False, float_format='%.3f')

    def impute(self) -> np.ndarray:
        load_path  = self.conf['base']['load_path']
        n_samples  = self.conf['base']['n_samples']
        parent_dir = os.path.dirname(load_path)
        self.load(load_path)
        results = self.model.impute(test_set=self.test_set, n_sampling_times=n_samples)
        # [n_samples, L, D]
        n_samples_imputation = results['imputation']
        imputation_median    = results['imputation_median']
        # denormalization
        for i in range(n_samples):
            n_samples_imputation[i, :, :] = self.inverse(n_samples_imputation[i, :, :])
        imputation_median = self.inverse(imputation_median)

        observed_data = self.test_set['X_ori']
        observed_mask = 1 - np.isnan(observed_data)
        gt_mask       = 1 - np.isnan(self.test_set['X'])
        target_mask   = observed_mask - gt_mask
        mae = calc_mae(observed_data, imputation_median, target_mask)
        rmse = calc_rmse(observed_data, imputation_median, target_mask)
        logger.info(f'MAE: {mae}, RMSE: {rmse}')

        self.save_csv(imputation=imputation_median, path=parent_dir)
        np.save(os.path.join(parent_dir, 'n_samples_imputation.npy'), n_samples_imputation)

    def load(self, path: str):
        self.model.load(path)