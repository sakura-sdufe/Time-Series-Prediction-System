# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation
# @Time     : 2024/12/24 20:10
# @Author   : 张浩
# @FileName : dataloader.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .data_convert import split_dataset


class SeqDataset(Dataset):
    def __init__(self, feature, target):
        """
        将 Tensor 数据转化成 Dataset 类型数据。
        :param feature: 3D Tensor, 维度为：(sample_number, time_step, number)。
        :param target: 1D Tensor, 维度为：(sample_number,)。
        """
        self.feature = feature
        self.target = target
        self.input_size = feature.shape[-1]

    def __getitem__(self, item):
        """
        根据索引返回数据集中的一个样本。
        :param item: 表示需要获取的样本索引，int 类型。
        :return: 返回一个 torch.Tensor 类型的样本，特征的尺寸为 (time_step, number)，目标的尺寸为 (batch_size,)。
        """
        return self.feature[item], self.target[item]

    def __len__(self):
        """返回数据集的样本数量。"""
        return self.feature.shape[0]


class SeqLoader:
    def __init__(self, time_feature, time_target, parameters_data, normalization=None):
        """
        将时间序列数据转化成 DataLoader 类型数据，主要用于时间序列预测。
        :param time_feature: 2D 时间特征数据，数据类型为 pd.DataFrame。
        :param time_target: 1D 时间目标数据，数据类型为 pd.Series。
        :param parameters_data: 数据参数，要求具有 __getitem__ 方法。
        :param normalization: 标准化类，需要需要包含 get_norm_result, norm, denorm 方法。如果标准化，那么需要传入标准化类。
        Note: 如果在推理过程中，最后一个时间点没有时变未知变量和目标的话，可以使用 np.nan 或者 None 进行填充。在生成数据集时，会自动剔除这些数据。
        """
        # 按照 时变已知变量 和 时变未知变量 进行数据划分
        time_unknown_variables = [col for col in parameters_data["time_unknown_variables"] if col in time_feature.columns]
        time_known_variables = [col for col in parameters_data["time_known_variables"] if col in time_feature.columns]
        feature_unknown = pd.concat([time_feature[time_unknown_variables], time_target], axis=1)  # 时变未知变量
        feature_known = time_feature[time_known_variables]  # 时变已知变量
        feature = pd.concat([feature_known, feature_unknown.shift(1)], axis=1)
        self.feature, self.target = feature[1:].values, np.expand_dims(time_target.values, axis=1)  # 扩展维度用于标准化
        # 提取参数
        self.normalization = normalization
        self.parameters_data = parameters_data
        self.time_step = parameters_data["time_step"]
        sample_gap = parameters_data["sample_gap"]
        self.train_start_rate, self.train_end_rate = parameters_data["train_start_rate"], parameters_data["train_end_rate"]
        self.valid_start_rate, self.valid_end_rate = parameters_data["valid_start_rate"], parameters_data["valid_end_rate"]
        self.test_start_rate, self.test_end_rate = parameters_data["test_start_rate"], parameters_data["test_end_rate"]
        # 计算样本数量、样本特征起始位置和训练集特征起始位置
        sample_number = len(self.feature) - self.time_step + 1
        self.position = list(range(sample_number))
        self.train_position = list(
            range(round(sample_number*self.train_start_rate), round(sample_number*self.train_end_rate), sample_gap))
        # 生成 DataLoader 类型数据
        self.normalization_feature, self.normalization_target = None, None  # 用于标准化的类
        self.train_dataloader_origin, self.valid_dataloader_origin, self.test_dataloader_origin = None, None, None
        self.train_dataloader_norm, self.valid_dataloader_norm, self.test_dataloader_norm = None, None, None
        self.train_eval_dataloader_origin, self.train_eval_dataloader_norm = None, None  # 用于评估训练集的 DataLoader
        self._generate_loader_origin()
        if normalization:
            self._generate_loader_norm()

    def _generate_loader_origin(self):
        # 将特征和目标转为 Tensor 类型数据
        feature_origin = torch.tensor(self.feature, dtype=torch.float32)
        target_origin = torch.tensor(self.target, dtype=torch.float32)
        # 将时序的特征和目标转为样本的特征和目标，特征维度：(sample_number, time_step, number)，目标维度：(sample_number,)
        feature_origin_sample = torch.stack([feature_origin[start:start+self.time_step, :] for start in self.position])
        target_origin_sample = torch.tensor([target_origin[start + self.time_step].item() for start in self.position])
        # 划分训练集、验证集和测试集
        train_feature_origin, train_target_origin = \
            feature_origin_sample[self.train_position], target_origin_sample[self.train_position]
        train_eval_feature_origin, train_eval_target_origin = split_dataset(
            feature_origin_sample, target_origin_sample, self.train_start_rate, self.train_end_rate)
        valid_feature_origin, valid_target_origin = split_dataset(
            feature_origin_sample, target_origin_sample, self.valid_start_rate, self.valid_end_rate)
        test_feature_origin, test_target_origin = split_dataset(
            feature_origin_sample, target_origin_sample, self.test_start_rate, self.test_end_rate)
        # 封装为 Dataset 类型数据
        self.train_dataset_origin = SeqDataset(train_feature_origin, train_target_origin)
        self.train_eval_dataset_origin = SeqDataset(train_eval_feature_origin, train_eval_target_origin)
        self.valid_dataset_origin = SeqDataset(valid_feature_origin, valid_target_origin)
        self.test_dataset_origin = SeqDataset(test_feature_origin, test_target_origin)
        # 封装为 DataLoader 类型数据
        self.train_dataloader_origin = DataLoader(self.train_dataset_origin,
                                                  batch_size=self.parameters_data["train_batch_size"],
                                                  shuffle=self.parameters_data["dataloader_shuffle"])
        self.train_eval_dataloader_origin = DataLoader(self.train_eval_dataset_origin,
                                                       batch_size=self.parameters_data["eval_batch_size"],
                                                       shuffle=False)
        self.valid_dataloader_origin = DataLoader(self.valid_dataset_origin,
                                                  batch_size=self.parameters_data["eval_batch_size"],
                                                  shuffle=False)
        self.test_dataloader_origin = DataLoader(self.test_dataset_origin,
                                                 batch_size=self.parameters_data["eval_batch_size"],
                                                 shuffle=False)

    def _generate_loader_norm(self):
        # 标准化特征和目标
        self.normalization_feature = self.normalization(self.feature)
        self.normalization_target = self.normalization(self.target)
        feature_norm = self.normalization_feature.get_norm_result()
        target_norm = self.normalization_target.get_norm_result()
        feature_norm = torch.tensor(feature_norm, dtype=torch.float32)
        target_norm = torch.tensor(target_norm, dtype=torch.float32)
        # 将时序的特征和目标转为样本的特征和目标，特征维度：(sample_number, time_step, number)，目标维度：(sample_number,)
        feature_norm_sample = torch.stack([feature_norm[start:start+self.time_step, :] for start in self.position])
        target_norm_sample = torch.tensor([target_norm[start+self.time_step].item() for start in self.position])
        # 划分训练集、验证集和测试集
        train_feature_norm, train_target_norm = \
            feature_norm_sample[self.train_position], target_norm_sample[self.train_position]
        train_eval_feature_norm, train_eval_target_norm = split_dataset(
            feature_norm_sample, target_norm_sample, self.train_start_rate, self.train_end_rate)
        valid_feature_norm, valid_target_norm = split_dataset(
            feature_norm_sample, target_norm_sample, self.valid_start_rate, self.valid_end_rate)
        test_feature_norm, test_target_norm = split_dataset(
            feature_norm_sample, target_norm_sample, self.test_start_rate, self.test_end_rate)
        # 封装为 Dataset 类型数据
        self.train_dataset_norm = SeqDataset(train_feature_norm, train_target_norm)
        self.train_eval_dataset_norm = SeqDataset(train_eval_feature_norm, train_eval_target_norm)
        self.valid_dataset_norm = SeqDataset(valid_feature_norm, valid_target_norm)
        self.test_dataset_norm = SeqDataset(test_feature_norm, test_target_norm)
        # 封装为 DataLoader 类型数据
        self.train_dataloader_norm = DataLoader(self.train_dataset_norm,
                                                batch_size=self.parameters_data["train_batch_size"],
                                                shuffle=self.parameters_data["dataloader_shuffle"])
        self.train_eval_dataloader_norm = DataLoader(self.train_eval_dataset_norm,
                                                     batch_size=self.parameters_data["eval_batch_size"],
                                                     shuffle=False)
        self.valid_dataloader_norm = DataLoader(self.valid_dataset_norm,
                                                batch_size=self.parameters_data["eval_batch_size"],
                                                shuffle=False)
        self.test_dataloader_norm = DataLoader(self.test_dataset_norm,
                                               batch_size=self.parameters_data["eval_batch_size"],
                                               shuffle=False)

    def get_loader(self, dataname, *, user='predict', scale=None):
        """
        获取数据集。
        :param dataname: str, 数据集名称，可选值为 'train', 'valid', 'test'。
        :param user: str，数据集用途，可选 'fit', 'predict'，该参数只有在 dataname 为 'train' 时生效。
        :param scale: bool, 是否获取标准化后的数据集。默认为 None，表示如果有标准化类则设置为 True，如果没有标准化类则设置为 False。
        :return: DataLoader 类型数据。
        """
        if (scale is None) and self.normalization:
            scale = True
        elif (scale is None) and not self.normalization:
            scale = False
        # 获取数据集
        if (dataname == 'train') and (scale == False) and (user == 'fit'):
            return self.train_dataloader_origin
        elif (dataname == 'train') and (scale == False) and (user == 'predict'):
            return self.train_eval_dataloader_origin
        elif (dataname == 'train') and (scale == True) and (user == 'fit'):
            return self.train_dataloader_norm
        elif (dataname == 'train') and (scale == True) and (user == 'predict'):
            return self.train_eval_dataloader_norm
        elif (dataname == 'valid') and (scale == False):
            return self.valid_dataloader_origin
        elif (dataname == 'valid') and (scale == True):
            return self.valid_dataloader_norm
        elif (dataname == 'test') and (scale == False):
            return self.test_dataloader_origin
        elif (dataname == 'test') and (scale == True):
            return self.test_dataloader_norm
        else:
            raise ValueError('请输入正确的参数！dataname 参数只能为 train、valid、test；scale 参数只能为 True 或 False。')

    def get_all_loader(self, scale=None):
        """
        获取所有数据集。
        :param scale: bool, 是否获取标准化后的数据集。默认为 None，表示如果有标准化类则设置为 True，如果没有标准化类则设置为 False。
        :return: DataLoader 类型数据。
        """
        if (scale is None) and self.normalization:
            scale = True
        elif (scale is None) and not self.normalization:
            scale = False
        # 获取数据集
        if scale:
            return self.train_dataloader_norm, self.train_eval_dataloader_norm, \
                self.valid_dataloader_norm, self.test_dataloader_norm
        else:
            return self.train_dataloader_origin, self.train_eval_dataloader_origin, \
                self.valid_dataloader_origin, self.test_dataloader_origin
