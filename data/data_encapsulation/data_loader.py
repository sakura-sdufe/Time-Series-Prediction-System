# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation
# @Time     : 2024/12/24 20:10
# @Author   : 张浩
# @FileName : data_loader.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .data_convert import split_dataset


class Seqset(Dataset):
    def __init__(self, feature, target):
        """
        将 Tensor 数据转化成 Dataset 类型数据。
        :param feature: 3D Tensor, 维度为：(sample_number, time_step, feature_number)。
        :param target: 1D Tensor, 维度为：(sample_number,)。
        """
        self.feature = feature
        self.target = target

    def __getitem__(self, item):
        """
        根据索引返回数据集中的一个样本。
        :param item: 表示需要获取的样本索引，int 类型。
        :return: 返回一个 torch.Tensor 类型的样本，特征的尺寸为 (time_step, feature_number)，目标的尺寸为 (batch_size,)。
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
        feature, target = feature[1:].values, np.expand_dims(time_target.values, 1)
        # 标准化数据
        if normalization:
            self.normalization_feature = normalization(feature)
            self.normalization_target = normalization(target)
            feature = self.normalization_feature.get_norm_result()
            target = self.normalization_target.get_norm_result()
        feature = torch.tensor(feature, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        # 提取参数。
        time_step = parameters_data["time_step"]
        sample_gap = parameters_data["sample_gap"]
        train_start_rate, train_end_rate = parameters_data["train_start_rate"], parameters_data["train_end_rate"]
        valid_start_rate, valid_end_rate = parameters_data["valid_start_rate"], parameters_data["valid_end_rate"]
        test_start_rate, test_end_rate = parameters_data["test_start_rate"], parameters_data["test_end_rate"]

        # 将时序的特征和目标转为样本的特征和目标，特征维度：(sample_number, time_step, feature_number)，目标维度：(sample_number,)
        sample_number = len(feature) - time_step + 1
        position = list(range(sample_number))
        self.feature_tensor = torch.stack([feature[start:start+time_step, :] for start in position])
        self.target_tensor = torch.tensor([target[start+time_step].item() for start in position])

        # 划分 验证集和测试集
        self.valid_feature, self.valid_target = split_dataset(
            self.feature_tensor, self.target_tensor, valid_start_rate, valid_end_rate)
        self.test_feature, self.test_target = split_dataset(
            self.feature_tensor, self.target_tensor, test_start_rate, test_end_rate)
        # 划分 训练集
        train_position = list(range(round(sample_number*train_start_rate), round(sample_number*train_end_rate), sample_gap))
        self.train_feature, self.train_target = self.feature_tensor[train_position], self.target_tensor[train_position]

        # 封装为 Dataset 类型数据
        self.train_dataset = Seqset(self.train_feature, self.train_target)
        self.valid_dataset = Seqset(self.valid_feature, self.valid_target)
        self.test_dataset = Seqset(self.test_feature, self.test_target)
        # 封装为 DataLoader 类型数据
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=parameters_data["train_batch"],
                                           shuffle=parameters_data["dataloader_shuffle"])
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=parameters_data["test_batch"], shuffle=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=parameters_data["test_batch"], shuffle=False)

    def get_loader(self, dataname):
        """
        获取数据集。
        :param dataname: str, 数据集名称，可选值为 'train', 'valid', 'test'。
        :return: DataLoader 类型数据。
        """
        if dataname == 'train':
            return self.train_dataloader
        elif dataname == 'valid':
            return self.valid_dataloader
        elif dataname == 'test':
            return self.test_dataloader
        else:
            raise ValueError("dataname 只能取值为 'train', 'valid', 'test'。")

    def get_all_loader(self):
        """
        获取所有数据集。
        :return: DataLoader 类型数据。
        """
        return self.train_dataloader, self.valid_dataloader, self.test_dataloader
