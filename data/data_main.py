# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation-Project
# @Time     : 2024/11/17 19:34
# @Author   : 张浩
# @FileName : data_main.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import pandas as pd

from .data_encapsulation.dataset_split import split_dataset
from .data_encapsulation.convert_feature import convert_feature


class DataSplit:
    """为时间序列数据添加历史特征，分割数据集，标准化数据集，转化成 ndarray 数据集。"""
    def __init__(self, time_feature: pd.DataFrame, time_target: pd.Series, parameters_data, normalization=None):
        self.time_feature, self.time_target = time_feature, time_target  # 时间特征和目标
        self.parameters_data = parameters_data  # DataParameters 类
        self.scaler = normalization  # 标准化类

        # 对特征和标签进行数据放缩（Norm）
        if self.scaler:
            self.normalization_feature = normalization(time_feature)
            self.normalization_target = normalization(time_target)
            self.time_feature_norm = self.normalization_feature.get_norm_result()
            self.time_target_norm = self.normalization_target.get_norm_result()

        # 数据封装：在特征中添加历史信息（分为标准化后和不标准化两种）
        self.feature, self.target, self.feature_norm, self.target_norm = [None]*4
        self._add_history_feature()

        # 分割数据集（有的模型要求标准化，有的模型要求不标准化）（数据类型为 DataFrame 和 Series）
        self.train_feature, self.valid_feature, self.test_feature = [None]*3
        self.train_target, self.valid_target, self.test_target = [None]*3
        self._split_dataset()
        if self.scaler:
            self.train_feature_norm, self.valid_feature_norm, self.test_feature_norm = [None] * 3
            self.train_target_norm, self.valid_target_norm, self.test_target_norm = [None] * 3
            self._split_dataset_norm()

    def _add_history_feature(self):
        """添加历史特征。"""
        # 没有标准化的特征和目标
        self.feature, self.target = convert_feature(
            self.time_feature, self.time_target,
            time_unknown_variables = self.parameters_data["time_unknown_variables"],
            time_known_variables = self.parameters_data["time_known_variables"],
            history_steps = self.parameters_data["history_steps"],
            is_features_history = self.parameters_data["is_features_history"]
        )
        if self.scaler:
            self.feature_norm, self.target_norm = convert_feature(
                self.time_feature_norm, self.time_target_norm,
                time_unknown_variables = self.parameters_data["time_unknown_variables"],
                time_known_variables = self.parameters_data["time_known_variables"],
                history_steps = self.parameters_data["history_steps"],
                is_features_history = self.parameters_data["is_features_history"]
            )

    def _split_dataset(self):
        self.train_feature, self.train_target = split_dataset(
            self.feature, self.target,
            start_position = self.parameters_data["train_start_rate"],
            end_position = self.parameters_data["train_end_rate"]
        )
        self.valid_feature, self.valid_target = split_dataset(
            self.feature, self.target,
            start_position = self.parameters_data["valid_start_rate"],
            end_position = self.parameters_data["valid_end_rate"]
        )
        self.test_feature, self.test_target = split_dataset(
            self.feature, self.target,
            start_position = self.parameters_data["test_start_rate"],
            end_position = self.parameters_data["test_end_rate"]
        )

    def _split_dataset_norm(self):
        self.train_feature_norm, self.train_target_norm = split_dataset(
            self.feature_norm, self.target_norm,
            start_position = self.parameters_data["train_start_rate"],
            end_position = self.parameters_data["train_end_rate"]
        )
        self.valid_feature_norm, self.valid_target_norm = split_dataset(
            self.feature_norm, self.target_norm,
            start_position = self.parameters_data["valid_start_rate"],
            end_position = self.parameters_data["valid_end_rate"]
        )
        self.test_feature_norm, self.test_target_norm = split_dataset(
            self.feature_norm, self.target_norm,
            start_position = self.parameters_data["test_start_rate"],
            end_position = self.parameters_data["test_end_rate"]
        )

    def get_dataset(self, dataset=None, scale=True, to_numpy=True):
        """获取数据集。"""
        if (dataset=='train') and (scale==False):
            feature, target = self.train_feature, self.train_target
        elif (dataset=='valid') and (scale==False):
            feature, target = self.valid_feature, self.valid_target
        elif (dataset=='test') and (scale==False):
            feature, target = self.test_feature, self.test_target
        elif (dataset=='train') and (scale==True) and self.scaler:
            feature, target = self.train_feature_norm, self.train_target_norm
        elif (dataset=='valid') and (scale==True) and self.scaler:
            feature, target = self.valid_feature_norm, self.valid_target_norm
        elif (dataset=='test') and (scale==True) and self.scaler:
            feature, target = self.test_feature_norm, self.test_target_norm
        else:
            raise ValueError('请输入正确的参数！dataset 参数只能为 train、valid、test；scale 参数只能为 True 或 False。')
        if to_numpy:
            feature, target = feature.to_numpy(), target.to_numpy()
        return feature, target

    def persistence_predict(self, dataset=None):
        """获取持久性模型的预测结果。"""
        col_name = self.parameters_data["target"] + "_1"
        if dataset == 'train':
            return self.train_feature[col_name].to_numpy()
        elif dataset == 'valid':
            return self.valid_feature[col_name].to_numpy()
        elif dataset == 'test':
            return self.test_feature[col_name].to_numpy()
        else:
            raise ValueError('请输入正确的参数！dataset 参数只能为 train、valid、test。')
