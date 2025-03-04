# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation
# @Time     : 2025/2/17 17:27
# @Author   : 张浩
# @FileName : ensemble_DL.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import os
import torch
import traceback
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class EnsembleDataset(Dataset):
    def __init__(self, feature, target):
        """
        将 Tensor 数据转为 Dataset 类型数据。
        :param feature: 特征数据。数据类型为 torch.Tensor，维度为 (sample_number, feature_dim)。
        :param target: 目标数据。数据类型为 torch.Tensor，维度为 (sample_number, )。
        """
        self.feature = feature
        self.target = target
        self.sample_number = feature.shape[0]
        self.input_size = feature.shape[1]

    def __getitem__(self, item):
        """
        根据索引返回数据集中的一个样本。
        :param item: 表示需要获取的样本索引，int 类型。
        :return: 返回一个 torch.Tensor 类型的样本，特征的尺寸为 (1, feature_dim)，目标的尺寸为 (1,)。
        """
        return self.feature[item], self.target[item]

    def __len__(self):
        """
        返回数据集的样本数量。
        :return: 数据集的样本数量，int 类型。
        """
        return self.sample_number


def EnsembleOnceLoader(feature, target, batch_size, shuffle):
    """
    将多个预测结果和目标值组合成一个数据集，如果存在特征值也会加入特征值。
    :param feature: 预测结果。数据类型为 pd.Series, pd.DataFrame, 1D np.ndarray 或 2D np.ndarray。
    :param target: 目标。数据类型为 pd.Series, pd.DataFrame, 1D np.ndarray。
    :param batch_size: DataLoader 的批大小。
    :param shuffle: 是否打乱数据集。
    """
    # 检查输入数据类型
    SUPPORT_TYPES = (pd.Series, pd.DataFrame, np.ndarray)
    assert isinstance(target, SUPPORT_TYPES), f"target 的数据类型只能是 {SUPPORT_TYPES} 中的一种。"
    assert isinstance(feature, SUPPORT_TYPES), f"feature 的数据类型只能是 {SUPPORT_TYPES} 中的一种。"
    # 转换为 np.ndarray 类型、并检查维度
    target_value = target if isinstance(target, np.ndarray) else target.values
    target_value = np.squeeze(target_value) if target_value.ndim > 1 else target_value  # 1D ndarray
    feature_value = np.squeeze(feature) if isinstance(feature, np.ndarray) else feature.values
    feature_value = np.expand_dims(feature_value, axis=1) if feature_value.ndim == 1 else feature_value  # 2D ndarray
    assert target_value.ndim == 1 and feature_value.ndim == 2, "目标的维度必须为 1，特征的维度必须为 2。"
    # 记录目标和预测列名
    if isinstance(target, pd.Series):
        target_name = target.name
    elif isinstance(target, pd.DataFrame):
        target_name = target.columns[0]  # 只有一列
    else:
        target_name = 'target'
    if isinstance(feature, pd.Series):
        feature_names = [feature.name]
    elif isinstance(feature, pd.DataFrame):
        feature_names = feature.columns.tolist()  # 多列
    else:
        feature_names = [f'feature_{i}' for i in range(1, feature_value.shape[1]+1)]
    # 转为 Tensor 类型
    target_value = torch.from_numpy(target_value).float()
    feature_value = torch.from_numpy(feature_value).float()
    # 创建 DataSet 和 DataLoader
    dataset = EnsembleDataset(feature_value, target_value)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, feature_names, target_name


class EnsembleLoader:
    def __init__(self, result_dir, batch_size, normalization=None):
        """
        将多个模型的预测结果和目标值组合成一个 DataLoader 数据集。
        :param result_dir: 预测结果所在的根目录。
        :param batch_size: DataLoader 的批大小。
        :param normalization: 数据标准化方法。默认值为 None，不进行标准化。
        """
        assert os.path.isdir(result_dir), f"{result_dir} 不是一个目录或该目录不存在。"
        # 读取 .result 识别文件
        assert os.path.exists(os.path.join(result_dir, ".result")), "未找到 .result 识别文件，无法读取预测结果和特征值。"
        IDENTIFY_KEYS = {'VERSION', 'TIME', 'PATH'}
        with open(os.path.join(result_dir, ".result"), 'rb') as f:  # 读取识别文件
            IDENTIFY_read = f.read().decode('utf-8')
        IDENTIFY_lines = IDENTIFY_read.split('\n')
        for line in IDENTIFY_lines:
            key, value = line.split(': ')
            if key == 'PATH':
                assert os.path.samefile(value, result_dir), "识别文件中的路径与当前路径不一致。"
                IDENTIFY_KEYS.remove(key)
            elif key == 'VERSION':
                IDENTIFY_KEYS.remove(key)
            elif key == 'TIME':
                IDENTIFY_KEYS.remove(key)
        assert not IDENTIFY_KEYS, "识别文件内容不完整。"

        self.batch_size = batch_size
        self.normalization = normalization
        # 定位和读取预测结果文件
        try:
            train_file = pd.read_excel(os.path.join(result_dir, 'results', 'train predict.xlsx'), index_col=0)
            valid_file = pd.read_excel(os.path.join(result_dir, 'results', 'valid predict.xlsx'), index_col=0)
            test_file = pd.read_excel(os.path.join(result_dir, 'results', 'test predict.xlsx'), index_col=0)
        except FileNotFoundError as e:
            error_message = traceback.format_exc()
            full_message = (f"{error_message}\n未找到预测结果文件，请检查文件是否存在！预测结果文件应位于 './results' 目录下"
                            f" 且文件名为 'train predict.xlsx', 'valid predict.xlsx', 'test predict.xlsx'。")
            raise FileNotFoundError(full_message)
        # 对预测结果文件进行整合和记录
        self.train_number, self.valid_number, self.test_number = len(train_file), len(valid_file), len(test_file)
        all_file = pd.concat([train_file, valid_file, test_file], axis=0)
        self.feature, self.target = all_file.drop(columns='True'), all_file['True']
        self.feature_names, self.target_name = None, None  # 特征列名、目标列名
        # 初始化未标准化的 DataLoader
        self.train_trainer, self.train_evaler, self.valid_evaler, self.test_evaler = None, None, None, None
        self._generate_loader_origin()
        if normalization:
            # 标准化对象
            # self.normalization_feature = self.normalization(self.feature)
            self.normalization_target = self.normalization(self.target)
            self.feature_norm, self.target_norm = None, None
            # 生成标准化后的 DataLoader
            self.train_trainer_norm, self.train_evaler_norm = None, None
            self.valid_evaler_norm, self.test_evaler_norm = None, None
            self._generate_loader_norm()

    def _generate_loader_origin(self):
        """获取 train_trainer, train_evaler, valid_evaler, test_evaler（未标准化）"""
        self.train_trainer, _, _ = EnsembleOnceLoader(
            feature=self.feature.iloc[:self.train_number],
            target=self.target.iloc[:self.train_number],
            batch_size=self.batch_size, shuffle=True
        )
        self.train_evaler, train_feature_names, train_target_name = EnsembleOnceLoader(
            feature=self.feature.iloc[:self.train_number],
            target=self.target.iloc[:self.train_number],
            batch_size=self.batch_size, shuffle=False
        )
        self.valid_evaler, valid_feature_names, valid_target_name = EnsembleOnceLoader(
            feature=self.feature.iloc[self.train_number : self.train_number+self.valid_number],
            target=self.target.iloc[self.train_number : self.train_number+self.valid_number],
            batch_size=self.batch_size, shuffle=False
        )
        self.test_evaler, test_feature_names, test_target_name = EnsembleOnceLoader(
            feature=self.feature.iloc[self.train_number+self.valid_number :],
            target=self.target.iloc[self.train_number+self.valid_number :],
            batch_size=self.batch_size, shuffle=False
        )
        assert train_feature_names == valid_feature_names == test_feature_names, "特征列名不一致！"
        assert train_target_name == valid_target_name == test_target_name, "目标列名不一致！"
        self.feature_names, self.target_name = train_feature_names, train_target_name

    def _generate_loader_norm(self):
        """获取 train_trainer, train_evaler, valid_evaler, test_evaler（标准化）"""
        # 获取标准化后的数据
        # self.feature_norm = self.normalization_feature.get_norm_result()
        self.target_norm = self.normalization_target.get_norm_result()
        self.feature_norm = self.normalization_target.norm(self.feature, numbers=0)
        # 生成 DataLoader
        self.train_trainer_norm, _, _ = EnsembleOnceLoader(
            feature=self.feature_norm.iloc[:self.train_number],
            target=self.target_norm.iloc[:self.train_number],
            batch_size=self.batch_size, shuffle=True
        )
        self.train_evaler_norm, _, _ = EnsembleOnceLoader(
            feature=self.feature_norm.iloc[:self.train_number],
            target=self.target_norm.iloc[:self.train_number],
            batch_size=self.batch_size, shuffle=False
        )
        self.valid_evaler_norm, _, _ = EnsembleOnceLoader(
            feature=self.feature_norm.iloc[self.train_number : self.train_number+self.valid_number],
            target=self.target_norm.iloc[self.train_number : self.train_number+self.valid_number],
            batch_size=self.batch_size, shuffle=False
        )
        self.test_evaler_norm, _, _ = EnsembleOnceLoader(
            feature=self.feature_norm.iloc[self.train_number+self.valid_number :],
            target=self.target_norm.iloc[self.train_number+self.valid_number :],
            batch_size=self.batch_size, shuffle=False
        )

    def get_loader(self, dataname, *, mode='predict', scale=None):
        """
        获取指定数据集的 DataLoader。
        :param dataname: str，数据集名称。可选值为 'train', 'valid', 'test'。
        :param mode: str，数据集用途，可选 'fit', 'predict'，该参数只有在 dataname 为 'train' 时生效。
        :param scale: bool, 是否获取标准化后的数据集。默认为 None，表示如果有标准化类则设置为 True，如果没有标准化类则设置为 False。
        :return: DataLoader 类型数据。
        """
        if (scale is None) and self.normalization:
            scale = True
        elif (scale is None) and not self.normalization:
            scale = False
        # 获取 DataLoader
        if (dataname == 'train') and (scale is False) and (mode == 'fit'):
            return self.train_trainer
        elif (dataname == 'train') and (scale is False) and (mode == 'predict'):
            return self.train_evaler
        elif (dataname == 'train') and (scale is True) and (mode == 'fit'):
            return self.train_trainer_norm
        elif (dataname == 'train') and (scale is True) and (mode == 'predict'):
            return self.train_evaler_norm
        elif (dataname == 'valid') and (scale is False):
            return self.valid_evaler
        elif (dataname == 'valid') and (scale is True):
            return self.valid_evaler_norm
        elif (dataname == 'test') and (scale is False):
            return self.test_evaler
        elif (dataname == 'test') and (scale is True):
            return self.test_evaler_norm
        else:
            raise ValueError('请输入正确的参数！dataname 参数只能为 train、valid、test；scale 参数只能为 True 或 False。')

    def get_all_loader(self, scale=None):
        """
        获取所有的 DataLoader。
        :param scale: bool, 是否获取标准化后的数据集。默认为 None，表示如果有标准化类则设置为 True，如果没有标准化类则设置为 False。
        :return: DataLoader 类型数据。
        """
        if (scale is None) and self.normalization:
            scale = True
        elif (scale is None) and not self.normalization:
            scale = False
        # 获取 DataLoader
        if scale:
            return self.train_trainer_norm, self.train_evaler_norm, self.valid_evaler_norm, self.test_evaler_norm
        else:
            return self.train_trainer, self.train_evaler, self.valid_evaler, self.test_evaler

    def get_names(self):
        """
        获取数据集的特征列名和目标列名。
        :return: 特征列名和目标列名。
        """
        return self.feature_names, self.target_name
