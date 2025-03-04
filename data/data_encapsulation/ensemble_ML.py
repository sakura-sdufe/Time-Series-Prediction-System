# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation
# @Time     : 2025/2/26 10:23
# @Author   : 张浩
# @FileName : ensemble_ML.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import os
import traceback
import pandas as pd

class EnsembleSplit:
    def __init__(self, result_dir, normalization=None):
        """
        将多个模型的预测结果作为回归的特征；真实值作为回归的目标。
        :param result_dir: 预测结果所在的根目录。
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
        train_number, valid_number, test_number = len(train_file), len(valid_file), len(test_file)
        all_file = pd.concat([train_file, valid_file, test_file], axis=0)
        self.feature, self.target = all_file.drop(columns='True'), all_file['True']
        # 划分数据集
        self.train_feature, self.train_target = self.feature.iloc[:train_number], self.target.iloc[:train_number]
        self.valid_feature, self.valid_target = (self.feature.iloc[train_number : train_number+valid_number],
                                                 self.target.iloc[train_number : train_number+valid_number])
        self.test_feature, self.test_target = (self.feature.iloc[train_number+valid_number :],
                                               self.target.iloc[train_number+valid_number :])
        if normalization:
            self.normalization_feature = normalization(self.feature)
            self.normalization_target = normalization(self.target)
            self.feature_norm = self.normalization_feature.get_norm_result()
            self.target_norm = self.normalization_target.get_norm_result()
            # 划分标准化数据集
            self.train_feature_norm, self.train_target_norm = (self.feature_norm.iloc[:train_number],
                                                               self.target_norm.iloc[:train_number])
            self.valid_feature_norm, self.valid_target_norm = (self.feature_norm.iloc[train_number : train_number+valid_number],
                                                               self.target_norm.iloc[train_number : train_number+valid_number])
            self.test_feature_norm, self.test_target_norm = (self.feature_norm.iloc[train_number+valid_number :],
                                                             self.target_norm.iloc[train_number+valid_number :])

    def get_dataset(self, dataname, *, scale=None, to_numpy=True):
        """
        获取数据集。
        :param dataname: str, 数据集名称，可选值为 'train', 'valid', 'test'。
        :param scale: bool, 是否获取标准化后的数据集。默认为 None，表示如果有标准化类则设置为 True，如果没有标准化类则设置为 False。
        :param to_numpy: bool, 是否转换为 numpy 类型数据。
        :return: 返回特征和目标数据。
        """
        if (scale is None) and self.normalization:
            scale = True
        elif (scale is None) and not self.normalization:
            scale = False

        if (dataname == 'train') and (scale == False):
            feature, target = self.train_feature, self.train_target
        elif (dataname == 'valid') and (scale == False):
            feature, target = self.valid_feature, self.valid_target
        elif (dataname == 'test') and (scale == False):
            feature, target = self.test_feature, self.test_target
        elif (dataname == 'train') and (scale == True) and self.normalization:
            feature, target = self.train_feature_norm, self.train_target_norm
        elif (dataname == 'valid') and (scale == True) and self.normalization:
            feature, target = self.valid_feature_norm, self.valid_target_norm
        elif (dataname == 'test') and (scale == True) and self.normalization:
            feature, target = self.test_feature_norm, self.test_target_norm
        else:
            raise ValueError('请输入正确的参数！dataname 参数只能为 train、valid、test；scale 参数只能为 True 或 False。')
        if to_numpy:
            feature, target = feature.to_numpy(), target.to_numpy()
        return feature, target
