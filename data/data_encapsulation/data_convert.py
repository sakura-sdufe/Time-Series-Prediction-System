# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation-Project
# @Time     : 2024/11/17 19:34
# @Author   : 张浩
# @FileName : data_convert.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import warnings
import pandas as pd


def split_dataset(feature, target, start_position, end_position):
    """
    将 DataFrame 或者 array 类型数据进行切分（按照 dim=0 维度切分），切分的位置由 start_position 和 end_position 控制。
    :param feature: 特征数据。数据类型为 DataFrame 或者 array。
    :param target: 目标数据。数据类型为 DataFrame 或者 array。
    :param start_position: 切分的起始位置。如果是介于 0 和 1 之间的小数，则表示按照比例切分。如果是大于 1 的整数，则表示按照数量切分。
    :param end_position: 切分的结束位置。如果是介于 0 和 1 之间的小数，则表示按照比例切分。如果是大于 1 的整数，则表示按照数量切分。
        Note: feature 和 target 也可以接受其他可以被 len 函数调用和可以切片的数据类型。
            当 start_position 和 end_position 取值为 0.0 和 1.0 时，表示按照比例切分（输入为 float 类型数据按比例切分）；
            当 start_position 和 end_position 取值为 0 和 1 时，表示按照数量切分（输入为 int 类型数据按数量切分）。
    :return: feature_split, target_split 分别表示指定范围内的特征数据和目标数据。
    """
    assert type(start_position) == type(end_position), "start_position 和 end_position 的数据类型必须相同！"
    assert (start_position>=0) and (end_position>=0), "start_position 和 end_position 的取值必须大于等于 0！"
    assert start_position < end_position, "start_position 的取值必须小于 end_position！"
    if len(feature) != len(target):
        warnings.warn(
            "start_position 和 end_position 在 dim=0 维度上长度不相同；如果你使用的时按比例划分，那么使用 feature 的长度。",
            category=UserWarning)

    if isinstance(start_position, float):  # 按照比例切分
        if (start_position<=1) and (end_position<=1):
            start_position = round(start_position * len(feature))
            end_position = round(end_position * len(feature))
        else:
            raise ValueError("当 start_position 和 end_position 按照比例划分时，取值必须在 0 和 1 之间！")

    assert end_position <= len(feature), "end_position 的取值不能大于特征数据的长度！"
    return feature[start_position:end_position], target[start_position:end_position]


def convert_feature(feature: pd.DataFrame, target: pd.Series, time_unknown_variables, time_known_variables,
                    time_step, is_features_history=False):
    """
    将 DataFrame 中的每日数据按照时变已知变量、时变未知变量、目标变量分别处理，返回可以用于时间序列的特征数据。
    :param feature: 每日的特征数据（其他影响预测数据的变量），接受 DataFrame 类型数据。
    :param target: 每日的目标数据（需要预测的数据），接受 Series 类型数据。
    :param time_unknown_variables: 时变未知变量，接受 list 和 tuple 类型（内部的每个元素是字符串，表示 DataFrame 的列名）。
    :param time_known_variables: 时变已知变量，接受 list 和 tuple 类型（内部的每个元素是字符串，表示 DataFrame 的列名）。
    :param time_step: 使用前多少天的数据进行预测。
    :param is_features_history: 是否使用 feature 中的历史数据作为特征，默认为 False。
        如果使用特征历史数据，那么将会根据历史时间步为每个时变未知变量、时变已知变量、目标变量分别添加历史数据（其中时变未知变量将会创建
        time_step 个列，时变已知变量将会创建 time_step+1 个列）；
        如果不使用特征历史数据，那么只会添加最新观测的数据（时变已知变量用当前时间步表示，时变未知变量用上一时间步表示）。

    :return: 返回 DataFrame 类型处理后的时间序列特征数据，其中包含时变未知变量、时变已知变量、目标变量历史值。

    Note:
        1. 如果 feature 中存在 time_unknown_variables 和 time_known_variables 未包含的变量，那么那些变量将不会加入到返回的时间序列特征中。
        2. 如果 is_features_history 为 True，那么返回的时间序列特征将会对列重新命名，列的命名格式为：'变量名_i' 表示前 i 天的数据，
            '变量名_0' 表示当前时间步的数据（这只会存在于时变已知变量中）。
        3. 该函数将会返回 DataFrame 类型数据，一般适用于时间序列模型和部分机器学习回归模型，并不适合循环神经网络模型。
    """
    # Step 1.1：创建一个新的 DataFrame 用于存储时间序列特征数据
    time_series_feature = pd.DataFrame()

    # Step 1.2：将 time_unknown_variables 和 time_known_variables 存在，但 feature 中不存在的变量删除。
    time_unknown_variables = [col for col in time_unknown_variables if col in feature.columns]
    time_known_variables = [col for col in time_known_variables if col in feature.columns]

    # Step 2：处理时变未知变量（需要判断是否加入历史数据）
    feature_unknown = pd.DataFrame()  # 用于存储时变未知变量的历史数据
    if is_features_history:
        for i in range(1, time_step+1):
            feature_shift = feature[time_unknown_variables].shift(i)
            feature_shift.columns = [f"{col}_{i}" for col in feature_shift.columns]
            feature_unknown = pd.concat([feature_unknown, feature_shift], axis=1)
    else:
        feature_unknown = feature[time_unknown_variables].shift(1)
    feature_unknown = feature_unknown[time_step:]  # 去掉前 time_step 行（对齐）
    feature_unknown.reset_index(drop=True, inplace=True)
    time_series_feature = pd.concat([time_series_feature, feature_unknown], axis=1)

    # Step 3：处理时变已知变量（需要判断是否加入历史数据）
    feature_known = pd.DataFrame()  # 用于存储时变已知变量的历史数据
    if is_features_history:
        for i in range(time_step):
            feature_shift = feature[time_known_variables].shift(i)
            feature_shift.columns = [f"{col}_{i}" for col in feature_shift.columns]
            feature_known = pd.concat([feature_known, feature_shift], axis=1)
    else:
        feature_known = feature[time_known_variables]
    feature_known = feature_known[time_step:]  # 去掉前 time_step 行（对齐）
    feature_known.reset_index(drop=True, inplace=True)
    time_series_feature = pd.concat([time_series_feature, feature_known], axis=1)

    # Step 4：处理目标变量，加入目标变量的历史数据
    for i in range(1, time_step+1):
        target_shift = target.shift(i)
        target_shift.name = f"{target.name}_{i}"
        target_shift = target_shift[time_step:]  # 去掉前 time_step 行（这些行包括 NaN 值）
        target_shift.reset_index(drop=True, inplace=True)
        time_series_feature = pd.concat([time_series_feature, target_shift], axis=1)

    # Step 5：处理目标
    time_series_target = target[time_step:]  # 去掉前 time_step 行（对齐）

    assert len(time_series_feature) == len(time_series_target), "特征数据和目标数据的长度不一致。【请检查该函数内部】"
    return time_series_feature, time_series_target


class DataSplit:
    def __init__(self, time_feature: pd.DataFrame, time_target: pd.Series, parameters_data, normalization=None):
        """
        为时间序列数据添加历史特征，分割数据集，标准化数据集，转化成 ndarray 数据集。
        :param time_feature: 时间特征数据，数据类型为 DataFrame。
        :param time_target: 时间目标数据，数据类型为 Series。
        :param parameters_data: 数据参数，要求具有 __getitem__ 方法。
        :param normalization: 标准化类，需要需要包含 get_norm_result, norm, denorm 方法。如果标准化，那么需要传入标准化类。
        """
        self.time_feature, self.time_target = time_feature, time_target  # 时间特征和目标
        self.parameters_data = parameters_data  # DataParameters 类
        self.normalization = normalization  # 标准化类

        self.feature, self.target = [None] * 2  # 存放添加历史信息的特征总表和目标总表
        self.train_feature, self.valid_feature, self.test_feature = [None] * 3  # 存放分割后的特征，数据类型为 DataFrame
        self.train_target, self.valid_target, self.test_target = [None] * 3  # 存放分割后的目标，数据类型为 Series

        if self.normalization:
            self.normalization_feature = normalization(time_feature)  # 对特征进行标准化
            self.normalization_target = normalization(time_target)  # 对目标进行标准化
            self.time_feature_norm = self.normalization_feature.get_norm_result()  # 获取标准化后的特征
            self.time_target_norm = self.normalization_target.get_norm_result()  # 获取标准化后的目标
            self.feature_norm, self.target_norm = [None] * 2  # 存放标准化后的特征总表和目标总表
            self.train_feature_norm, self.valid_feature_norm, self.test_feature_norm = [None] * 3  # 存放标准化后分割的特征
            self.train_target_norm, self.valid_target_norm, self.test_target_norm = [None] * 3  # 存放标准化后分割的目标

        self._add_history_feature()  # 添加历史特征
        self._split_dataset()  # 分割数据集

    def _add_history_feature(self):
        """添加历史特征。"""
        # 没有标准化的特征和目标
        self.feature, self.target = convert_feature(
            self.time_feature, self.time_target,
            time_unknown_variables = self.parameters_data["time_unknown_variables"],
            time_known_variables = self.parameters_data["time_known_variables"],
            time_step = self.parameters_data["time_step"],
            is_features_history = self.parameters_data["is_features_history"]
        )
        if self.normalization:
            self.feature_norm, self.target_norm = convert_feature(
                self.time_feature_norm, self.time_target_norm,
                time_unknown_variables = self.parameters_data["time_unknown_variables"],
                time_known_variables = self.parameters_data["time_known_variables"],
                time_step = self.parameters_data["time_step"],
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
        if self.normalization:
            self.train_feature_norm, self.train_target_norm = split_dataset(
                self.feature_norm, self.target_norm,
                start_position=self.parameters_data["train_start_rate"],
                end_position=self.parameters_data["train_end_rate"]
            )
            self.valid_feature_norm, self.valid_target_norm = split_dataset(
                self.feature_norm, self.target_norm,
                start_position=self.parameters_data["valid_start_rate"],
                end_position=self.parameters_data["valid_end_rate"]
            )
            self.test_feature_norm, self.test_target_norm = split_dataset(
                self.feature_norm, self.target_norm,
                start_position=self.parameters_data["test_start_rate"],
                end_position=self.parameters_data["test_end_rate"]
            )

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
        elif (scale is None) and (not self.normalization):
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
            raise ValueError('请输入正确的参数！dataname 参数只能为 train、valid、test。')
