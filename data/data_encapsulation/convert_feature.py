# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation-Project
# @Time     : 2024/11/14 19:07
# @Author   : 张浩
# @FileName : convert_feature.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import pandas as pd


def convert_feature(feature: pd.DataFrame, target: pd.Series, time_unknown_variables, time_known_variables,
                    history_steps, is_features_history=False):
    """
    将 DataFrame 中的每日数据按照时变已知变量、时变未知变量、目标变量分别处理，返回可以用于时间序列的特征数据。
    :param feature: 每日的特征数据（其他影响预测数据的变量），接受 DataFrame 类型数据。
    :param target: 每日的目标数据（需要预测的数据），接受 Series 类型数据。
    :param time_unknown_variables: 时变未知变量，接受 list 和 tuple 类型（内部的每个元素是字符串，表示 DataFrame 的列名）。
    :param time_known_variables: 时变已知变量，接受 list 和 tuple 类型（内部的每个元素是字符串，表示 DataFrame 的列名）。
    :param history_steps: 使用前多少天的数据进行预测。
    :param is_features_history: 是否使用 feature 中的历史数据作为特征，默认为 False。
        如果使用特征历史数据，那么将会根据历史时间步为每个时变未知变量、时变已知变量、目标变量分别添加历史数据（其中时变未知变量将会创建
        history_steps 个列，时变已知变量将会创建 history_steps+1 个列）；
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
        for i in range(1, history_steps+1):
            feature_shift = feature[time_unknown_variables].shift(i)
            feature_shift.columns = [f"{col}_{i}" for col in feature_shift.columns]
            feature_unknown = pd.concat([feature_unknown, feature_shift], axis=1)
    else:
        feature_unknown = feature[time_unknown_variables].shift(1)
    feature_unknown = feature_unknown[history_steps:]  # 去掉前 history_steps 行（对齐）
    feature_unknown.reset_index(drop=True, inplace=True)
    time_series_feature = pd.concat([time_series_feature, feature_unknown], axis=1)

    # Step 3：处理时变已知变量（需要判断是否加入历史数据）
    feature_known = pd.DataFrame()  # 用于存储时变已知变量的历史数据
    if is_features_history:
        for i in range(history_steps+1):
            feature_shift = feature[time_known_variables].shift(i)
            feature_shift.columns = [f"{col}_{i}" for col in feature_shift.columns]
            feature_known = pd.concat([feature_known, feature_shift], axis=1)
    else:
        feature_known = feature[time_known_variables]
    feature_known = feature_known[history_steps:]  # 去掉前 history_steps 行（对齐）
    feature_known.reset_index(drop=True, inplace=True)
    time_series_feature = pd.concat([time_series_feature, feature_known], axis=1)

    # Step 4：处理目标变量，加入目标变量的历史数据
    for i in range(1, history_steps+1):
        target_shift = target.shift(i)
        target_shift.name = f"{target.name}_{i}"
        target_shift = target_shift[history_steps:]  # 去掉前 history_steps 行（这些行包括 NaN 值）
        target_shift.reset_index(drop=True, inplace=True)
        time_series_feature = pd.concat([time_series_feature, target_shift], axis=1)

    # Step 5：处理目标
    time_series_target = target[history_steps:]  # 去掉前 history_steps 行（对齐）

    assert len(time_series_feature) == len(time_series_target), "特征数据和目标数据的长度不一致。【请检查该函数内部】"
    return time_series_feature, time_series_target
