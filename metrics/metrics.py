# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation-Project
# @Time     : 2024/11/17 17:38
# @Author   : 张浩
# @FileName : metrics.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score


def _convert_to_numpy(data):
    if isinstance(data, pd.Series):
        data = data.to_numpy()
    elif isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("DataFrame 只能有一列！")
        data = data.to_numpy().reshape(-1)
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            pass
        elif (data.ndim == 2) and (data.shape[0] == 1 or data.shape[1] == 1):
            data = data.reshape(-1)
        else:
            raise ValueError("numpy.ndarray 只能为一列或者一行！")
    elif isinstance(data, list):
        data = np.array(data)
        if data.ndim != 1:
            raise ValueError("list 必须是一维的！")
    else:
        raise ValueError("转换类型失败！")
    return data


def RMSE(true_value, predict_value):
    return np.sqrt(mean_squared_error(true_value, predict_value))


def sMAPE(true_value, predict_value):
    epsilon = 1e-6  # 防止分母为0
    Numerator = np.abs(true_value - predict_value)
    Denominator = (np.abs(true_value) + np.abs(predict_value)) / 2 + epsilon
    return np.mean(Numerator / Denominator)


def calculate_metrics(true_value, predict_value, metrics=None):
    """
    计算评价指标。
    :param true_value: 真实值。支持 Series，DataFrame，numpy.ndarray，list。
        DataFrame 的只能有一列，numpy.ndarray 只能为一列或者一行，list 必须是一维的。
    :param predict_value: 预测值。支持 Series，DataFrame，numpy.ndarray，list。
        DataFrame 的只能有一列，numpy.ndarray 只能为一列或者一行，list 必须是一维的。
    :param metrics: 评价指标。支持 "sMAPE", "MAPE", "RMSE", "MSE", "MAE", "R2"
    :return: 以字典的形式返回评价指标值。
    """
    if metrics is None:
        metrics = ["sMAPE", "RMSE", "MAE", "R2"]
    true_value = _convert_to_numpy(true_value)
    predict_value = _convert_to_numpy(predict_value)

    metrics_function = {
        "sMAPE": sMAPE,
        "MAPE": mean_absolute_percentage_error,
        "RMSE": RMSE,
        "MSE": mean_squared_error,
        "MAE": mean_absolute_error,
        "R2": r2_score,
    }

    metrics_result = {}
    for metric in metrics:
        if metric in metrics_function.keys():
            metrics_result[metric] = metrics_function[metric](true_value, predict_value)
        else:
            raise ValueError(f"不支持的评价指标：{metric}！")

    return metrics_result
