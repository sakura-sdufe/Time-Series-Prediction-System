# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation-Project
# @Time     : 2024/11/11 21:49
# @Author   : 张浩
# @FileName : parameters_data.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

class DataParameters:
    def __init__(self):
        # 数据基础信息
        self.history_steps = 10  # 历史步长（使用前多少天的数据进行预测）
        self.is_features_history = False  # 是否使用特征历史数据（该变量仅用于类似 SVR 之类的回归模型，深度学习模型将会使用所有特征的历史数据）

        # 特征选择
        self.feature_number = 15  # 特征选择后特征的数量

        self.target = 'Power'  # 目标变量，即要预测的变量
        self.time_unknown_variables = ['MaxPower', 'MinPower', 'StdDevPower', 'AvgRPow', 'Pitch', 'GenRPM',
                                       'RotorRPM', 'NacelTemp', 'GearOilTemp', 'GearBearTemp', 'GenPh1Temp',
                                       'GenPh2Temp', 'GenPh3Temp', 'GenBearTemp']  # 时变未知变量
        self.time_known_variables = ['WindSpeed', 'StdDevWindSpeed', 'WindDirAbs', 'WindDirRel', 'EnvirTemp']  # 时变已知变量
        self.feature = self.time_unknown_variables + self.time_known_variables  # 特征变量

        # 数据集划分比例。
        self.train_start_rate = 0.0
        self.train_end_rate = 0.9
        self.valid_start_rate = 0.9
        self.valid_end_rate = 1.0
        self.test_start_rate = 0.9
        self.test_end_rate = 1.0

    def __getitem__(self, item):
        return getattr(self, item)
