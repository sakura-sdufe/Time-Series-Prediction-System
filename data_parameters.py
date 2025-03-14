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
        self.time_step = 10  # 时间步长（使用多少天的数据进行预测）

        # 特征选择
        self.feature_selection = {
            'method': '互信息',  # 特征选择方法。可选值为 None、'互信息'、'F检验'、'卡方检验'、'相关系数'。
            'number': 15  # 特征选择后特征的数量。
        }

        "self.time_unknown_variables 和 self.time_known_variables 可以在特征中不存在，但是同一个变量不能同时位于这两个列表中"
        self.target = 'Power'  # 目标变量，即要预测的变量
        self.time_unknown_variables = ['MaxPower', 'MinPower', 'StdDevPower', 'AvgRPow', 'Pitch', 'GenRPM',
                                       'RotorRPM', 'NacelTemp', 'GearOilTemp', 'GearBearTemp', 'GenPh1Temp',
                                       'GenPh2Temp', 'GenPh3Temp', 'GenBearTemp']  # 时变未知变量
        self.time_known_variables = ['WindSpeed', 'StdDevWindSpeed', 'WindDirAbs', 'WindDirRel', 'EnvirTemp']  # 时变已知变量
        self.feature = self.time_unknown_variables + self.time_known_variables  # 从源文件中选择参与回归的变量
        self.selected_feature = self.feature  # 特征选择后的变量

        # 数据集划分比例。
        """
        - 当 start_position 和 end_position 取值为 0.0 和 1.0 时，表示按照比例切分（输入为 float 类型数据按比例切分）；
        - 当 start_position 和 end_position 取值为 0 和 1 时，表示按照数量切分（输入为 int 类型数据按数量切分）。
        """
        self.train_start_rate = 0.0
        self.train_end_rate = 0.9
        self.valid_start_rate = 0.9
        self.valid_end_rate = 1.0
        self.test_start_rate = 0.9
        self.test_end_rate = 1.0

        # 回归模型专属参数
        self.is_features_history = True  # 是否使用特征历史数据

        # 深度学习专属参数
        """
        - sample_gap 的含义是这个样本和上个样本之间的时间间隔，如果 sample_gap=1，表示连续采样。例如：
        当前样本特征时间为 [0, 1, 2]，目标时间为 [3]；那么下一个样本的时间特征为 [0+sample_gap, 1+sample_gap, 2+sample_gap]，
        目标时间为 [3+sample_gap]。如果是推理过程，这个值应当为 1；如果是训练过程，为了防止过拟合，可以设置为大于 1 的值，但不宜超过时间步。
        """
        self.sample_gap = 1  # 采样间隔，数据类型为 int。默认为 1，表示连续采样。该参数只在训练集上生效，在验证集和测试集上为 1。
        self.dataloader_shuffle = True  # 是否在每个 epoch 开始时打乱数据集的顺序。该参数只在训练集上生效，在验证集和测试集上为 False。
        self.predictor_batch_size = 256  # 预测模型的训练集批量大小
        self.ensemble_batch_size = 256  # 集成模型的训练集批量大小
        self.eval_batch_size = 1024  # 评估模式下训练集、验证集、测试集批量大小（较大的批可以拥有更快的推理速度）

    def __getitem__(self, item):
        return getattr(self, item)

    def __iter__(self):
        return iter(self.__dict__.items())

    def items(self):
        return self.__dict__.items()

    def add(self, key, value):
        self.__dict__[key] = value
