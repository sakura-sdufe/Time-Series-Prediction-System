# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation
# @Time     : 2025/2/26 11:14
# @Author   : 张浩
# @FileName : ensemble_parameters.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

class EnsembleParameters:
    def __init__(self):
        """
        1. 类内变量名需与模型类名保持一致，否则无法正常解析。
        2. 深度学习模型参数无需传入 input_size 和 time_step，会自动解析。
        """
        # AttentionEnsemble 模型参数
        self.AttentionEnsemble = {
            'output_size': 1,
            'dropout': 0.0,
            'bias': True,
            'activation': 'relu',
        }
        # AttentionProEnsemble 模型参数
        self.AttentionProEnsemble = {
            'output_size': 1,
            'project_size': 16,
            'feedforward': 32,
            'dropout': 0.1,
            'bias': True,
            'activation': 'relu',
        }
        # C2L 模型参数
        self.C2L = {
            'output_size': 1,
            'activation': 'relu',
        }
        # C3B2H 模型参数
        self.C3B2H = {
            'output_size': 1,
            'bias': True,
            'dropout': 0.0,
            'activation': 'relu',
        }
        # 深度学习训练参数
        self.DL_train = {
            'epochs': 300,  # 训练轮数，默认为 300。
            'learning_rate': 1e-2,  # 学习率，默认为 1e-3。
            'weight_decay': 1e-4,  # 权重衰减，默认为 1e-4。
            'clip_norm': 0.5,  # 梯度裁剪阈值，默认为 None，表示不裁剪。
            'ReduceLROnPlateau_factor': 0.3,  # 学习率衰减因子，默认为 0.5。
            'ReduceLROnPlateau_patience': 30,  # 监测器函数不再减小的累计次数，默认为 30。
            'ReduceLROnPlateau_threshold': 1e-4,  # 只关注超过阈值的显著变化，默认为 1e-4。
        }

    def __getitem__(self, item):
        return getattr(self, item)

    def __iter__(self):
        return iter(self.__dict__.items())

    def items(self):
        return self.__dict__.items()
