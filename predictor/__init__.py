# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation-Project
# @Time     : 2024/11/13 13:42
# @Author   : 张浩
# @FileName : __init__.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

from .predictor_main import Predictors
from .predictor_utils import save_model, load_model  # 机器学习模型读取与保存（.pkl）

from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor

from .DLModel.Base import ModelBase
from .DLModel.RNNs import RNNModel, LSTMModel, GRUModel
from .DLModel.Transformers import TransformerWithLinear, TransformerWithAttention

from .DLCriterion import MSELoss_scale, sMAPELoss  # 损失函数、监测函数


tools = ['Predictors', 'save_model', 'load_model']
machine_learning = ['SVR', 'Ridge', 'RandomForestRegressor', 'GradientBoostingRegressor', 'AdaBoostRegressor',
                    'BaggingRegressor']
deep_learning = ['ModelBase', 'RNNModel', 'LSTMModel', 'GRUModel', 'TransformerWithLinear', 'TransformerWithAttention']
deep_criterion = ['MSELoss_scale', 'sMAPELoss']

__all__ = tools + machine_learning + deep_learning + deep_criterion
