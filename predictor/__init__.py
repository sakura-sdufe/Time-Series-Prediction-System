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

from .predictor import Predictors

from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor

from .DLModel.MLP import MLPModel
from .DLModel.RNNs import RNNModel, LSTMModel, GRUModel
from .DLModel.Transformers import TransformerWithLinear, TransformerWithAttention

tools = ['Predictors', ]
machine_learning = ['SVR', 'Ridge', 'RandomForestRegressor', 'GradientBoostingRegressor', 'AdaBoostRegressor',
                    'BaggingRegressor']
deep_learning = ['ModelBase', 'MLPModel', 'RNNModel', 'LSTMModel', 'GRUModel',
                 'TransformerWithLinear', 'TransformerWithAttention']

__all__ = tools + machine_learning + deep_learning
