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

from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor

from .predictor_main import Predictors
from .predictor_utils import save_model, load_model
from .DLModel import RNNModel, LSTMModel, GRUModel
from .DLCriterion import MSELoss_scale, sMAPELoss


tools = ['Predictors', 'save_model', 'load_model']
machine_learning = ['SVR', 'Ridge', 'RandomForestRegressor', 'GradientBoostingRegressor', 'AdaBoostRegressor',
                    'BaggingRegressor']
deep_learning = ['RNNModel', 'LSTMModel', 'GRUModel']
deep_criterion = ['MSELoss_scale', 'sMAPELoss']

__all__ = tools + machine_learning + deep_learning
