# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation
# @Time     : 2025/1/22 20:53
# @Author   : 张浩
# @FileName : __init__.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

from .ensemble import Ensembles

from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor

from .DLModel.Attentions import AttentionEnsemble, AttentionProEnsemble
from .DLModel.CNNs import C3B2H


tools = ['Ensembles', ]
machine_learning = ['SVR', 'Ridge', 'RandomForestRegressor', 'GradientBoostingRegressor', 'AdaBoostRegressor',
                    'BaggingRegressor']
deep_learning = ['AttentionEnsemble', 'AttentionProEnsemble', 'C3B2H']

__all__ = tools + machine_learning + deep_learning
