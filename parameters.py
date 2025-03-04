# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation-Project
# @Time     : 2024/11/12 14:02
# @Author   : 张浩
# @FileName : parameters.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

from data_parameters import DataParameters
from predictor_parameters import PredictorParameters
from project_parameters import ProjectParameters
from ensemble_parameters import EnsembleParameters


# 是否可以考虑使用多个类的继承和多个类的初始化，实现参数的初始化传递。

__all__ = ['DataParameters', 'PredictorParameters', 'EnsembleParameters', 'ProjectParameters']
