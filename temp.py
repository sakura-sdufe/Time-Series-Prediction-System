# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation
# @Time     : 2025/1/18 14:25
# @Author   : 张浩
# @FileName : temp.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import os
import pandas as pd
from torch.nn import MSELoss

from utils import Writer
from data import SeqLoader

from predictor import Predictors
from DLCriterion import MSELoss_scale, sMAPELoss, MAPELoss


def test(target, *args, feature=None):
    print('target_ndarray:', target)
    print('args:', args)
    print('predict:', feature)

test('target_ndarray', 'args1', 'args2', feature='predict')
