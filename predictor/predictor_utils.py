# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation-Project
# @Time     : 2024/11/20 15:08
# @Author   : 张浩
# @FileName : predictor_utils.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import torch
from sklearn import base
import pickle


def save_model(model, save_path):
    """
    保存模型
    :param model: 模型
    :param save_path: 保存路径
    :return: None
    """
    if isinstance(model, base.BaseEstimator) and save_path.endswith('.pkl'):
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
    elif isinstance(model, torch.nn.Module) and save_path.endswith('.pth'):
        torch.save(model, save_path)
    else:
        raise ValueError('请检查模型与路径是否匹配。')


def load_model(load_path):
    """
    加载模型
    :param load_path: 加载路径
    :return: 模型
    """
    if load_path.endswith('.pkl'):
        with open(load_path, 'rb') as f:
            model = pickle.load(f)
    elif load_path.endswith('.pth'):
        model = torch.load(load_path)
    else:
        raise ValueError('请检查路径是否正确。')
    return model
