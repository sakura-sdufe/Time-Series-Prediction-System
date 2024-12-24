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

import pickle


def save_model(model, save_path):
    """
    保存模型
    :param model: 模型
    :param save_path: 保存路径
    :return: None
    """
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(load_path):
    """
    加载模型
    :param load_path: 加载路径
    :return: 模型
    """
    with open(load_path, 'rb') as f:
        model = pickle.load(f)
    return model
