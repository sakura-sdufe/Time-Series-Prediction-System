# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation
# @Time     : 2025/1/9 20:56
# @Author   : 张浩
# @FileName : data_selection.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression, f_regression, chi2, r_regression


class select_best_feature:
    def __init__(self, method:str, number:int):
        """
        使用 sklearn.feature_selection.SelectKBest 选择特征。
        :param method: 特征选择方法。可选值为 None、'互信息'、'F检验'、'卡方检验'、'相关系数'。
        :param number: 特征选择后特征的数量。
        :return: 特征选择结果，类型为 DataFrame。
        """
        # 特征选择方法
        if method is None:
            self.selector = None
        elif method == '互信息':
            self.selector = SelectKBest(score_func=mutual_info_regression, k=number)
        elif method == 'F检验':
            self.selector = SelectKBest(score_func=f_regression, k=number)
        elif method == '卡方检验':
            self.selector = SelectKBest(score_func=chi2, k=number)
        elif method == '相关系数':
            self.selector = SelectKBest(score_func=r_regression, k=number)
        else:
            raise ValueError("method 参数错误！")

    def fit_transform(self, feature, target):
        if self.selector is None:
            return feature
        else:
            feature_selected = self.selector.fit_transform(feature, target)  # feature_selected 是 ndarray 类型
            # 转为 DataFrame 类型
            feature_selected = pd.DataFrame(feature_selected, columns=feature.columns[self.selector.get_support()])
            return feature_selected
