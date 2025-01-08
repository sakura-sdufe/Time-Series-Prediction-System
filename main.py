# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation-Project
# @Time     : 2024/11/10 11:37
# @Author   : 张浩
# @FileName : preprocessing_main.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import os
import pandas as pd
from matplotlib import rcParams
from sklearn.feature_selection import SelectKBest, mutual_info_regression

from display import Writer
from preprocessing import read_validation_file
from data import Norm, DataSplit, SeqLoader
from predictor import SVR, Ridge, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from predictor import Predictors

from parameters import DataParameters, PredictorParameters

# 设置中文字体为 SimHei（黑体）
rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
rcParams['axes.unicode_minus'] = False   # 用来正常显示负号


if __name__ == '__main__':
    # 获取超参数
    parameters_data = DataParameters()  # 获取 DataSplit 模块的超参数
    parameters_predictor = PredictorParameters()  # 获取 Predictors 模块的超参数
    # 使用 Writer 类保存所有配置参数（最前面的 Writer 类，如果想删除之前的结果后再保存新的结果，可以在这里将 is_delete 设置为 True）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    writer_dir = os.path.join(current_dir, parameters_predictor["save_dir"].strip(r'./\\'))
    writer = Writer(writer_dir, is_delete=False)
    writer.add_parameters('data parameters', parameters_data, key='config parameters')  # 保存数据参数
    writer.add_parameters('predictor parameters', parameters_predictor, key='config parameters')  # 保存预测器参数
    writer.write('a+')

    # 读取预处理
    # preprocessing_main()  # 为指定文件进行预处理
    data = read_validation_file('VestasV52 convert gap=60 part2.csv', 'VestasV52 convert gap=60 hash.txt')
    # 获取特征和目标
    feature = data[list(parameters_data["feature"])]  # 获取特征（DataFrame）
    target = data[parameters_data["target"]]  # 获取目标（Series）

    # 特征选择：SelectKBest(score_func=mutual_info_regression)
    selector = SelectKBest(score_func=mutual_info_regression, k=parameters_data["feature_number"])
    feature_selected = selector.fit_transform(feature, target)  # feature_selected 是 ndarray 类型
    feature_selected = pd.DataFrame(feature_selected, columns=feature.columns[selector.get_support()])  # 转为 DataFrame 类型
    "未被选择的特征：Index(['NacelTemp', 'GenBearTemp', 'WindDirAbs', 'EnvirTemp']"
    "feature_selected 是经过特征选择后的特征；feature 是原始特征"

    # 数据集标准化、数据集封装、数据集划分
    data_split = DataSplit(feature_selected, target, parameters_data, normalization=Norm)
    seq_loader = SeqLoader(feature_selected, target, parameters_data, normalization=Norm)
    """
    train_feature, train_target = data_split.get_dataset(dataname='train', to_numpy=True)
    valid_feature, valid_target = data_split.get_dataset(dataname='valid', to_numpy=True)
    test_feature, test_target = data_split.get_dataset(dataname='test', to_numpy=True)
    train_loader, valid_loader, test_loader = seq_loader.get_all_loader()
    """

    # 选择预测器
    models_cls = [SVR, Ridge, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor]
    is_normalization = [True, True, False, False, False, False]  # 是否需要标准化
    predictors = Predictors(data_split, save_relpath=parameters_predictor['save_dir'], y_label='风电值')
    models_trained = predictors.all_models(models_cls, parameters_predictor, is_normalization=is_normalization)



