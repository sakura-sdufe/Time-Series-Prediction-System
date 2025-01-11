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
from matplotlib import rcParams

from utils import Writer
from preprocessing import read_validation_file
from data import select_best_feature, Norm, DataSplit, SeqLoader

from predictor import SVR, Ridge, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from predictor import RNNModel, LSTMModel, GRUModel, TransformerWithLinear, TransformerWithAttention
from predictor import Predictors, MSELoss_scale, sMAPELoss

from parameters import DataParameters, PredictorParameters, ProjectParameters

# 设置中文字体为 SimHei（黑体）
rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
rcParams['axes.unicode_minus'] = False   # 用来正常显示负号


if __name__ == '__main__':
    # 获取超参数
    parameters_project = ProjectParameters()  # 项目相关超参数
    parameters_data = DataParameters()  # 数据相关超参数
    parameters_predictor = PredictorParameters()  # 模型相关超参数

    # 获取绝对路径，实例化 Writer 类。
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, parameters_project["save_dir"].strip(r'./\\'))
    writer = Writer(save_dir, is_delete=parameters_project['delete_dir'])  # 实例化 Writer 类，用于保存结果
    writer.add_parameters('project parameters', parameters_project, key='config parameters')  # 保存项目参数
    writer.add_parameters('data parameters', parameters_data, key='config parameters')  # 保存数据参数
    writer.add_parameters('predictor parameters', parameters_predictor, key='config parameters')  # 保存预测器参数

    # 读取预处理
    # preprocessing_main()  # 为指定文件进行预处理
    data = read_validation_file('VestasV52 convert gap=60 part2.csv', 'VestasV52 convert gap=60 hash.txt')
    # 获取特征（DataFrame）和目标（Series）
    feature, target = data[parameters_data["feature"]], data[parameters_data["target"]]
    # 特征选择
    selector = select_best_feature(**parameters_data["feature_selection"])
    feature_selected = selector.fit_transform(feature, target)  # feature_selected 是 DataFrame 类型（包含列名）
    writer.add_log(f"特征选择结果：{feature_selected.columns.tolist()}。", key='log')  # 保存特征选择结果到日志中

    # 数据集标准化、数据集封装、数据集划分
    data_split = DataSplit(feature_selected, target, parameters_data, normalization=Norm)
    seq_loader = SeqLoader(feature_selected, target, parameters_data, normalization=Norm)
    # train_dataloader_norm, train_eval_dataloader_norm, valid_dataloader_norm, test_dataloader_norm = seq_loader.get_all_loader()

    # 选择预测器
    predictors = Predictors(data_split, seq_loader, writer=writer)

    # svr_trained = predictors.model(SVR, parameters_predictor['SVR'], is_normalization=True)
    transformer_linear_trained = predictors.model(
        TransformerWithLinear, parameters_predictor['TransformerWithLinear'], is_normalization=True,
        criterion=MSELoss_scale, monitor=sMAPELoss, train_parameters=parameters_predictor['DL_train']
    )
    transformer_attention_trained = predictors.model(
        TransformerWithAttention, parameters_predictor['TransformerWithAttention'], is_normalization=True,
        criterion=MSELoss_scale, monitor=sMAPELoss, train_parameters=parameters_predictor['DL_train']
    )

    # ML_cls = [SVR, Ridge, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor]
    # ML_normalization = [True, True, False, False, False, False]
    # DL_cls = [RNNModel, LSTMModel, GRUModel]
    # DL_normalization = [True, True, True]
    # models_cls = ML_cls + DL_cls
    # model_normalization = ML_normalization + DL_normalization
    # models_trained = predictors.all_models(models_cls, parameters_predictor, is_normalization=model_normalization,
    #                                        criterion=MSELoss_scale, monitor=sMAPELoss)

    writer.write(parameters_project['write_mode'])
