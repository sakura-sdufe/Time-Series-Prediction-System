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
import traceback
import pandas as pd
from torch.nn import MSELoss

from utils import Writer
from preprocessing import read_validation_file
from data import select_best_feature, Norm, SeqSplit, SeqLoader, EnsembleSplit, EnsembleLoader

from predictor import SVR, Ridge, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from predictor import MLPModel, RNNModel, LSTMModel, GRUModel, TransformerWithLinear, TransformerWithAttention
from predictor import Predictors
from DLCriterion import MSELoss_scale, sMAPELoss, MAPELoss

from parameters import DataParameters, PredictorParameters, ProjectParameters


if __name__ == '__main__':
    # 获取超参数
    parameters_project = ProjectParameters()  # 项目相关超参数
    parameters_data = DataParameters()  # 数据相关超参数
    parameters_predictor = PredictorParameters()  # 模型相关超参数

    # 获取绝对路径，实例化 Writer 类。
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, parameters_project["save_predictor_dir"].strip(r'./\\'))
    writer = Writer(save_dir, is_delete=parameters_project['delete_dir'])  # 实例化 Writer 类，用于保存结果
    # 保存项目参数、数据参数、预测器参数
    writer.add_param('project parameters', parameters_project, filename='config parameters', folder='documents')
    writer.add_param('data parameters', parameters_data, filename='config parameters', folder='documents')
    writer.add_param('predictor parameters', parameters_predictor, filename='config parameters', folder='documents',
                     save_mode='a+')

    try:
        # 读取预处理
        # preprocessing_main()  # 为指定文件进行预处理
        data = read_validation_file('VestasV52 convert gap=60 part2.csv',
                                    'VestasV52 convert gap=60 hash.txt')
        # 获取特征（DataFrame）和目标（Series）
        feature, target = data[parameters_data["feature"]], data[parameters_data["target"]]
        # 特征选择
        selector = select_best_feature(**parameters_data["feature_selection"])
        feature_selected = selector.fit_transform(feature, target)  # feature_selected 是 DataFrame 类型（包含列名）
        writer.add_text(f"特征选择结果：{feature_selected.columns.tolist()}。\n" + "-" * 100 + "\n",
                        filename="Logs", folder="documents", suffix="log", save_mode='a+')  # 保存特征选择结果到日志中

        # 数据集标准化、数据集封装、数据集划分
        seq_split = SeqSplit(feature_selected, target, parameters_data, normalization=Norm)
        seq_loader = SeqLoader(feature_selected, target, parameters_data, normalization=Norm)
        # train_dataloader_norm, train_eval_dataloader_norm, valid_dataloader_norm, test_dataloader_norm = seq_loader.get_all_loader()

        # Predictors
        predictors = Predictors(seq_split, seq_loader, writer=writer)

        # # SINGLE MODEL
        # predictors.persistence()
        # rf_trained = predictors.model(RandomForestRegressor, parameters_ensemble['RandomForestRegressor'], is_normalization=True)
        # svr_trained = predictors.model(SVR, parameters_ensemble['SVR'], is_normalization=True)
        # transformer_linear_trained = predictors.model(
        #     TransformerWithLinear, parameters_ensemble['TransformerWithLinear'], is_normalization=True,
        #     criterion=MSELoss_scale, monitor=sMAPELoss, train_parameters=parameters_ensemble['DL_train']
        # )
        # # transformer_attention_trained = predictors.model(
        #     TransformerWithAttention, parameters_ensemble['TransformerWithAttention'], is_normalization=True,
        #     criterion=MSELoss_scale, monitor=sMAPELoss, train_parameters=parameters_ensemble['DL_train']
        # )

        # ALL MODELS
        ML_cls = [SVR, Ridge, GradientBoostingRegressor, AdaBoostRegressor]
        ML_normalization = [True, True, False, False]
        DL_cls = [RNNModel, LSTMModel, GRUModel, TransformerWithLinear, TransformerWithAttention]
        DL_normalization = [True, True, True, True, True]

        models_cls = ML_cls + DL_cls
        model_normalization = ML_normalization + DL_normalization
        models_trained = predictors.all_models(models_cls, parameters_predictor, is_normalization=model_normalization,
                                               criterion=MSELoss_scale, monitor=MAPELoss)

        # Save the temporary storage results
        writer.write(parameters_project['save_mode'])

    except Exception as e:
        error_message = traceback.format_exc()
        writer.add_text(f"", filename="Logs", folder="documents", suffix="log", save_mode='a+')
        writer.add_text(f"Error: {error_message}", filename="Logs", folder="documents", suffix="log", save_mode='a+')
        writer.write(parameters_project['save_mode'])
        raise e
