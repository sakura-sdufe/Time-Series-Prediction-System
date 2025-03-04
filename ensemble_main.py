# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation
# @Time     : 2025/2/27 12:34
# @Author   : 张浩
# @FileName : ensemble_main.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import os
import traceback
import pandas as pd
from torch.nn import MSELoss

from utils import Writer
from data import Norm, EnsembleSplit, EnsembleLoader

from ensemble import AttentionEnsemble, AttentionProEnsemble, C3B2H
from ensemble import Ensembles
from DLCriterion import MSELoss_scale, MSELoss_sqrt, sMAPELoss, MAPELoss
from metrics import calculate_metrics

from parameters import DataParameters, EnsembleParameters, ProjectParameters


if __name__ == '__main__':
    # 获取超参数
    parameters_project = ProjectParameters()  # 项目相关超参数
    parameters_data = DataParameters()  # 数据相关超参数
    parameters_ensemble = EnsembleParameters()  # 集成相关超参数

    # 获取绝对路径，实例化 Writer 类。
    current_dir = os.path.dirname(os.path.abspath(__file__))
    predictor_dir = os.path.join(current_dir, parameters_project["save_predictor_dir"].strip(r'./\\'))
    save_dir = os.path.join(current_dir, parameters_project["save_ensemble_dir"].strip(r'./\\'))
    writer = Writer(save_dir, is_delete=parameters_project['delete_dir'])  # 实例化 Writer 类，用于保存结果
    # 保存项目参数、数据参数、预测器参数
    writer.add_param('project parameters', parameters_project, filename='config parameters', folder='documents')
    writer.add_param('data parameters', parameters_data, filename='config parameters', folder='documents')
    writer.add_param('ensemble parameters', parameters_ensemble, filename='config parameters', folder='documents',
                     save_mode='a+')

    try:
        # 数据集标准化、数据集封装、数据集划分
        ens_split = EnsembleSplit(predictor_dir, normalization=Norm)
        ens_loader = EnsembleLoader(predictor_dir, batch_size=parameters_data['ensemble_batch_size'], normalization=Norm)
        # train_dataloader_norm, train_eval_dataloader_norm, valid_dataloader_norm, test_dataloader_norm = ens_loader.get_all_loader()
        # Ensembles
        ensembles = Ensembles(ens_split, ens_loader, writer=writer)

        # SINGLE MODEL
        transformer_trained = ensembles.model(
            AttentionProEnsemble, parameters_ensemble['AttentionProEnsemble'], is_normalization=False,
            criterion=MSELoss_scale, monitor=sMAPELoss, train_parameters=parameters_ensemble['DL_train']
        )
        # attention_trained = ensembles.model(
        #     AttentionProEnsemble, parameters_ensemble['AttentionProEnsemble'], is_normalization=False,
        #     criterion=MSELoss_scale, monitor=sMAPELoss, train_parameters=parameters_ensemble['DL_train']
        # )

        # # ALL MODELS
        # ML_cls = []
        # ML_normalization = []
        # DL_cls = [AttentionProEnsemble, AttentionEnsemble, C3B2H]
        # DL_normalization = [False, False, False]
        #
        # models_cls = ML_cls + DL_cls
        # model_normalization = ML_normalization + DL_normalization
        # models_trained = ensembles.all_models(models_cls, parameters_ensemble, is_normalization=model_normalization,
        #                                       criterion=MSELoss_scale, monitor=sMAPELoss)
        #
        # # Save the temporary storage results
        # writer.write(parameters_project['save_mode'])

    except (Exception, KeyboardInterrupt) as e:
        error_message = traceback.format_exc()
        writer.add_text(f"", filename="Logs", folder="documents", suffix="log", save_mode='a+')
        writer.add_text(f"Error: {error_message}", filename="Logs", folder="documents", suffix="log", save_mode='a+')
        writer.write(parameters_project['save_mode'])
        raise e
