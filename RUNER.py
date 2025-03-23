# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation
# @Time     : 2025/3/21 22:14
# @Author   : 张浩
# @FileName : RUNER.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import os
import traceback
from typing import Union, List, Tuple

from utils import Writer, cprint
from data import read_file, Norm, select_best_feature, SeqSplit, SeqLoader, EnsembleSplit, EnsembleLoader
from predictor import Predictors
from ensemble import Ensembles

from parameters import ProjectParameters, DataParameters, PredictorParameters, EnsembleParameters


REGRESSION_MODELS = (
    'SVR', 'Ridge', 'RandomForestRegressor', 'GradientBoostingRegressor', 'AdaBoostRegressor', 'BaggingRegressor'
)
PREDICTOR_MODELS = (
    'MLPModel', 'RNNModel', 'LSTMModel', 'GRUModel', 'TransformerWithLinear', 'TransformerWithAttention'
)
ENSEMBLE_MODELS = (
    'AttentionEnsemble', 'AttentionProjEnsemble', 'C3B2H'
)


class Runer:
    """运行预测器和集成器"""
    def __init__(self):
        # 初始化变量
        self.predictor_writer, self.ensemble_writer = None, None  # Writer 类，用于保存结果
        self.seq_split, self.seq_loader = None, None  # 机器学习数据集和深度学习数据集（预测器）
        self.ens_split, self.ens_loader = None, None  # 机器学习数据集和深度学习数据集（集成器）
        self.predictors, self.ensembles = None, None  # 预测器和集成器
        self.trained_predictors, self.trained_ensembles = [], []  # 训练好的预测器和集成器
        # 获取超参数
        self.parameters_project = ProjectParameters()  # 项目相关超参数
        self.parameters_data = DataParameters()  # 数据相关超参数
        self.parameters_predictor = PredictorParameters()  # 模型相关超参数
        self.parameters_ensemble = EnsembleParameters()  # 集成相关超参数
        # 获取绝对路径，实例化 Writer 类。
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.read_predictor_file = os.path.join(current_dir, 'Datasets', self.parameters_project['file_name'])
        self.save_predictor_dir = os.path.join(current_dir, self.parameters_project["save_predictor_dir"].strip(r'./\\'))
        self.read_ensemble_dir = os.path.join(current_dir, self.parameters_project["read_predictor_dir"].strip(r'./\\'))
        self.save_ensemble_dir = os.path.join(current_dir, self.parameters_project["save_ensemble_dir"].strip(r'./\\'))


    def writer_base_param(self, writer: Writer):
        writer.add_param(
            param_desc = '[project parameters]',
            param_dict = self.parameters_project,
            filename = 'config parameters',
            folder = 'documents',
            save_mode = 'a+'
        )
        writer.add_param(
            param_desc = '[data parameters]',
            param_dict = self.parameters_data,
            filename = 'config parameters',
            folder = 'documents',
            save_mode = 'a+'
        )

    def write_predictor_param(self):
        self.predictor_writer.add_param(
            param_desc = '[predictor parameters]',
            param_dict = self.parameters_predictor,
            filename='config parameters',
            folder='documents',
            save_mode='a+'
        )

    def write_ensemble_param(self):
        self.ensemble_writer.add_param(
            param_desc = '[ensemble parameters]',
            param_dict = self.parameters_ensemble,
            filename = 'config parameters',
            folder = 'documents',
            save_mode = 'a+'
        )

    def predictor_data(self, norm_method=Norm, **kwargs):
        """添加数据 和 特征选择。"""
        data = read_file(self.read_predictor_file, **kwargs)
        # 获取特征（DataFrame）和目标（Series）
        feature, target = data[self.parameters_data["feature"]], data[self.parameters_data["target"]]
        # 特征选择
        print(f"预测器开始特征选择...")
        selector = select_best_feature(**self.parameters_data["feature_selection"])
        feature_selected = selector.fit_transform(feature, target)  # feature_selected 是 DataFrame 类型（包含列名）
        self.predictor_writer.add_text(
            f"特征选择结果：{feature_selected.columns.tolist()}。\n" + "-" * 100 + "\n",
            filename = "Logs", folder = "documents", suffix = "log", save_mode = 'a+'
        )  # 保存特征选择结果到日志中
        # 数据集标准化、数据集封装、数据集划分
        print(f"预测器开始数据封装...")
        self.seq_split = SeqSplit(feature_selected, target, self.parameters_data, normalization=norm_method)  # 机器学习模型的数据集划分
        self.seq_loader = SeqLoader(feature_selected, target, self.parameters_data, normalization=norm_method)  # 深度学习模型的数据集划分

    def ensemble_data(self, norm_method=Norm):
        print(f"集成器开始数据封装...")
        self.ens_split = EnsembleSplit(self.read_ensemble_dir, normalization=norm_method)
        self.ens_loader = EnsembleLoader(
            self.read_ensemble_dir,
            batch_size = self.parameters_data['ensemble_batch_size'],
            normalization = norm_method
        )

    def run_predictor(
            self,
            models: Union[type, List[type], Tuple[type]],
            normalizations: Union[bool, List[bool], Tuple[bool]],
            criterion=None,
            monitor=None
    ):
        try:
            # 实例化 Writer 类，用于保存结果；追加写入保存项目参数、数据参数、集成器参数。
            if self.predictor_writer is None:
                self.predictor_writer = Writer(self.save_predictor_dir, is_delete=self.parameters_project['delete_dir'])
                self.writer_base_param(self.predictor_writer)
                self.write_predictor_param()
            # 封装数据
            if self.seq_split is None or self.seq_loader is None:
                self.predictor_data()
            # 初始化预测器
            if self.predictors is None:
                self.predictors = Predictors(self.seq_split, self.seq_loader, writer=self.predictor_writer)
            # 运行模型
            print("开始尝试运行预测器...")
            if isinstance(models, type) and isinstance(normalizations, bool):
                assert models.__name__ in REGRESSION_MODELS+PREDICTOR_MODELS, "models 参数不在预定义的模型中！"
                trained_model = self.predictors.model(
                    models, self.parameters_predictor[models.__name__], is_normalization=normalizations,
                    criterion=criterion, monitor=monitor, train_parameters=self.parameters_predictor['DL_train']
                )
                self.trained_predictors.append(trained_model)
            elif isinstance(models, (list, tuple)) and isinstance(normalizations, (list, tuple)):
                assert all([m.__name__ in REGRESSION_MODELS+PREDICTOR_MODELS for m in models]), "models 参数不在预定义的模型中！"
                trained_models = self.predictors.all_models(
                    models, self.parameters_predictor, is_normalization=normalizations,
                    criterion=criterion, monitor=monitor
                )
                self.trained_predictors.extend(trained_models)
            else:
                raise ValueError("models 和 normalizations 参数类型不匹配！")
            # 保存暂存结果
            self.predictor_writer.write(self.parameters_project['save_mode'])
            cprint("-"*100 + '\n预测器训练结束！\n' + '-'*100 + '\n', text_color='红色', background_color='青色', style='加粗')
            return self.trained_predictors
        except BaseException as e:
            error_message = traceback.format_exc()
            self.predictor_writer.add_text(
                "", filename="Logs", folder="documents", suffix="log", save_mode='a+'
            )
            self.predictor_writer.add_text(
                f"Error: {error_message}", filename="Logs", folder="documents", suffix="log", save_mode='a+'
            )
            self.predictor_writer.write(self.parameters_project['save_mode'])
            raise e

    def run_ensemble(
            self,
            models: Union[type, List[type], Tuple[type]],
            normalizations: Union[bool, List[bool], Tuple[bool]],
            criterion=None,
            monitor=None
    ):
        try:
            # 实例化 Writer 类，用于保存结果；追加写入保存项目参数、数据参数、集成器参数。
            if self.ensemble_writer is None:
                self.ensemble_writer = Writer(self.save_ensemble_dir, is_delete=self.parameters_project['delete_dir'])
                self.writer_base_param(self.ensemble_writer)
                self.write_ensemble_param()
            # 封装数据
            if self.ens_split is None or self.ens_loader is None:
                self.ensemble_data()
            # 初始化集成器
            if self.ensembles is None:
                self.ensembles = Ensembles(self.ens_split, self.ens_loader, writer=self.ensemble_writer)
            # 运行模型
            print("开始尝试运行集成器...")
            if isinstance(models, type) and isinstance(normalizations, bool):
                assert models.__name__ in REGRESSION_MODELS+ENSEMBLE_MODELS, "models 参数不在预定义的模型中！"
                trained_model = self.ensembles.model(
                    models, self.parameters_ensemble[models.__name__], is_normalization=normalizations,
                    criterion=criterion, monitor=monitor, train_parameters=self.parameters_ensemble['DL_train']
                )
                self.trained_ensembles.append(trained_model)
            elif isinstance(models, (list, tuple)) and isinstance(normalizations, (list, tuple)):
                assert all([m.__name__ in REGRESSION_MODELS+ENSEMBLE_MODELS for m in models]), "models 参数不在预定义的模型中！"
                trained_models = self.ensembles.all_models(
                    models, self.parameters_ensemble, is_normalization=normalizations,
                    criterion=criterion, monitor=monitor
                )
                self.trained_ensembles.extend(trained_models)
            else:
                raise ValueError("models 和 normalizations 参数类型不匹配！")
            # 保存暂存结果
            self.ensemble_writer.write(self.parameters_project['save_mode'])
            cprint("-"*100 + '\n集成器训练结束！\n' + '-'*100 + '\n', text_color='红色', background_color='青色', style='加粗')
            return self.trained_ensembles
        except BaseException as e:
            error_message = traceback.format_exc()
            self.ensemble_writer.add_text(
                "", filename="Logs", folder="documents", suffix="log", save_mode='a+'
            )
            self.ensemble_writer.add_text(
                f"Error: {error_message}", filename="Logs", folder="documents", suffix="log", save_mode='a+'
            )
            self.ensemble_writer.write(self.parameters_project['save_mode'])
            raise e
