# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation
# @Time     : 2025/2/26 11:10
# @Author   : 张浩
# @FileName : ensemble.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import os
import time
import inspect
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
from sklearn import base
from copy import deepcopy
from typing import List, Dict, Tuple, Union

from data import EnsembleSplit, EnsembleLoader
from metrics import calculate_metrics
from utils import cprint, Writer

from ensemble_parameters import EnsembleParameters


class Ensembles:
    def __init__(self, ens_split:EnsembleSplit, ens_loader:EnsembleLoader, *, writer: Writer):
        """
        集成器类，用于将多个预测结果作为特征，真实值作为目标，进行回归。实现集成学习的训练和预测，并保存结果和评估指标。
        :param ens_split: EnsembleSplit 类的实例，用于读取预测结果和真实值（机器学习封装格式）。
        :param ens_loader: EnsembleLoader 类的实例，用于封装数据集（深度学习封装格式）。
        :param writer: Writer 类的实例，用于保存预测结果、评价指标和训练后的模型。
        """
        # 转为类内属性
        self.ens_split = ens_split  # 机器学习数据集
        self.ens_loader = ens_loader  # 深度学习数据集
        self.writer = writer  # 保存结果的 Writer 类
        self.save_dir = writer.save_dir  # 保存结果的根目录
        # 机器学习数据集
        self.train_feature, self.valid_feature, self.test_feature = None, None, None  # 未标准化的特征
        self.train_target, self.valid_target, self.test_target = None, None, None  # 未标准化的目标
        if ens_split.normalization:
            self.ML_normalization_target = None  # 机器学习目标标准化类
            self.train_feature_norm, self.valid_feature_norm, self.test_feature_norm = None, None, None  # 标准化的特征
            self.train_target_norm, self.valid_target_norm, self.test_target_norm = None, None, None  # 标准化的目标
        self._machine_learning_data()
        # 深度学习数据集
        self.train_trainer, self.train_evaler, self.valid_evaler, self.test_evaler = None, None, None, None
        if ens_loader.normalization:
            self.DL_normalization_target = None  # 深度学习目标标准化类
            self.train_trainer_norm, self.train_evaler_norm, self.valid_evaler_norm, self.test_evaler_norm = \
                None, None, None, None
        self._deep_learning_data()
        # 保存各数据集中的真实值
        self._write_feature()
        self._writer_target()

    def _machine_learning_data(self):
        # 准备未标准化数据集
        self.train_feature, self.train_target = self.ens_split.get_dataset('train', scale=False, to_numpy=True)
        self.valid_feature, self.valid_target = self.ens_split.get_dataset('valid', scale=False, to_numpy=True)
        self.test_feature, self.test_target = self.ens_split.get_dataset('test', scale=False, to_numpy=True)
        if self.ens_split.normalization:
            self.ML_normalization_target = self.ens_split.normalization_target  # 机器学习目标标准化类
            # 标准化后的机器学习数据集
            self.train_feature_norm, self.train_target_norm = self.ens_split.get_dataset(
                'train', scale=True, to_numpy=True
            )
            self.valid_feature_norm, self.valid_target_norm = self.ens_split.get_dataset(
                'valid', scale=True, to_numpy=True
            )
            self.test_feature_norm, self.test_target_norm = self.ens_split.get_dataset(
                'test', scale=True, to_numpy=True
            )

    def _deep_learning_data(self):
        self.train_trainer, self.train_evaler, self.valid_evaler, self.test_evaler = \
            self.ens_loader.get_all_loader(scale=False)
        if self.ens_loader.normalization:
            self.DL_normalization_target = self.ens_loader.normalization_target  # 深度学习目标标准化类
            self.train_trainer_norm, self.train_evaler_norm, self.valid_evaler_norm, self.test_evaler_norm = \
                self.ens_loader.get_all_loader(scale=True)

    def _write_feature(self):
        # 写入 data
        self.writer.add_df(
            data_df=self.ens_split.train_feature.set_index(pd.Index(range(1, len(self.ens_split.train_feature) + 1))),
            axis=1, filename="train data (ML)", folder="data", save_mode='a+'
        )
        self.writer.add_df(
            data_df=self.ens_split.valid_feature.set_index(pd.Index(range(1, len(self.ens_split.valid_feature) + 1))),
            axis=1, filename="valid data (ML)", folder="data", save_mode='a+'
        )
        self.writer.add_df(
            data_df=self.ens_split.test_feature.set_index(pd.Index(range(1, len(self.ens_split.test_feature) + 1))),
            axis=1, filename="test data (ML)", folder="data", save_mode='a+'
        )
        self.writer.write_file(self.train_trainer, filename="train trainer", folder="data")
        self.writer.write_file(self.train_evaler, filename="train evaler", folder="data")
        self.writer.write_file(self.valid_evaler, filename="valid evaler", folder="data")
        self.writer.write_file(self.test_evaler, filename="test evaler", folder="data")
        # 写入标准化后 data
        if self.ens_split.normalization:
            feature = deepcopy(self.ens_split.train_feature_norm.set_index(
                pd.Index(range(1, len(self.ens_split.train_feature_norm) + 1))))
            feature.columns = [f"{col} (standardized)" for col in feature.columns]
            self.writer.add_df(data_df=feature, axis=1, filename="train data (standardized, ML)", folder="data",
                               save_mode='a+')
            feature = deepcopy(self.ens_split.valid_feature_norm.set_index(
                pd.Index(range(1, len(self.ens_split.valid_feature_norm) + 1))))
            feature.columns = [f"{col} (standardized)" for col in feature.columns]
            self.writer.add_df(data_df=feature, axis=1, filename="valid data (standardized, ML)", folder="data",
                               save_mode='a+')
            feature = deepcopy(self.ens_split.test_feature_norm.set_index(
                pd.Index(range(1, len(self.ens_split.test_feature_norm) + 1))))
            feature.columns = [f"{col} (standardized)" for col in feature.columns]
            self.writer.add_df(data_df=feature, axis=1, filename="test data (standardized, ML)", folder="data",
                               save_mode='a+')
            self.writer.write_file(self.train_trainer_norm, filename="train trainer (standardized)", folder="data")
            self.writer.write_file(self.train_evaler_norm, filename="train evaler (standardized)", folder="data")
            self.writer.write_file(self.valid_evaler_norm, filename="valid evaler (standardized)", folder="data")
            self.writer.write_file(self.test_evaler_norm, filename="test evaler (standardized)", folder="data")

    def _writer_target(self):
        # 判断机器学习目标和深度学习目标是否对齐（标准化后目标结果可以选择不验证）。
        assert np.allclose(self.train_target, self.train_evaler.dataset.target.cpu().numpy()), "机器学习训练集目标和深度学习训练集目标不一致！"
        assert np.allclose(self.valid_target, self.valid_evaler.dataset.target.cpu().numpy()), "机器学习验证集目标和深度学习验证集目标不一致！"
        assert np.allclose(self.test_target, self.test_evaler.dataset.target.cpu().numpy()), "机器学习测试集目标和深度学习测试集目标不一致！"
        assert np.allclose(self.train_target_norm,
                           self.train_evaler_norm.dataset.target.cpu().numpy()), "机器学习训练集标准后目标和深度学习训练集标准后目标不一致！"
        assert np.allclose(self.valid_target_norm,
                           self.valid_evaler_norm.dataset.target.cpu().numpy()), "机器学习验证集标准后目标和深度学习验证集标准后目标不一致！"
        assert np.allclose(self.test_target_norm,
                           self.test_evaler_norm.dataset.target.cpu().numpy()), "机器学习测试集标准后目标和深度学习测试集标准后目标不一致！"
        # 将 target 写入预测结果文件
        self.writer.add_df(
            data_df=pd.DataFrame(self.train_target, columns=['True'], index=range(1, len(self.train_target) + 1)),
            axis=1, filename="train predict", folder="results", suffix='xlsx', save_mode='a+'
        )
        self.writer.add_df(
            data_df=pd.DataFrame(self.valid_target, columns=['True'], index=range(1, len(self.valid_target) + 1)),
            axis=1, filename="valid predict", folder="results", suffix='xlsx', save_mode='a+'
        )
        self.writer.add_df(
            data_df=pd.DataFrame(self.test_target, columns=['True'], index=range(1, len(self.test_target) + 1)),
            axis=1, filename="test predict", folder="results", suffix='xlsx', save_mode='a+'
        )
        # 将 target 和标准化后 target 写入 data 下的文件中
        if self.ens_split.normalization:
            self.writer.add_df(
                data_df=pd.DataFrame(np.stack([self.train_target, self.train_target_norm]).T,
                                     columns=['True', 'True (standardized)'],
                                     index=range(1, len(self.train_target) + 1)),
                axis=1, filename="train target", folder="data", save_mode='a+'
            )
            self.writer.add_df(
                data_df=pd.DataFrame(np.stack([self.valid_target, self.valid_target_norm]).T,
                                     columns=['True', 'True (standardized)'],
                                     index=range(1, len(self.valid_target) + 1)),
                axis=1, filename="valid target", folder="data", save_mode='a+'
            )
            self.writer.add_df(
                data_df=pd.DataFrame(np.stack([self.test_target, self.test_target_norm]).T,
                                     columns=['True', 'True (standardized)'],
                                     index=range(1, len(self.test_target) + 1)),
                axis=1, filename="test target", folder="data", save_mode='a+'
            )

    def ML(self, ensemble, ensemble_parameters:Dict, is_normalization:bool):
        """
        实例化、训练和预测机器学习模型。
        :param ensemble: 机器学习模型类，不需要实例化。
        :param ensemble_parameters: 机器学习模型参数，dict 类型。
        :param is_normalization: bool 类型，是否标准化。
        :return: 训练后的模型, 各数据集预测结果。
        """
        model = ensemble(**ensemble_parameters)
        if is_normalization:
            model.fit(self.train_feature_norm, self.train_target_norm)
            train_predict = self.ML_normalization_target.denorm(model.predict(self.train_feature_norm))
            valid_predict = self.ML_normalization_target.denorm(model.predict(self.valid_feature_norm))
            test_predict = self.ML_normalization_target.denorm(model.predict(self.test_feature_norm))
        else:
            model.fit(self.train_feature, self.train_target)
            train_predict = model.predict(self.train_feature)
            valid_predict = model.predict(self.valid_feature)
            test_predict = model.predict(self.test_feature)
        return model, (train_predict, valid_predict, test_predict)

    def DL(self, ensemble, ensemble_parameters:Dict, is_normalization:bool, criterion, monitor,
           figure_type, **kwargs):
        """
        实例化、训练和预测深度学习模型。
        :param ensemble: 深度学习模型类，不需要实例化。
        :param ensemble_parameters: 深度学习模型参数，dict 类型。
        :param is_normalization: bool 类型，是否标准化。
        :param criterion: 损失函数。
        :param monitor: 模型监视器。
        :param figure_type: 图像保存类型。
        :param kwargs: 其他参数。
        :return: 训练后的模型, 各数据集预测结果。
        """
        # 从 predictor_parameters 和 kwargs 筛选出深度学习模型参数 和 深度学习训练参数
        train_known_parameters = (
            'epochs', 'clip_norm', 'device', 'best_model_dir', 'loss_figure_path',
            'monitor_figure_path', 'loss_result_path', 'monitor_result_path', 'monitor_name', 'loss_title',
            'monitor_title', 'loss_yscale', 'monitor_yscale'
        )
        ignore_parameters = (
            'learning_rate', 'weight_decay', 'ReduceLROnPlateau_factor', 'ReduceLROnPlateau_patience',
            'ReduceLROnPlateau_threshold'
        )
        ensemble_parameters.update(kwargs)  # 更新参数。这里的更新原位修改，但初始化模型和训练用的就是这组参数，这是一件好事（会记录在文档里）。
        model_parameters, train_parameters = dict(), dict()
        for key in ensemble_parameters.keys():
            if key in train_known_parameters:
                train_parameters[key] = ensemble_parameters[key]
            elif key in ignore_parameters:
                continue
            else:
                model_parameters[key] = ensemble_parameters[key]
        # 添加模型参数 input_size 和 训练参数 monitor_name。如果存在 time_step，则添加 time_step 参数。
        model_parameters['input_size'] = self.train_trainer.dataset.input_size
        if 'time_step' in inspect.signature(ensemble).parameters:  # 如果模型有 time_step 参数
            model_parameters['time_step'] = self.train_trainer.dataset.time_step
        if 'monitor_name' not in train_parameters:
            train_parameters['monitor_name'] = monitor.__name__
        if 'monitor_title' not in train_parameters:
            train_parameters['monitor_title'] = f"{ensemble.__name__} 模型 {monitor.__name__} 监视器"
        if 'loss_title' not in train_parameters:
            train_parameters['loss_title'] = f"{ensemble.__name__} 模型 {criterion.__name__} 损失值"
        # 生成保存路径参数
        model_dir = os.path.join(self.save_dir, 'models', ensemble.__name__)
        figure_dir = os.path.join(self.save_dir, 'figures', ensemble.__name__)
        result_dir = os.path.join(self.save_dir, 'results', ensemble.__name__)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        train_parameters['best_model_dir'] = model_dir
        train_parameters['loss_figure_path'] = os.path.join(figure_dir, ensemble.__name__+'损失值曲线.'+figure_type)
        train_parameters['monitor_figure_path'] = os.path.join(figure_dir, ensemble.__name__+'监视值曲线.'+figure_type)
        train_parameters['loss_result_path'] = os.path.join(result_dir, ensemble.__name__+'损失值.csv')
        train_parameters['monitor_result_path'] = os.path.join(result_dir, ensemble.__name__+'监视值.csv')
        train_parameters['lr_sec_path'] = os.path.join(result_dir, ensemble.__name__+'学习率与每秒样本数.csv')
        # 实例化深度学习模型
        model = ensemble(**model_parameters)
        criterion_instance = criterion()
        monitor_instance = monitor()
        optimizer = optim.Adam(model.parameters(), lr=ensemble_parameters['learning_rate'],
                               weight_decay=ensemble_parameters['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=ensemble_parameters['ReduceLROnPlateau_factor'],
            patience=ensemble_parameters['ReduceLROnPlateau_patience'],
            threshold=ensemble_parameters['ReduceLROnPlateau_threshold']
        )
        if is_normalization:
            model.fit(train_trainer=self.train_trainer_norm, train_evaler=self.train_evaler_norm,
                      valid_evaler=self.valid_evaler_norm, criterion=criterion_instance, optimizer=optimizer,
                      scheduler=scheduler, monitor=monitor_instance, **train_parameters)
            train_predict = self.DL_normalization_target.denorm(model.predict(self.train_evaler_norm).cpu().flatten().numpy())
            valid_predict = self.DL_normalization_target.denorm(model.predict(self.valid_evaler_norm).cpu().flatten().numpy())
            test_predict = self.DL_normalization_target.denorm(model.predict(self.test_evaler_norm).cpu().flatten().numpy())
        else:
            model.fit(train_trainer=self.train_trainer, train_evaler=self.train_evaler,
                      valid_evaler=self.valid_evaler, criterion=criterion_instance, optimizer=optimizer,
                      scheduler=scheduler, monitor=monitor_instance, **train_parameters)
            train_predict = model.predict(self.train_evaler).cpu().flatten().numpy()
            valid_predict = model.predict(self.valid_evaler).cpu().flatten().numpy()
            test_predict = model.predict(self.test_evaler).cpu().flatten().numpy()
        return model, (train_predict, valid_predict, test_predict)

    def model(self, ensemble, ensemble_parameters:Dict, *, is_normalization=True, figure_type='svg',
              save_result=True, save_figure=True, show_result=True, show_figure=False, criterion=None, monitor=None,
              train_parameters:Dict=None, **kwargs):
        """
        训练和预测一个模型，并保存结果和评估指标。
        :param ensemble: 模型类，不需要实例化。需要有 fit 和 predict 方法。
        :param ensemble_parameters: 模型参数，dict 类型。
        :param is_normalization: 是否标准化，默认为 True。
        :param figure_type: 图像保存类型，默认为 'svg'。
        :param save_result: 是否保存预测结果、预测指标、日志信息，默认为 True。
        :param save_figure: 是否保存预测结果走势图，默认为 True。
        :param show_result: 是否在控制台打印预测指标信息，默认为 True。
        :param show_figure: 是否展示绘制的预测结果走势图，默认为 False。
        :param criterion: 深度学习损失函数类，无需实例化，深度学习必须参数。
        :param monitor: 深度学习模型监视器类，无需实例化，深度学习必须参数。
        :param train_parameters: 深度学习训练参数，Dict 类型，深度学习必须参数。该字典必须包含 epochs, learning_rate, clip_norm,
                                weight_decay, ReduceLROnPlateau_factor, ReduceLROnPlateau_patience,
                                ReduceLROnPlateau_threshold 键值对。
        :param kwargs: 其他参数。一般用于接收深度学习模型训练参数（来自 sklearn 的模型不可用）。具体参数可以参考深度学习模型的 fit 方法。
                        常用参数有：monitor_name, loss_title, monitor_title, loss_yscale, monitor_yscale。
                        在深度学习模型中，虽然 kwargs 也可以接受并更改 predictor_parameters 中的参数，但是不推荐这样做。
        Note: 如果是深度学习模型，则需要传入 criterion、monitor 参数。
        :return: 训练后的模型。
        """
        start_time = time.perf_counter()
        predictor_name = ensemble.__name__  # 模型名称
        if show_result:
            cprint(f"开始训练和预测 {predictor_name} 模型...", text_color="白色", end='\n')
        # 模型实例化、训练、预测
        if issubclass(ensemble, base.BaseEstimator):
            model, predict_result = self.ML(ensemble, ensemble_parameters, is_normalization)
        elif issubclass(ensemble, nn.Module):
            kwargs.update(train_parameters)
            model, predict_result = self.DL(ensemble, ensemble_parameters, is_normalization, criterion, monitor,
                                            figure_type=figure_type, **kwargs)
        else:
            raise ValueError("模型类必须是 sklearn.base.BaseEstimator 或 torch.nn.Module 的子类！")
        train_predict, valid_predict, test_predict = predict_result

        # 计算模型的评估指标
        train_metrics = calculate_metrics(self.train_target, train_predict)
        valid_metrics = calculate_metrics(self.valid_target, valid_predict)
        test_metrics = calculate_metrics(self.test_target, test_predict)
        # 绘制和保存集成图像
        if show_figure or save_figure:
            title = f"{predictor_name} 训练集集成"
            filename = title if save_figure else None
            self.writer.draw([self.train_target, train_predict], title=title, legend=['true', predictor_name],
                             filename=filename, folder='figures', suffix=figure_type, show=show_figure)
            title = f"{predictor_name} 验证集集成"
            filename = title if save_figure else None
            self.writer.draw([self.valid_target, valid_predict], title=title, legend=['true', predictor_name],
                             filename=filename, folder='figures', suffix=figure_type, show=show_figure)
            title = f"{predictor_name} 测试集集成"
            filename = title if save_figure else None
            self.writer.draw([self.test_target, test_predict], title=title, legend=['true', predictor_name],
                             filename=filename, folder='figures', suffix=figure_type, show=show_figure)
        end_time = time.perf_counter()
        if show_result:
            cprint(f"{predictor_name} 模型训练集评估指标：", text_color="青色", end='\n', **train_metrics)
            cprint(f"{predictor_name} 模型验证集评估指标：", text_color="紫色", end='\n', **valid_metrics)
            cprint(f"{predictor_name} 模型测试集评估指标：", text_color="黄色", end='\n', **test_metrics)
            cprint(f"{predictor_name} 模型训练和预测结束，用时 {end_time - start_time} 秒。", text_color="蓝色")
        if save_result:
            if issubclass(ensemble, base.BaseEstimator):
                self.writer.write_file(model, filename=predictor_name, folder='models')
            # 保存预测结果
            self.writer.add_df(
                data_df=pd.DataFrame(train_predict, columns=[predictor_name], index=range(1, len(train_predict) + 1)),
                axis=1, filename="train predict", folder="results", suffix="xlsx", save_mode='a+'
            )
            self.writer.add_df(
                data_df=pd.DataFrame(valid_predict, columns=[predictor_name], index=range(1, len(valid_predict) + 1)),
                axis=1, filename="valid predict", folder="results", suffix="xlsx", save_mode='a+'
            )
            self.writer.add_df(
                data_df=pd.DataFrame(test_predict, columns=[predictor_name], index=range(1, len(test_predict) + 1)),
                axis=1, filename="test predict", folder="results", suffix="xlsx", save_mode='a+'
            )
            # 保存评估指标
            self.writer.add_df(pd.DataFrame(train_metrics, index=[predictor_name]),
                               axis=0, filename="train metrics", folder="results", suffix="xlsx")
            self.writer.add_df(pd.DataFrame(valid_metrics, index=[predictor_name]),
                               axis=0, filename="valid metrics", folder="results", suffix="xlsx")
            self.writer.add_df(pd.DataFrame(test_metrics, index=[predictor_name]),
                               axis=0, filename="test metrics", folder="results", suffix="xlsx")
            # 保存日志信息
            self.writer.add_text(f"{predictor_name} 模型训续和预测用时 {(end_time - start_time):.2f} 秒。",
                                 filename="Logs", folder="documents", suffix="log")
            self.writer.add_param(param_desc=f"{predictor_name} 模型参数", param_dict=ensemble_parameters,
                                  filename="predictor parameters", folder='documents')
        return model

    def all_models(self, ensembles: Union[List,Tuple], parameters: EnsembleParameters, is_normalization:Union[List,Tuple,bool]=True,
                   *, figure_type='svg', save_result=True, save_figure=True, show_result=True, show_figure=False,
                   criterion=None, monitor=None, **kwargs) -> List:
        """
        训练和预测多个模型，并保存结果和评估指标。
        :param ensembles: 由模型类构成的 list 或 tuple，不需要实例化。
        :param parameters: 模型参数，PredictorParameters 类的实例。
        :param is_normalization: 是否标准化。可以接受 bool 类型、list 类型或 tuple 类型。如果为 bool 类型，则所有模型都使用相同的标准化方式；
                              如果为 list 类型或 tuple 类型，则每个模型使用对应的标准化方式（保证长度与 models_trained 长度一致）。
        :param figure_type: 图像保存类型，默认为 'svg'。
        :param save_result: 是否保存结果，默认为 True。
        :param save_figure: 是否保存图像，默认为 True。
        :param show_result: 是否显示结果，默认为 True。
        :param show_figure: 是否显示图像，默认为 False。
        :param criterion: 深度学习损失函数类，无需实例化，深度学习必须参数。
        :param monitor: 深度学习模型监视器类，无需实例化，深度学习必须参数。
        :param kwargs: 其他参数。一般用于接收深度学习模型训练参数（来自 sklearn 的模型不可用），具体参数 predict 方法。
        :return: 由训练后模型构成的列表。
        """
        if isinstance(is_normalization, bool):
            is_normalization = [is_normalization] * len(ensembles)
        trained_models = []
        for m, is_norm in zip(ensembles, is_normalization):
            m = self.model(m, parameters[m.__name__], is_normalization=is_norm, figure_type=figure_type,
                           save_result=save_result, save_figure=save_figure, show_result=show_result,
                           show_figure=show_figure, criterion=criterion, monitor=monitor,
                           train_parameters=parameters['DL_train'], **kwargs)
            trained_models.append(m)
        return trained_models

    def __getitem__(self, item):
        return getattr(self, item)
