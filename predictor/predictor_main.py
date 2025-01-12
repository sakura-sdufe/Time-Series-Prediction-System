# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation-Project
# @Time     : 2024/11/26 21:08
# @Author   : 张浩
# @FileName : predictor_main.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import os
import time
import warnings
import pandas as pd
from torch import nn
from torch import optim
from sklearn import base
from typing import List, Dict, Tuple, Union

from data import DataSplit, SeqLoader
from metrics import calculate_metrics
from utils import cprint, Writer

from .predictor_parameters import PredictorParameters


class Predictors:
    def __init__(self, data_split: DataSplit, seq_loader: SeqLoader, *, writer: Writer):
        """
        预测器类，用于训练和预测多个模型，并保存结果和评估指标。
        :param data_split: 机器学习数据集划分类，数据类型为 DataSplit 类的实例。
        :param seq_loader: 深度学习数据集封装类，数据类型为 SeqLoader 类的实例。
        :param writer: Writer 类的实例，用于保存预测结果、评价指标和训练后的模型。
        """
        # 转为类内属性
        self.data_split = data_split  # 机器学习数据集划分类
        self.seq_loader = seq_loader  # 深度学习数据集封装类
        self.writer = writer  # Writer 类的实例，用于保存预测结果、评价指标和训练后的模型
        self.save_dir:str = self.writer.save_dir  # 保存路径

        self.ML_normalization_target, self.DL_normalization_target = None, None  # 机器学习和深度学习目标标准化类
        # 机器学习数据集
        self.train_feature, self.valid_feature, self.test_feature = None, None, None  # 未标准化的特征
        self.train_target, self.valid_target, self.test_target = None, None, None  # 未标准化的目标
        self.train_feature_norm, self.valid_feature_norm, self.test_feature_norm = None, None, None  # 标准化的特征
        self.train_target_norm, self.valid_target_norm, self.test_target_norm = None, None, None  # 标准化的目标
        self._machine_learning_data()
        # 深度学习数据集
        self.train_trainer, self.train_evaler, self.valid_evaler, self.test_evaler = None, None, None, None
        self.train_trainer_norm, self.train_evaler_norm, self.valid_evaler_norm, self.test_evaler_norm = None, None, None, None
        self._deep_learning_data()
        # 保存各数据集中的真实值
        self._writer_true()

    def _machine_learning_data(self):
        # 准备未标准化数据集
        self.train_feature, self.train_target = self.data_split.get_dataset('train', scale=False, to_numpy=True)
        self.valid_feature, self.valid_target = self.data_split.get_dataset('valid', scale=False, to_numpy=True)
        self.test_feature, self.test_target = self.data_split.get_dataset('test', scale=False, to_numpy=True)
        if self.data_split.normalization:
            self.ML_normalization_target = self.data_split.normalization_target  # 机器学习目标标准化类
            # 标准化后的机器学习数据集
            self.train_feature_norm, self.train_target_norm = self.data_split.get_dataset('train', scale=True, to_numpy=True)
            self.valid_feature_norm, self.valid_target_norm = self.data_split.get_dataset('valid', scale=True, to_numpy=True)
            self.test_feature_norm, self.test_target_norm = self.data_split.get_dataset('test', scale=True, to_numpy=True)

    def _deep_learning_data(self):
        self.DL_normalization_target = self.seq_loader.normalization_target  # 深度学习目标标准化类
        self.train_trainer, self.train_evaler, self.valid_evaler, self.test_evaler = \
            self.seq_loader.get_all_loader(scale=False)
        self.train_trainer_norm, self.train_evaler_norm, self.valid_evaler_norm, self.test_evaler_norm = \
            self.seq_loader.get_all_loader(scale=True)

    def _writer_true(self):
        self.writer.add_data(
            data_df=pd.DataFrame(self.train_target, columns=['ML True'], index=range(1, len(self.train_target)+1)),
            axis=1, key="train predict"
        )
        self.writer.add_data(
            data_df=pd.DataFrame(self.valid_target, columns=['ML True'], index=range(1, len(self.valid_target)+1)),
            axis=1, key="valid predict"
        )
        self.writer.add_data(
            data_df=pd.DataFrame(self.test_target, columns=['ML True'], index=range(1, len(self.test_target)+1)),
            axis=1, key="test predict"
        )
        DL_train_target = self.train_evaler.dataset.target.cpu().numpy()
        self.writer.add_data(
            data_df=pd.DataFrame(DL_train_target, columns=['DL True'], index=range(1, len(DL_train_target)+1)),
            axis=1, key="train predict"
        )
        DL_valid_target = self.valid_evaler.dataset.target.cpu().numpy()
        self.writer.add_data(
            data_df=pd.DataFrame(DL_valid_target, columns=['DL True'], index=range(1, len(DL_valid_target)+1)),
            axis=1, key="valid predict"
        )
        DL_test_target = self.test_evaler.dataset.target.cpu().numpy()
        self.writer.add_data(
            data_df=pd.DataFrame(DL_test_target, columns=['DL True'], index=range(1, len(DL_test_target)+1)),
            axis=1, key="test predict"
        )

    def ML(self, predictor, predictor_parameters:Dict, is_normalization:bool):
        """
        实例化、训练和预测机器学习模型。
        :param predictor: 机器学习模型类，不需要实例化。
        :param predictor_parameters: 机器学习模型参数，dict 类型。
        :param is_normalization: bool 类型，是否标准化。
        :return: 训练后的模型, 各数据集预测结果。
        """
        model = predictor(**predictor_parameters)
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

    def DL(self, predictor, predictor_parameters:Dict, is_normalization:bool, criterion, monitor,
           figure_type, **kwargs):
        """
        实例化、训练和预测深度学习模型。
        :param predictor: 深度学习模型类，不需要实例化。
        :param predictor_parameters: 深度学习模型参数，dict 类型。
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
            'learning_rate', 'ReduceLROnPlateau_factor', 'ReduceLROnPlateau_patience', 'ReduceLROnPlateau_threshold'
        )
        predictor_parameters.update(kwargs)  # 更新参数。这里的更新会影响 self.model 方法中的参数，但初始化模型和训练用的就是这组参数，这是一件好事。
        model_parameters, train_parameters = dict(), dict()
        for key in predictor_parameters.keys():
            if key in train_known_parameters:
                train_parameters[key] = predictor_parameters[key]
            elif key in ignore_parameters:
                continue
            else:
                model_parameters[key] = predictor_parameters[key]
        # 添加模型参数 input_size 和 训练参数 monitor_name。
        model_parameters['input_size'] = self.train_trainer.dataset.input_size
        if 'monitor_name' not in train_parameters:
            train_parameters['monitor_name'] = monitor.__name__
        if 'monitor_title' not in train_parameters:
            train_parameters['monitor_title'] = f"{predictor.__name__} 模型 {monitor.__name__} 监视器"
        if 'loss_title' not in train_parameters:
            train_parameters['loss_title'] = f"{predictor.__name__} 模型 {criterion.__name__} 损失值"
        # 生成保存路径参数
        model_dir = os.path.join(self.save_dir, 'models_trained', predictor.__name__)
        figure_dir = os.path.join(self.save_dir, 'figures', predictor.__name__)
        result_dir = os.path.join(self.save_dir, 'results', predictor.__name__)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        train_parameters['best_model_dir'] = model_dir
        train_parameters['loss_figure_path'] = os.path.join(figure_dir, predictor.__name__+'损失值曲线.'+figure_type)
        train_parameters['monitor_figure_path'] = os.path.join(figure_dir, predictor.__name__+'监视值曲线.'+figure_type)
        train_parameters['loss_result_path'] = os.path.join(result_dir, predictor.__name__+'损失值.csv')
        train_parameters['monitor_result_path'] = os.path.join(result_dir, predictor.__name__+'监视值.csv')
        train_parameters['lr_sec_path'] = os.path.join(result_dir, predictor.__name__+'学习率与每秒样本数.csv')
        # 实例化深度学习模型
        model = predictor(**model_parameters)
        criterion_instance = criterion()
        monitor_instance = monitor()
        optimizer = optim.Adam(model.parameters(), lr=predictor_parameters['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=predictor_parameters['ReduceLROnPlateau_factor'],
            patience=predictor_parameters['ReduceLROnPlateau_patience'],
            threshold=predictor_parameters['ReduceLROnPlateau_threshold']
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

    def model(self, predictor, predictor_parameters:Dict, *, is_normalization=True, figure_type='svg',
              save_result=True, save_figure=True, show_result=True, show_figure=False, criterion=None, monitor=None,
              train_parameters:Dict=None, **kwargs):
        """
        训练和预测一个模型，并保存结果和评估指标。
        :param predictor: 模型类，不需要实例化。需要有 fit 和 predict 方法。
        :param predictor_parameters: 模型参数，dict 类型。
        :param is_normalization: 是否标准化，默认为 True。
        :param figure_type: 图像保存类型，默认为 'svg'。
        :param save_result: 是否保存预测结果、预测指标、日志信息，默认为 True。
        :param save_figure: 是否保存预测结果走势图，默认为 True。
        :param show_result: 是否在控制台打印预测指标信息，默认为 True。
        :param show_figure: 是否展示绘制的预测结果走势图，默认为 False。
        :param criterion: 深度学习损失函数类，无需实例化，深度学习必须参数。
        :param monitor: 深度学习模型监视器类，无需实例化，深度学习必须参数。
        :param train_parameters: 深度学习训练参数，Dict 类型，深度学习必须参数。该字典必须包含 epochs, learning_rate, clip_norm,
                                ReduceLROnPlateau_factor, ReduceLROnPlateau_patience, ReduceLROnPlateau_threshold 键值对。
        :param kwargs: 其他参数。一般用于接收深度学习模型训练参数（来自 sklearn 的模型不可用）。具体参数可以参考深度学习模型的 fit 方法。
                        常用参数有：monitor_name, loss_title, monitor_title, loss_yscale, monitor_yscale。
                        在深度学习模型中，虽然 kwargs 也可以接受并更改 predictor_parameters 中的参数，但是不推荐这样做。
        Note: 如果是深度学习模型，则需要传入 criterion、monitor 参数。
        :return: 训练后的模型。
        """
        start_time = time.perf_counter()
        predictor_name = predictor.__name__  # 模型名称
        if show_result:
            cprint(f"开始训练和预测 {predictor_name} 模型...", text_color="白色", end='\n')
        # 模型实例化、训练、预测
        if issubclass(predictor, base.BaseEstimator):
            model, predict_result = self.ML(predictor, predictor_parameters, is_normalization)
        elif issubclass(predictor, nn.Module):
            kwargs.update(train_parameters)
            model, predict_result = self.DL(predictor, predictor_parameters, is_normalization, criterion, monitor,
                                            figure_type=figure_type, **kwargs)
        else:
            raise ValueError("模型类必须是 sklearn.base.BaseEstimator 或 torch.nn.Module 的子类！")
        train_predict, valid_predict, test_predict = predict_result

        # 计算模型的评估指标
        train_metrics = calculate_metrics(self.train_target, train_predict)
        valid_metrics = calculate_metrics(self.valid_target, valid_predict)
        test_metrics = calculate_metrics(self.test_target, test_predict)
        # 绘制和保存预测图像
        if show_figure or save_figure:
            title = f"{predictor_name} 训练集预测"
            filename = title + '.' + figure_type if save_figure else None
            self.writer.draw([self.train_target, train_predict], title=title, legend=['true', predictor_name],
                             filename=filename, show=show_figure)
            title = f"{predictor_name} 验证集预测"
            filename = title + '.' + figure_type if save_figure else None
            self.writer.draw([self.valid_target, valid_predict], title=title, legend=['true', predictor_name],
                             filename=filename, show=show_figure)
            title = f"{predictor_name} 测试集预测"
            filename = title + '.' + figure_type if save_figure else None
            self.writer.draw([self.test_target, test_predict], title=title, legend=['true', predictor_name],
                             filename=filename, show=show_figure)
        end_time = time.perf_counter()
        if show_result:
            cprint(f"{predictor_name} 模型训练集评估指标：", text_color="青色", end='\n', **train_metrics)
            cprint(f"{predictor_name} 模型验证集评估指标：", text_color="紫色", end='\n', **valid_metrics)
            cprint(f"{predictor_name} 模型测试集评估指标：", text_color="黄色", end='\n', **test_metrics)
            cprint(f"{predictor_name} 模型训练和预测结束，用时 {end_time - start_time} 秒。", text_color="蓝色")
        if save_result:
            if issubclass(predictor, base.BaseEstimator):
                self.writer.write_model(model, f"{predictor_name}.pkl")
            self.writer.add_data(
                data_df=pd.DataFrame(train_predict, columns=[predictor_name], index=range(1, len(train_predict)+1)),
                axis=1, key="train predict"
            )
            self.writer.add_data(
                data_df=pd.DataFrame(valid_predict, columns=[predictor_name], index=range(1, len(valid_predict)+1)),
                axis=1, key="valid predict"
            )
            self.writer.add_data(
                data_df=pd.DataFrame(test_predict, columns=[predictor_name], index=range(1, len(test_predict)+1)),
                axis=1, key="test predict"
            )
            self.writer.add_metrics(metrics_df=pd.DataFrame(train_metrics, index=[predictor_name]), axis=0, key="train metrics")
            self.writer.add_metrics(metrics_df=pd.DataFrame(valid_metrics, index=[predictor_name]), axis=0, key="valid metrics")
            self.writer.add_metrics(metrics_df=pd.DataFrame(test_metrics, index=[predictor_name]), axis=0, key="test metrics")
            self.writer.add_log(f"{predictor_name} 模型训续和预测用时 {(end_time - start_time):.2f} 秒。", key="log")
            self.writer.add_parameters(content_text=f"{predictor_name} 模型参数", content_dict=predictor_parameters,
                                       key="predictor parameters")
        return model

    def persistence(self, *, figure_type='svg', save_result=True, save_figure=True, show_result=True, show_figure=False):
        """
        计算 Persistence 模型的预测结果和评估指标。
        :param figure_type: 图像保存类型，默认为 'svg'。
        :param save_result: 是否保存结果，默认为 True。
        :param save_figure: 是否保存图像，默认为 True。
        :param show_result: 是否显示结果，默认为 True。
        :param show_figure: 是否显示图像，默认为 False。
        :return:
        """
        start_time = time.time()
        if show_result:
            cprint(f"开始计算 Persistence 模型预测结果和指标...", text_color="白色", end='\n')
        # 获取 Persistence 模型的预测结果
        train_predict = self.data_split.persistence_predict(dataset='train')  # 训练集预测
        valid_predict = self.data_split.persistence_predict(dataset='valid')  # 验证集预测
        test_predict = self.data_split.persistence_predict(dataset='test')  # 测试集预测
        # 计算 Persistence 模型的评估指标
        train_metrics = calculate_metrics(self.train_target, train_predict)
        valid_metrics = calculate_metrics(self.valid_target, valid_predict)
        test_metrics = calculate_metrics(self.test_target, test_predict)
        # 绘制和保存预测图像
        if show_figure or save_figure:
            title = f"Persistence 训练集预测"
            filename = title + '.' + figure_type if save_figure else None
            self.writer.draw([self.train_target, train_predict], title=title, legend=['true', "Persistence"],
                             filename=filename, show=show_figure)
            title = f"Persistence 验证集预测"
            filename = title + '.' + figure_type if save_figure else None
            self.writer.draw([self.valid_target, valid_predict], title=title, legend=['true', "Persistence"],
                             filename=filename, show=show_figure)
            title = f"Persistence 测试集预测"
            filename = title + '.' + figure_type if save_figure else None
            self.writer.draw([self.test_target, test_predict], title=title, legend=['true', "Persistence"],
                             filename=filename, show=show_figure)
        end_time = time.time()
        if show_result:
            cprint(f"Persistence 模型训练集评估指标：", text_color="青色", end='\n', **train_metrics)
            cprint(f"Persistence 模型验证集评估指标：", text_color="紫色", end='\n', **valid_metrics)
            cprint(f"Persistence 模型测试集评估指标：", text_color="黄色", end='\n', **test_metrics)
            cprint(f"Persistence 模型训练和预测结束，用时 {end_time - start_time} 秒。", text_color="蓝色")
        if save_result:
            self.writer.add_data(
                data_df=pd.DataFrame(train_predict, columns=["Persistence"], index=range(1, len(train_predict)+1)),
                axis=1, key="train predict"
            )
            self.writer.add_data(
                data_df=pd.DataFrame(valid_predict, columns=["Persistence"], index=range(1, len(valid_predict)+1)),
                axis=1, key="valid predict"
            )
            self.writer.add_data(
                data_df=pd.DataFrame(test_predict, columns=["Persistence"], index=range(1, len(test_predict)+1)),
                axis=1, key="test predict"
            )
            self.writer.add_metrics(metrics_df=pd.DataFrame(train_metrics, index=["Persistence"]), axis=0, key="train metrics")
            self.writer.add_metrics(metrics_df=pd.DataFrame(valid_metrics, index=["Persistence"]), axis=0, key="valid metrics")
            self.writer.add_metrics(metrics_df=pd.DataFrame(test_metrics, index=["Persistence"]), axis=0, key="test metrics")
            self.writer.add_log(f"Persistence 模型训续和预测用时 {(end_time - start_time):.2f} 秒。", key="log")
        return None

    def all_models(self, models: Union[List,Tuple], parameters: PredictorParameters, is_normalization:Union[List,Tuple,bool]=True,
                   *, figure_type='svg', save_result=True, save_figure=True, show_result=True, show_figure=False,
                   criterion=None, monitor=None, **kwargs) -> List:
        """
        训练和预测多个模型，并保存结果和评估指标。
        :param models: 由模型类构成的 list 或 tuple，不需要实例化。
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
        :param kwargs: 其他参数。一般用于接收深度学习模型训练参数（来自 sklearn 的模型不可用），具体参数 model 方法。
        :return: 由训练后模型构成的列表。
        """
        if isinstance(is_normalization, bool):
            is_normalization = [is_normalization] * len(models)
        trained_models = []
        self.persistence(figure_type=figure_type, save_result=save_result, save_figure=save_figure, show_result=show_result,
                         show_figure=show_figure)
        for m, is_norm in zip(models, is_normalization):
            m = self.model(m, parameters[m.__name__], is_normalization=is_norm, figure_type=figure_type,
                           save_result=save_result, save_figure=save_figure, show_result=show_result,
                           show_figure=show_figure, criterion=criterion, monitor=monitor,
                           train_parameters=parameters['DL_train'], **kwargs)
            trained_models.append(m)
        return trained_models

    def __getitem__(self, item):
        return getattr(self, item)
