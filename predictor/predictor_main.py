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
import pandas as pd
from typing import List, Dict, Tuple, Union

from data import DataSplit
from metrics import calculate_metrics
from display import cprint, Drawer, Writer

from .predictor_parameters import PredictorParameters


class Predictors:
    def __init__(self, data_split: DataSplit, save_relpath:str, *, x_label='时间', y_label='预测结果'):
        """
        预测器类，用于训练和预测多个模型，并保存结果和评估指标。
        :param data_split: 数据集划分类
        :param save_relpath: 保存路径的相对路径
        :param x_label: x 轴标签
        :param y_label: y 轴标签
        """
        # 转为类内属性
        self.data_split = data_split  # 数据集划分类

        # 将 parameters_predictor['save_dir'] 中的相对路径转为绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件绝对路径
        self.save_dir = os.path.join(os.path.dirname(current_dir), save_relpath.strip(r"./\\"))  # 保存路径
        self.normalization_target = self.data_split.normalization_target  # 目标标准化属性
        self.writer = Writer(self.save_dir)  # 实例化 Writer 类，用于保存预测结果、评价指标和训练后的模型
        self.drawer = Drawer(self.save_dir, xlabel=x_label, ylabel=y_label)  # 实例化 Drawer 类，用于绘制预测结果

        # 准备未标准化数据集
        self.train_feature, self.train_target = data_split.get_dataset('train', scale=False, to_numpy=True)
        self.valid_feature, self.valid_target = data_split.get_dataset('valid', scale=False, to_numpy=True)
        self.test_feature, self.test_target = data_split.get_dataset('test', scale=False, to_numpy=True)
        # 准备标准化数据集
        self.train_feature_norm, self.train_target_norm = data_split.get_dataset('train', scale=True, to_numpy=True)
        self.valid_feature_norm, self.valid_target_norm = data_split.get_dataset('valid', scale=True, to_numpy=True)
        self.test_feature_norm, self.test_target_norm = data_split.get_dataset('test', scale=True, to_numpy=True)
        # 保存未标准化的数据集
        self.writer.add_data(data_df=pd.DataFrame(self.train_target, columns=['True']), axis=1, key="train predict")
        self.writer.add_data(data_df=pd.DataFrame(self.valid_target, columns=['True']), axis=1, key="valid predict")
        self.writer.add_data(data_df=pd.DataFrame(self.test_target, columns=['True']), axis=1, key="test predict")

    def __getitem__(self, item):
        return getattr(self, item)

    def all_models(self, models: Union[List,Tuple], parameters: PredictorParameters, is_normalization:Union[List,Tuple,bool]=True,
                   *, figure_type='svg', save_result=True, save_figure=True, show_result=True, show_figure=False,
                   to_save=True) -> List:
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
        :param to_save: 是否将保存的结果导出到本地文件，默认为 True。如果为 False，则只是将 save_result 的结果进行暂存，不导出到本地文件。
                        可以通过 predictors.writer.write() 方法将暂存的结果导出到本地文件。
        :return: 由训练后模型构成的列表。
        """
        if isinstance(is_normalization, bool):
            is_normalization = [is_normalization] * len(models)
        trained_models = []
        self.persistence(figure_type=figure_type, save_result=save_result, save_figure=save_figure, show_result=show_result,
                         show_figure=show_figure, to_save=False)
        for m, is_norm in zip(models, is_normalization):
            m = self.model(m, parameters[m.__name__], is_normalization=is_norm, figure_type=figure_type,
                           save_result=save_result, save_figure=save_figure, show_result=show_result,
                           show_figure=show_figure, to_save=False)
            trained_models.append(m)
        if to_save:
            self.writer.write()
        return trained_models

    def model(self, predictor, predictor_parameters:Dict, *, is_normalization=True, figure_type='svg', save_result=True,
              save_figure=True, show_result=True, show_figure=False, to_save=True):
        """
        训练和预测一个模型，并保存结果和评估指标。
        :param predictor: 模型类，不需要实例化。需要有 fit 和 predict 方法。
        :param predictor_parameters: 模型参数，dict 类型。
        :param is_normalization: 是否标准化，默认为 True。
        :param figure_type: 图像保存类型，默认为 'svg'。
        :param save_result: 是否保存结果，默认为 True。
        :param save_figure: 是否保存图像，默认为 True。
        :param show_result: 是否显示结果，默认为 True。
        :param show_figure: 是否显示图像，默认为 False。
        :param to_save: 是否将保存的结果导出到本地文件，默认为 True。如果为 False，则只是将 save_result 的结果进行暂存，不导出到本地文件。
                        可以通过 predictors.writer.write() 方法将暂存的结果导出到本地文件。
        :return: 训练后的模型。
        """
        start_time = time.time()
        predictor_name = predictor.__name__  # 模型名称
        if show_result:
            cprint(f"开始训练和预测 {predictor_name} 模型...", text_color="白色", end='\n')
        # 模型实例化、训练、预测
        model = predictor(**predictor_parameters)
        if is_normalization:
            model.fit(self.train_feature_norm, self.train_target_norm)
            train_predict = self.normalization_target.denorm(model.predict(self.train_feature_norm))
            valid_predict = self.normalization_target.denorm(model.predict(self.valid_feature_norm))
            test_predict = self.normalization_target.denorm(model.predict(self.test_feature_norm))
        else:
            model.fit(self.train_feature, self.train_target)
            train_predict = model.predict(self.train_feature)
            valid_predict = model.predict(self.valid_feature)
            test_predict = model.predict(self.test_feature)
        # 计算模型的评估指标
        train_metrics = calculate_metrics(self.train_target, train_predict)
        valid_metrics = calculate_metrics(self.valid_target, valid_predict)
        test_metrics = calculate_metrics(self.test_target, test_predict)
        # 绘制和保存预测图像
        if show_figure or save_figure:
            title = f"{predictor_name} 训练集预测"
            filename = title + '.' + figure_type if save_figure else None
            self.drawer.draw([self.train_target, train_predict], title=title, legend=['true', predictor_name],
                             filename=filename, show=show_figure)
            title = f"{predictor_name} 验证集预测"
            filename = title + '.' + figure_type if save_figure else None
            self.drawer.draw([self.valid_target, valid_predict], title=title, legend=['true', predictor_name],
                             filename=filename, show=show_figure)
            title = f"{predictor_name} 测试集预测"
            filename = title + '.' + figure_type if save_figure else None
            self.drawer.draw([self.test_target, test_predict], title=title, legend=['true', predictor_name],
                             filename=filename, show=show_figure)
        end_time = time.time()
        if show_result:
            cprint(f"{predictor_name} 模型训练集评估指标：", text_color="青色", end='\n', **train_metrics)
            cprint(f"{predictor_name} 模型验证集评估指标：", text_color="紫色", end='\n', **valid_metrics)
            cprint(f"{predictor_name} 模型测试集评估指标：", text_color="黄色", end='\n', **test_metrics)
            cprint(f"{predictor_name} 模型训练和预测结束，用时 {end_time - start_time} 秒。", text_color="蓝色")
        if save_result:
            self.writer.write_model(model, f"{predictor_name}.pkl")
            self.writer.add_data(data_df=pd.DataFrame(train_predict, columns=[predictor_name]), axis=1, key="train predict")
            self.writer.add_data(data_df=pd.DataFrame(valid_predict, columns=[predictor_name]), axis=1, key="valid predict")
            self.writer.add_data(data_df=pd.DataFrame(test_predict, columns=[predictor_name]), axis=1, key="test predict")
            self.writer.add_metrics(metrics_df=pd.DataFrame(train_metrics, index=[predictor_name]), axis=0, key="train metrics")
            self.writer.add_metrics(metrics_df=pd.DataFrame(valid_metrics, index=[predictor_name]), axis=0, key="valid metrics")
            self.writer.add_metrics(metrics_df=pd.DataFrame(test_metrics, index=[predictor_name]), axis=0, key="test metrics")
            self.writer.add_log(f"{predictor_name} 模型训续和预测用时 {end_time - start_time} 秒。", key="log")
            self.writer.add_parameters(content_text=f"{predictor_name} 模型参数", content_dict=predictor_parameters,
                                       key="predictor parameters")
        if to_save:
            self.writer.write()
        return model

    def persistence(self, *, figure_type='svg', save_result=True, save_figure=True, show_result=True, show_figure=False,
                    to_save=True):
        """
        计算 Persistence 模型的预测结果和评估指标。
        :param figure_type: 图像保存类型，默认为 'svg'。
        :param save_result: 是否保存结果，默认为 True。
        :param save_figure: 是否保存图像，默认为 True。
        :param show_result: 是否显示结果，默认为 True。
        :param show_figure: 是否显示图像，默认为 False。
        :param to_save: 是否将保存的结果导出到本地文件，默认为 True。如果为 False，则只是将 save_result 的结果进行暂存，不导出到本地文件。
                        可以通过 predictors.writer.write() 方法将暂存的结果导出到本地文件。
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
            self.drawer.draw([self.train_target, train_predict], title=title, legend=['true', "Persistence"],
                             filename=filename, show=show_figure)
            title = f"Persistence 验证集预测"
            filename = title + '.' + figure_type if save_figure else None
            self.drawer.draw([self.valid_target, valid_predict], title=title, legend=['true', "Persistence"],
                             filename=filename, show=show_figure)
            title = f"Persistence 测试集预测"
            filename = title + '.' + figure_type if save_figure else None
            self.drawer.draw([self.test_target, test_predict], title=title, legend=['true', "Persistence"],
                             filename=filename, show=show_figure)
        end_time = time.time()
        if show_result:
            cprint(f"Persistence 模型训练集评估指标：", text_color="青色", end='\n', **train_metrics)
            cprint(f"Persistence 模型验证集评估指标：", text_color="紫色", end='\n', **valid_metrics)
            cprint(f"Persistence 模型测试集评估指标：", text_color="黄色", end='\n', **test_metrics)
            cprint(f"Persistence 模型训练和预测结束，用时 {end_time - start_time} 秒。", text_color="蓝色")
        if save_result:
            self.writer.add_data(data_df=pd.DataFrame(train_predict, columns=["Persistence"]), axis=1, key="train predict")
            self.writer.add_data(data_df=pd.DataFrame(valid_predict, columns=["Persistence"]), axis=1, key="valid predict")
            self.writer.add_data(data_df=pd.DataFrame(test_predict, columns=["Persistence"]), axis=1, key="test predict")
            self.writer.add_metrics(metrics_df=pd.DataFrame(train_metrics, index=["Persistence"]), axis=0, key="train metrics")
            self.writer.add_metrics(metrics_df=pd.DataFrame(valid_metrics, index=["Persistence"]), axis=0, key="valid metrics")
            self.writer.add_metrics(metrics_df=pd.DataFrame(test_metrics, index=["Persistence"]), axis=0, key="test metrics")
            self.writer.add_log(f"Persistence 模型训续和预测用时 {end_time - start_time} 秒。", key="log")
        if to_save:
            self.writer.write()
        return None
