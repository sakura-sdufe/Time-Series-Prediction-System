# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation-Project
# @Time     : 2024/11/20 15:24
# @Author   : 张浩
# @FileName : writer.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import os
import pickle
import pandas as pd


class Writer:
    def __init__(self, save_dir):
        """
        写入指标、参数、数据、日志；保存模型
        :param save_dir: 保存的根目录
        """
        # 指标、参数、数据的索引
        self.metric_key_index, self.parameter_key_index, self.data_key_index = 0, 0, 0
        self.metric = dict() # 指标（key 为关键词，value:pd.DataFrame 为所存储的指标）
        self.parameters = dict()  # 参数（key 为关键词，value:str 为所存储的参数）
        self.data = dict()  # 数据（key 为关键词，value:pd.DataFrame 为所存储的数据）
        self.log = dict()  # 日志（key 为关键词，value:str 为所存储的日志）

        self.save_documents_path = os.path.join(save_dir, 'documents')  # 保存的文档路径
        self.save_models_path = os.path.join(save_dir, 'models_trained')  # 保存的模型路径
        self.save_results_path = os.path.join(save_dir, 'results')  # 保存的结果路径

        if not os.path.exists(self.save_documents_path):
            os.makedirs(self.save_documents_path)
        if not os.path.exists(self.save_models_path):
            os.makedirs(self.save_models_path)
        if not os.path.exists(self.save_results_path):
            os.makedirs(self.save_results_path)

    def check_key(self, new_key, ignore_dict):
        """
        检查新的 key 是否已经存在于除 ignore_dict 之外的所有字典中。检查范围：self.metric, self.parameters, self.data, self.log。
        如果存在相同的 key，则返回 True，否则返回 False。当返回结果为 True 时，请不要使用该 key。
        """
        all_dict = [self.metric, self.parameters, self.data, self.log]
        check_result = any([new_key in d for d in all_dict if d is not ignore_dict])
        return check_result

    def add_metrics(self, metrics_df, axis, key=None):
        """
        添加指标到 self.metric 中。
        :param metrics_df: pd.DataFrame 类型，需要添加的指标结果。
        :param axis: 指定拼接的方向，0 为纵向拼接，1 为横向拼接。
        :param key: 指定的 key，如果为 None，则自动生成一个 key。
                    如果之前已经存在相同的 key，则将新的指标结果追加到原有的指标结果中；否则，新建一个 key。
        :return: None
        """
        if key is None:
            key = f"metric_{self.metric_key_index}"
            self.metric_key_index += 1
        if self.check_key(key, self.metric):
            raise ValueError("指标名称不能与参数名称或者数据名称相同！")
        elif key not in self.metric:
            self.metric[key] = metrics_df
        else:
            self.metric[key] = pd.concat([self.metric[key], metrics_df], axis=axis)

    def add_parameters(self, content_text, content_dict, key=None, sep='\n', end='\n\n'):
        """
        添加参数到 self.parameters 中。
        :param content_text: 参数的文本内容。
        :param content_dict: 参数的字典内容。
        :param key: 指定的 key，如果为 None，则自动生成一个 key。
                    如果之前已经存在相同的 key，则将新的参数追加到原有的参数中；否则，新建一个 key。
        :param sep: 参数文本和参数字典、参数字典之间的分隔符。
        :param end: 每次调用 add_parameters 时，在最后添加的分隔符。
        :return: None
        """
        if key is None:
            key = f"parameter_{self.parameter_key_index}"
            self.parameter_key_index += 1
        # 当前文本情况
        if content_text and content_dict:
            current_content = content_text + sep + \
                              sep.join([f"\t{(key+':').ljust(30, ' ')} {value}" for key, value in content_dict.items()])
        elif content_text and not content_dict:
            current_content = content_text
        elif not content_text and content_dict:
            current_content = sep.join([f"\t{(key+':').ljust(30, ' ')} {value}" for key, value in content_dict.items()])
        else:
            raise ValueError("content_text 和 content_dict 不能同时为空！")
        # 添加内容
        if self.check_key(key, self.parameters):
            raise ValueError("参数名称不能与指标名称或者数据名称相同！")
        elif key not in self.parameters:
            self.parameters[key] = current_content
        else:
            self.parameters[key] = self.parameters[key] + end + current_content

    def add_data(self, data_df, axis, key=None):
        """
        添加数据到 self.data 中。
        :param data_df: pd.DataFrame 类型，需要添加的数据。
        :param axis: 指定拼接的方向，0 为纵向拼接，1 为横向拼接。
        :param key: 指定的 key，如果为 None，则自动生成一个 key。
                    如果之前已经存在相同的 key，则将新的数据追加到原有的数据中；否则，新建一个 key。
        :return: None
        """
        if key is None:
            key = f"data_{self.data_key_index}"
            self.data_key_index += 1
        if self.check_key(key, self.data):
            raise ValueError("数据名称不能与指标名称或者参数名称相同！")
        elif key not in self.data:
            self.data[key] = data_df
        else:
            self.data[key] = pd.concat([self.data[key], data_df], axis=axis)

    def add_log(self, context, key=None, end='\n'):
        """
        添加日志到 self.log 中。
        :param context: 日志内容。
        :param key: 指定的 key，如果为 None，则自动生成一个 key。
                    如果之前已经存在相同的 key，则将新的日志追加到原有的日志中；否则，新建一个 key。
        :param end: 每次调用 add_log 时，在最后添加的分隔符。
        :return: None
        """
        if key is None:
            key = f"log_{self.data_key_index}"
            self.data_key_index += 1
        if self.check_key(key, self.log):
            raise ValueError("数据名称不能与指标名称或者参数名称或者数据名称相同！")
        elif key not in self.log:
            self.log[key] = context
        else:
            self.log[key] = self.log[key] + end + context


    def write_model(self, model, filename):
        """
        保存模型。模型保存目录：self.save_models_path
        :param model: 模型
        :param filename: 文件名
        :return: None
        """
        with open(os.path.join(self.save_models_path, filename), 'wb') as f:
            pickle.dump(model, f)

    def write(self):
        """
        将暂存的指标、参数、数据、日志导出到指定的文件夹中。
        - 指标保存目录：self.save_results_path
        - 数据保存目录：self.save_results_path
        - 参数保存目录：self.save_documents_path
        - 日志保存目录：self.save_documents_path
        :return: None
        """
        # 写入指标
        for key, value in self.metric.items():
            value.to_excel(os.path.join(self.save_results_path, f"{key}.xlsx"), index=True)
        # 写入数据
        for key, value in self.data.items():
            value.to_csv(os.path.join(self.save_results_path, f"{key}.csv"), index=True)
        # 写入参数
        for key, value in self.parameters.items():
            with open(os.path.join(self.save_documents_path, f"{key}.txt"), 'w', encoding='utf-8') as f:
                f.write(value)
        # 写入日志
        for key, value in self.log.items():
            with open(os.path.join(self.save_documents_path, f"{key}.txt"), 'w', encoding='utf-8') as f:
                f.write(value)
