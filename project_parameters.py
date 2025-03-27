# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation
# @Time     : 2025/1/10 16:15
# @Author   : 张浩
# @FileName : project_parameters.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import os
import datetime


class ProjectParameters:
    def __init__(self):
        # 保存设置
        self.file_name = 'VestasV52Gap60.csv'  # 仅支持 csv xlsx xls 格式。
        self.delete_dir = False  # 是否删除 Writer 类生成的目录。（该参数仅在 Windows 系统上测试）
        self.save_mode = 'a+'  # Writer 类写入本地模式，可选 'w+' 和 'a+'，默认为 'w+'。默认为 'a+'，

        # 数据基础信息
        self.dataset_name = os.path.splitext(self.file_name)[0]  # 数据集名称（去除后缀名）
        now = datetime.datetime.now()
        formatted_time = now.strftime("%Y-%m-%d")
        # formatted_time = now.strftime("%Y-%m-%d %H,%M,%S")
        # 保存预测器结果的目录
        self.save_predictor_dir = os.path.join(
            r'./result',
            f"{self.dataset_name} [Predictor] (time={formatted_time})"
            # f"{self.dataset_name} [Predictor] (time=2025-03-23)"
        )
        # 读取预测器结果和保存集成器结果的目录
        self.read_predictor_dir = self.save_predictor_dir
        self.save_ensemble_dir = os.path.join(
            r'./result',
            f"{self.dataset_name} [Ensemble] (time={formatted_time})"
            # f"{self.dataset_name} [Ensemble] (time=2025-03-23)"
        )

    def __getitem__(self, item):
        return getattr(self, item)

    def __iter__(self):
        return iter(self.__dict__.items())

    def items(self):
        return self.__dict__.items()
