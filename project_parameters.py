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
from data.data_parameters import DataParameters


class ProjectParameters:
    def __init__(self):
        # 保存设置
        self.delete_dir = False  # 是否删除 Writer 类生成的目录。（该参数仅在 Windows 系统上测试）
        self.write_mode = 'a+'  # Writer 类写入本地模式，可选 'w+' 和 'a+'，默认为 'w+'。
        parameters_data = DataParameters()  # 获取 DataSplit 模块的超参数

        # 数据基础信息
        self.save_dir = os.path.join(
            r'./result', ', '.join([f"{k}={v}" for k, v in parameters_data['feature_selection'].items()])
        )
        # self.save_dir = r'./result/测试'  # 模型保存路径

    def __getitem__(self, item):
        return getattr(self, item)

    def __iter__(self):
        return iter(self.__dict__.items())

    def items(self):
        return self.__dict__.items()
