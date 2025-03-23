# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation
# @Time     : 2025/3/21 21:45
# @Author   : 张浩
# @FileName : read_data.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import pandas as pd


def read_file(file_path: str, **kwargs) -> pd.DataFrame:
    """
    读取文件（仅支持 xlsx xls csv 文件）
    :param file_path: 文件路径
    :param kwargs: 其他参数
    :return: DataFrame
    """
    if file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path, **kwargs)
    elif file_path.endswith('.xls'):
        data = pd.read_excel(file_path, **kwargs)
    elif file_path.endswith('.csv'):
        data = pd.read_csv(file_path, **kwargs)
    else:
        raise ValueError("文件格式不支持！")
    return data
