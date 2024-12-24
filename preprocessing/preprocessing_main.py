# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation-Project
# @Time     : 2024/11/11 21:16
# @Author   : 张浩
# @FileName : preprocessing_main.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import os
import pandas as pd
from preprocessing.preprocessing_WindPower import read_file, pre_time_series, convert_time_gap, hash_file


def read_validation_file(data_file_name, hash_file_name):
    # 处理地址和路径
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件绝对路径
    dataset_rel_dir = r"Datasets/Vestas V52 Wind Turbine"
    dataset_abs_dir = os.path.join(os.path.dirname(current_dir), dataset_rel_dir)
    data_path = os.path.join(dataset_abs_dir, data_file_name)
    hash_path = os.path.join(dataset_abs_dir, hash_file_name)
    # 读取文件（判断 .csv .xlsx .xls）
    if data_file_name.endswith('.csv'):
        data = pd.read_csv(data_path)
    elif data_file_name.endswith('.xlsx') or data_file_name.endswith('.xls'):
        data = pd.read_excel(data_path)
    else:
        raise ValueError('仅支持读取 csv、xlsx、xls 格式的文件。')
    # 读取哈希值
    hash_value = None
    with open(hash_path, 'r') as f:
        hash_content = f.read()
        hash_list = hash_content.strip().split('\n')
        for hash_element in hash_list:
            if data_file_name.split('.')[-2] in hash_element:
                hash_value = hash_element.split(': ')[-1].strip()
                break
    # 计算哈希值 并 对比是否相同
    new_hash_value = hash_file(data_path)
    if hash_value != new_hash_value:
        raise ValueError('文件哈希值不一致，请检查文件是否被篡改。')
    return data


def preprocessing_main():
    # 处理地址和路径
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件绝对路径
    dataset_rel_dir = r"Datasets/Vestas V52 Wind Turbine"
    dataset_abs_dir = os.path.join(os.path.dirname(current_dir), dataset_rel_dir)
    origin_data_path = os.path.join(dataset_abs_dir, r"VestasV52_10_min_raw_SCADA_DkIT 30_Jan2006-12_Mar2020.csv")
    # 读取文件
    origin_data = read_file(origin_data_path)
    # 预处理数据：删除特征、异常值处理、数据集分割等操作。
    value_preprocessing_tuple = pre_time_series(origin_data, save_dir=dataset_abs_dir, is_save=True)
    # 时间间隔转换
    value_gap_tuple = convert_time_gap(value_preprocessing_tuple, save_dir=dataset_abs_dir, gap_min=60)


if __name__ == "__main__":
    preprocessing_main()
