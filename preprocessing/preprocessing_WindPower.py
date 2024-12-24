# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation-Project
# @Time     : 2024/11/10 11:37
# @Author   : 张浩
# @FileName : preprocessing_WindPower.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import os
import hashlib
import pandas as pd
from datetime import datetime, timedelta


def hash_file(file_path, hash_algorithm='md5', chunk_size=4096):
    """
    对大文件进行哈希，返回哈希值。
    """
    try:
        hasher = hashlib.new(hash_algorithm)
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except (FileNotFoundError, ValueError) as e:
        return f"Error: {e}"


def read_file(path, **kwargs):
    """
    读取 path 路径下的文件，并返回。
    :param path: 所需要读取的文件路径，支持 csv、xlsx、xls 格式。
    :param kwargs: 传递给 pd.read_csv 或 pd.read_excel 的参数。
    :return: None
    """
    if path.endswith('.csv'):
        return pd.read_csv(path, **kwargs)
    elif path.endswith('.xlsx') or path.endswith('.xls'):
        return pd.read_excel(path, **kwargs)
    else:
        raise ValueError('仅支持读取 csv、xlsx、xls 格式的文件。')


def pre_time_series(value, save_dir='../Datasets/Vestas V52 Wind Turbine', is_save=True):
    """
    预处理时间序列数据，并保存处理后的数据。
    """
    def locate_time(time):
        """二分法定位时间位置"""
        time = datetime.strptime(time, '%Y/%m/%d %H:%M')
        length = len(value["Timestamps"])
        left = 0
        right = length
        while left < right:
            mid = (left + right) // 2
            mid_time = datetime.strptime(value["Timestamps"][mid], '%Y/%m/%d %H:%M')
            if mid_time == time:
                return mid
            elif mid_time < time:
                left = mid + 1
            elif mid_time > time:
                right = mid

    # Step 1：删除 'GenTemp'（发电机温度（非激活999））列，因为这一列基本上都是 999。
    value.drop('GenTemp', axis=1, inplace=True)
    # Step 2：捕获异常值，对异常值直接删除。
    # 异常值检测标准：'WindSpeed'为0，'MaxPower'为-3276.8，'MinPower'为3276.7。
    for i in range(len(value)):
        if value['WindSpeed'][i] == 0 and value['MaxPower'][i] == -3276.8 and value['MinPower'][i] == 3276.7:
            value.drop(i, axis=0, inplace=True)
    value.reset_index(drop=True, inplace=True)
    # Step 3：将 value 分成两个 DataFrame，分别为变速箱更换前和变速箱更换后。
    # 2018年10月04日11:50 至 2019年07月28日16:00 期间发生了变速箱更换导致输出功率不稳定。
    start_time, end_time = '2018/10/4 11:50', '2019/7/28 16:00'
    start_index, end_index = locate_time(start_time), locate_time(end_time)
    value_part1 = value.iloc[:start_index, :].reset_index(drop=True)
    value_part2 = value.iloc[end_index - 1:, :].reset_index(drop=True)
    # Step 5：保存预处理结果和 hash 值。
    if is_save:
        preprocessing_path_part1 = os.path.join(save_dir, 'VestasV52 preprocessing part1.csv')
        preprocessing_path_part2 = os.path.join(save_dir, 'VestasV52 preprocessing part2.csv')
        hash_path = os.path.join(save_dir, 'VestasV52 preprocessing hash.txt')
        value_part1.to_csv(preprocessing_path_part1, index=False)
        value_part2.to_csv(preprocessing_path_part2, index=False)
        hash_part1 = hash_file(preprocessing_path_part1)
        hash_part2 = hash_file(preprocessing_path_part2)
        with open(hash_path, 'w+') as f:
            f.write(f"VestasV52 preprocessing part1: {hash_part1}\nVestasV52 preprocessing part2: {hash_part2}")
        return value_part1, value_part2


def convert_time_gap(value_tuple, save_dir='../Datasets/Vestas V52 Wind Turbine', gap_min=30):
    value_gap_list = []
    for i in range(len(value_tuple)):
        value = value_tuple[i]
        value["Timestamps_datetime"] = pd.to_datetime(value["Timestamps"])
        value.reset_index(drop=True, inplace=True)
        # 转换时间间隔
        selected_rows = [0]
        for ind in range(1, len(value)):
            if (value.loc[ind, "Timestamps_datetime"] - value.loc[selected_rows[-1], "Timestamps_datetime"]).total_seconds() >= gap_min*60:
                selected_rows.append(ind)
        value_gap = value.loc[selected_rows, :].reset_index(drop=True)
        value_gap.drop("Timestamps_datetime", axis=1, inplace=True)
        value_gap_list.append(value_gap)

        # 保存转换后的数据和 hash 值
        gap_path = os.path.join(save_dir, f'VestasV52 convert gap={gap_min} part{i+1}.csv')
        hash_path = os.path.join(save_dir, f'VestasV52 convert gap={gap_min} hash.txt')
        value_gap.to_csv(gap_path, index=False)
        hash_gap = hash_file(gap_path)
        with open(hash_path, 'a+') as f:
            f.write(f"VestasV52 convert gap={gap_min} part{i+1}: {hash_gap}\n")
    return value_gap_list
