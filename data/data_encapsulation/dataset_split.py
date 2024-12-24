# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation-Project
# @Time     : 2024/11/13 13:44
# @Author   : 张浩
# @FileName : dataset_split.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import warnings


def split_dataset(feature, target, start_position, end_position):
    """
    将 DataFrame 或者 array 类型数据进行切分（按照 dim=0 维度切分），切分的位置由 start_position 和 end_position 控制。
    :param feature: 特征数据。数据类型为 DataFrame 或者 array。
    :param target: 目标数据。数据类型为 DataFrame 或者 array。
    :param start_position: 切分的起始位置。如果是介于 0 和 1 之间的小数，则表示按照比例切分。如果是大于 1 的整数，则表示按照数量切分。
    :param end_position: 切分的结束位置。如果是介于 0 和 1 之间的小数，则表示按照比例切分。如果是大于 1 的整数，则表示按照数量切分。
        Note: feature 和 target 也可以接受其他可以被 len 函数调用和可以切片的数据类型。
            当 start_position 和 end_position 取值为 0.0 和 1.0 时，表示按照比例切分（输入为 float 类型数据按比例切分）；
            当 start_position 和 end_position 取值为 0 和 1 时，表示按照数量切分（输入为 int 类型数据按数量切分）。
    :return: feature_split, target_split 分别表示指定范围内的特征数据和目标数据。
    """
    assert type(start_position) == type(end_position), "start_position 和 end_position 的数据类型必须相同！"
    assert (start_position>=0) and (end_position>=0), "start_position 和 end_position 的取值必须大于等于 0！"
    assert start_position < end_position, "start_position 的取值必须小于 end_position！"
    if len(feature) != len(target):
        warnings.warn(
            "start_position 和 end_position 在 dim=0 维度上长度不相同；如果你使用的时按比例划分，那么使用 feature 的长度。",
            category=UserWarning)

    if isinstance(start_position, float):  # 按照比例切分
        if (start_position<=1) and (end_position<=1):
            start_position = round(start_position * len(feature))
            end_position = round(end_position * len(feature))
        else:
            raise ValueError("当 start_position 和 end_position 按照比例划分时，取值必须在 0 和 1 之间！")

    assert end_position <= len(feature), "end_position 的取值不能大于特征数据的长度！"
    return feature[start_position:end_position], target[start_position:end_position]
