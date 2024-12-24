# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : General Functions
# @Time     : 2023/9/19 21:41
# @Author   : 张浩
# @FileName : device_gpu.py
# @Software : PyCharm
# @Function : 返回GPU设备
-------------------------------------------------
"""

import torch


def try_gpu(i=0):
    """
    如果存在GPU，那么返回指定的第 i+1 个 GPU，否则返回CPU。
    Args:
        i: 返回第 i+1 个GPU。i从0开始

    Returns:
        返回GPU或者CPU
    """
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """
    返回所有的GPU，如果没有GPU那么返回CPU。
    Returns:
        返回一个list，其中包含可用的GPU（如果没有GPU那么返回CPU）
    """
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    if devices:
        return devices
    else:
        return [torch.device('cpu')]


if __name__ == "__main__":
    print(try_gpu())
    print(try_all_gpus())
