# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :40_class_ImageNet
# @Time     :2023/6/23 20:25
# @Author   :张浩
# @FileName :xc_Timer.py
# @Software :PyCharm

计时器类Timer，可以记录开始时间，结束时间；可以计算平均时间，总时间，累加和
-------------------------------------------------
"""

import numpy as np
import time


# 定义时间类Timer，可以记录开始时间，结束时间；可以计算平均时间，总时间，累加和
class Timer:
    def __init__(self):
        # 用于存放每次开始和暂停的时间，类似停止计时并保存，下次开启一个新的计时器
        self.times = []
        # 初始化self.tik
        self.tik = time.perf_counter()  # float类型。精确的计时器，用于记录时间

    def start(self):
        """开始计时"""
        self.tik = time.perf_counter()

    def stop(self):
        """终止计时，保存计时时间 并且返回该次计时时间"""
        self.times.append(time.perf_counter() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累加和"""
        return np.array(self.times).cumsum().tolist()
