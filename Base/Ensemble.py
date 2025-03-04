# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation
# @Time     : 2025/2/27 21:14
# @Author   : 张浩
# @FileName : Ensemble.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import time
import torch
from torch import nn

from utils.device_gpu import try_gpu
from Base.Base import ModelBase


class EnsembleBase(ModelBase):
    def __init__(self, **kwargs):
        """
        定义深度学习模型的基模型（用于集成模块）。需要重写 __init__ 和 forward 方法。
        若子类中 forward 方法不满足标记的条件，那么需要重写 train_epoch 和 predict 方法。
        note:
            1. DataLoader类型数据满足 X 的维度：[batch_size, input_size]，Y的维度：[batch_size]。
            2. _trainer 表示用于训练的 DataLoader 数据加载器，_evaler 表示用于评估的 DataLoader 数据加载器。主要差异是：
                _trainer 中的 shuffle=True 和 batch_size=predictor_batch_size；
                _evaler 中的 shuffle=False 和 batch_size=eval_batch_size。
            3. input_size 和 time_step 参数会自动解析为输入特征的维度和时间步数。
                input_size 是定义模型必须参数，但是模型参数无需指定。time_step 根据模型需求选择是否引入。
                input_size 为必须参数，即使没有使用也需要在模型中定义。
        """
        super(EnsembleBase, self).__init__(**kwargs)

    def forward(self, **kwargs):
        """
        深度学习模型向前计算，需要在子类中重写。
        note:
            1. 输入应当是一个 Tensor，且维度应为：[batch_size, input_size]；
            2. 输出应当是一个 Tensor，且维度应为：[batch_size, output_size]。
        """
        raise NotImplementedError

    def train_epoch(self, dataloader, criterion, optimizer, device=None, clip_norm=None):
        """
        深度学习模型训练过程中一个迭代周期
        :param dataloader: DataLoader 数据加载器。
        :param criterion: 损失函数
        :param optimizer: 优化器
        :param device: 运算设备。默认为 None，如果有 GPU 则使用 GPU 进行运算，否则使用 CPU 进行运算。
        :param clip_norm: 梯度裁剪时最大的梯度值（L2）。默认为 None，即不进行梯度裁剪。
        :return: None
        note:
            1. 控制台输出代码部分，用户在继承时可自行选择是否实现。我强烈推荐实现，因为这样可以更好地了解模型的训练情况。
            2. 在本方法中得到的损失值结果是用于训练的训练集的结果。两个结果基本上没有差异，因为这是回归问题，不涉及 sample_gap 参数。
            3. 为什么训练过程中的这些指标非常重要但是却仅输出在控制台的原因是：如果该函数有返回值的话，可能会让用户的后续继承带来困扰。
        """
        if device is None:
            device = try_gpu()
        self.to_device(device)
        self.train()
        #  --> 用于训练的训练集指标计算（用户可自行选择是否输出）
        start_time = time.perf_counter()
        loss_total = 0.0
        sample_total = 0
        # <-- 用于训练的训练集指标计算（用户可自行选择是否输出）
        for X, Y in dataloader:  # X 的维度：[batch_size, input_size]，Y 的维度：[batch_size]
            X, Y = X.to(device), Y.to(device)  # 将数据转移到设备上
            Y = Y.unsqueeze(dim=1)  # 将目标张量的维度调整为：[batch_size, output_size]
            Y_hat = self(X)
            loss_value = criterion(Y_hat, Y.float())
            optimizer.zero_grad()
            loss_value.backward()
            if clip_norm:
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_norm, norm_type=2)
            optimizer.step()
            # --> 用于训练的训练集指标计算（用户可自行选择是否输出）
            loss_total += loss_value.item() * Y.numel()
            sample_total += Y.numel()
            # <-- 用于训练的训练集指标计算（用户可自行选择是否输出）
        # --> 用于训练的训练集指标计算（用户可自行选择是否输出）
        sample_sec = sample_total / (time.perf_counter() - start_time)
        print('-' * 50 + '->')
        print("用于训练的训练集指标：")
        print(f"\t\t训练集损失值：{(loss_total / sample_total):.5f}，"
              f"实际每秒样本数：{sample_sec:.2f}。")  # 实际每秒样本数消除了函数调用的损耗。
        print('<-' + '-' * 50)
        # <-- 用于训练的训练集指标计算（用户可自行选择是否输出）

    def predict(self, dataloader, device=None):
        """
        预测深度学习模型
        :param dataloader: DataLoader 数据加载器。
        :param device: 运算设备。默认为 None，如果有 GPU 则使用 GPU 进行运算，否则使用 CPU 进行运算。
        :return: 预测结果。维度为：torch.Size([batch_size, output_size])
        """
        if device is None:
            device = try_gpu()
        self.to_device(device)
        self.eval()
        predict_list = []
        with torch.no_grad():
            for X, Y in dataloader:  # X 的维度：[batch_size, input_size]，Y 的维度：[batch_size]
                X, Y = X.to(device), Y.to(device)  # 将数据转移到设备上
                Y_hat = self(X)
                predict_list.append(Y_hat.cpu())
        predict_result = torch.cat(predict_list, dim=0)
        return predict_result
