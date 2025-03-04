# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation
# @Time     : 2024/12/24 19:30
# @Author   : 张浩
# @FileName : RNNs.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import time
import torch
import torch.nn as nn

from utils.device_gpu import try_gpu
from Base.Base import ModelBase


class RNNModelBase(ModelBase):
    def __init__(self, rnn_layer, output_size):
        """
        定义 RNNs 系列模型的基模型。
        :param rnn_layer: 声明所需的 RNNs 系列模型的层。可以为 nn.RNN, nn.LSTM, nn.GRU
        :param output_size: 输出层的特征维度，即输出节点个数
        note: DataLoader类型数据满足 X 的维度：[batch_size, steps, inputs_size]，Y的维度：[batch_size]。
        """
        super(RNNModelBase, self).__init__()
        self.rnn_layer = rnn_layer
        self.output_size = output_size
        self.hidden_size = self.rnn_layer.hidden_size
        self.direction_size = 2 if self.rnn_layer.bidirectional else 1

        # 定义 Linear 模型，通过最后一个隐藏层的输出结果得到下一时间步的预测结果。
        self.output_layer = nn.Linear(self.hidden_size*self.direction_size, self.output_size)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, inputs, hidden_state):
        """
        RNNs 系列模型向前计算
        :param inputs: 输入张量。维度为：[num_predictors, batch_size, input_size]。
        :param hidden_state: 隐藏层初始状态。
            如果为 RNN 和 GRU，那么维度为：[direction_size*num_layers, batch_size, hidden_size]。
            如果为 LSTM，那么为 Tuple：(h0, c0)，其中 h0 的维度为：[direction_size*num_layers, batch_size, proj_size]；
                c0 的维度为：[direction_size*num_layers, batch_size, hidden_size]。
            Note: proj_size 为 LSTM 的输出特征维度，hidden_size 为 LSTM 的隐藏层特征维度。如果没有设置 proj_size，那么 proj_size=hidden_size。
        :return: output_size, hidden_state
            output_size: 输出张量。维度为：[batch_size, output_size]。（仅保留最后一个时间步的输出结果）
            hidden_state: 隐藏层最后一个时间步的输出结果。
                如果为 RNN 和 GRU，那么维度为：[direction_size*num_layers, batch_size, hidden_size]。
                如果为 LSTM，那么为 Tuple：(hn, cn)，其中 hn 的维度为：[direction_size*num_layers, batch_size, proj_size]；
                    cn 的维度为：[direction_size*num_layers, batch_size, hidden_size]。
        """
        Y, hidden_state = self.rnn_layer(inputs, hidden_state)
        Y_least = Y[-1, :, :]  # 仅保留最后一个时间步的输出结果
        output_least = self.output_layer(self.relu(Y_least))
        return output_least, hidden_state

    def begin_state(self, batch_size, device):
        """
        初始化 RNNs 隐藏层状态
        :param batch_size: 批量大小。
        :param device: 生成的隐藏层初始状态所存在的设备。
        :return: hidden_state
            hidden_state: 隐藏层初始状态。如果为 RNN 和 GRU，那么维度为：[direction_size*num_layers, batch_size, hidden_size]。
                如果为 LSTM，那么为 Tuple：(h0, c0)，其中 h0 的维度为：[direction_size*num_layers, batch_size, proj_size]；
                c0 的维度为：[direction_size*num_layers, batch_size, hidden_size]。
        """
        if not isinstance(self.rnn_layer, nn.LSTM):
            return torch.zeros((self.direction_size * self.rnn_layer.num_layers, batch_size, self.hidden_size), device=device)
        else:
            return (torch.zeros((self.direction_size * self.rnn_layer.num_layers, batch_size, self.hidden_size), device=device),
                    torch.zeros((self.direction_size * self.rnn_layer.num_layers, batch_size, self.hidden_size), device=device))

    def train_epoch(self, dataloader, criterion, optimizer, device=None, clip_norm=None):
        """
        训练 RNNs 系列模型一个迭代周期
        :param dataloader: DataLoader 数据加载器。
        :param criterion: 损失函数
        :param optimizer: 优化器
        :param device: 运算设备。默认为 None，如果有 GPU 则使用 GPU 进行运算，否则使用 CPU 进行运算。
        :param clip_norm: 梯度裁剪时最大的梯度值（L2）。默认为 None，即不进行梯度裁剪。
        :return: None
        note:
            1. 控制台输出代码部分，用户在继承时可自行选择是否实现。我强烈推荐实现，因为这样可以更好地了解模型的训练情况。
            2. 在本方法中得到的损失值结果是用于训练的训练集的结果。如果 sample_gap 参数不为 1，
                那么用于训练的训练集的结果和用于评估的训练集的结果大概率是不同的，因为两个 Dataloader 的样本个数不同。
                可以理解为：trainer: dataloader 是 evaler: dataloader 的子集（下采样）。
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
        for X, Y in dataloader:
            # 处理隐藏层状态
            batch_size = X.shape[0]
            state = self.begin_state(batch_size=batch_size, device=device)
            # 将state从上一个计算图中剔除，防止之前计算图的梯度影响下一次的梯度更新。
            if isinstance(self, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()

            X, Y = X.to(device), Y.to(device)  # 将数据转移到设备上
            X = X.permute(1, 0, 2)  # 将输入张量的维度调整为：[num_predictors, batch_size, input_size]
            Y = Y.unsqueeze(dim=1)  # 将目标张量的维度调整为：[batch_size, output_size]
            Y_hat, hidden_state = self(X, state)
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
        预测 RNNs 系列模型
        :param dataloader: DataLoader 数据加载器。
        :param device: 运算设备。默认为 None，如果有 GPU 则使用 GPU 进行运算，否则使用 CPU 进行运算。
        :return: 预测结果。维度为：torch.Size([batch_size, outputs_node])
        """
        if device is None:
            device = try_gpu()
        self.to_device(device)
        self.eval()
        predict_list = []
        with torch.no_grad():
            for X, Y in dataloader:
                batch_size = X.shape[0]
                state = self.begin_state(batch_size=batch_size, device=device)
                # 将state从上一个计算图中剔除，防止之前计算图的梯度影响下一次的梯度更新。（是否可以删除？）
                if isinstance(self, nn.Module) and not isinstance(state, tuple):
                    state.detach_()
                else:
                    for s in state:
                        s.detach_()
                # RNNs 开始推理
                X, Y = X.to(device), Y.to(device)
                X = X.permute(1, 0, 2)  # 将输入张量的维度调整为：[num_predictors, batch_size, input_size]
                Y_hat, hidden_state = self(X, state)
                predict_list.append(Y_hat.cpu())
        predict_result = torch.cat(predict_list, dim=0)
        return predict_result


class RNNModel(RNNModelBase):
    def __init__(self, input_size, hidden_size, output_size, *, num_layers=1, bidirectional=False, **kwargs):
        """
        定义 RNN 模型
        :param input_size: 输入层的特征维度，即输入节点个数。
        :param hidden_size: 隐藏层的特征维度，即隐藏节点个数。
        :param output_size: 输出层的特征维度，即输出节点个数。
        :param num_layers: RNN 层的层数。默认为 1。
        :param bidirectional: 是否为双向 RNN。默认为 False。
        :param kwargs: 其他参数，具体参数见 nn.RNN 函数。
        """
        rnn_layer = nn.RNN(input_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, **kwargs)
        super(RNNModel, self).__init__(rnn_layer, output_size)


class LSTMModel(RNNModelBase):
    def __init__(self, input_size, hidden_size, output_size, *, num_layers=1, bidirectional=False, **kwargs):
        """
        定义 LSTM 模型
        :param input_size: 输入层的特征维度，即输入节点个数。
        :param hidden_size: 隐藏层的特征维度，即隐藏节点个数。
        :param output_size: 输出层的特征维度，即输出节点个数。
        :param num_layers: LSTM 层的层数。默认为 1。
        :param bidirectional: 是否为双向 LSTM。默认为 False。
        :param kwargs: 其他参数，具体参数见 nn.LSTM 函数。
        """
        lstm_layer = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, **kwargs)
        super(LSTMModel, self).__init__(lstm_layer, output_size)


class GRUModel(RNNModelBase):
    def __init__(self, input_size, hidden_size, output_size, *, num_layers=1, bidirectional=False, **kwargs):
        """
        定义 GRU 模型
        :param input_size: 输入层的特征维度，即输入节点个数。
        :param hidden_size: 隐藏层的特征维度，即隐藏节点个数。
        :param output_size: 输出层的特征维度，即输出节点个数。
        :param num_layers: GRU 层的层数。默认为 1。
        :param bidirectional: 是否为双向 GRU。默认为 False。
        :param kwargs: 其他参数，具体参数见 nn.GRU 函数。
        """
        gru_layer = nn.GRU(input_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, **kwargs)
        super(GRUModel, self).__init__(gru_layer, output_size)
