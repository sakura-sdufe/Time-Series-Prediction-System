# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation
# @Time     : 2025/2/28 15:21
# @Author   : 张浩
# @FileName : CNNs.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

from torch import nn
from Base import EnsembleBase, get_activation_fn, get_activation_nn


class CBA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, activation='relu'):
        super(CBA, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = get_activation_fn(activation)

    def forward(self, x):
        return self.activation(self.batch_norm(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), activation='relu', shortcut=True, width_factor=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(out_channels * width_factor)
        self.conv1 = CBA(in_channels, c_, kernel_size=kernel_size[0], stride=1, padding=kernel_size[0]//2,
                         activation=activation)
        self.conv2 = CBA(c_, out_channels, kernel_size=kernel_size[1], stride=1, padding=kernel_size[1]//2,
                         activation=activation)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return self.conv2(self.conv1(x)) + x if self.add else self.conv2(self.conv1(x))


class C3B2H(EnsembleBase):
    def __init__(self, input_size, output_size, bias=True, dropout=0.1, activation='relu'):
        """
        将多个卷积块进行融合。
        :param input_size: 每个特征的维度，在这里指的是预测器的个数。
        :param output_size: 输出的维度。
        :param bias: 多头注意力是否使用偏置。默认值为 True。
        :param dropout: dropout 概率。默认值为 0.0。
        :param activation: 激活函数。默认值为 'relu'，可选值 'relu', 'gelu'。
        """
        super(C3B2H, self).__init__()  # Example: [batch_size, 1, 10]
        self.input_size = input_size
        self.dropout = nn.Dropout(dropout)

        self.conv1 = CBA(in_channels=1, out_channels=2, kernel_size=3, stride=2, padding=1, bias=bias,
                         activation=activation)  # shape: [batch_size, 2, ceil(input_size/2)] Example: [batch_size, 2, 5]
        self.conv2 = CBA(in_channels=2, out_channels=4, kernel_size=3, stride=2, padding=1, bias=bias,
                         activation=activation)  # shape: [batch_size, 4, ceil(input_size/4)] Example: [batch_size, 4, 3]
        self.conv3 = CBA(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1, bias=bias,
                         activation=activation)  # shape: [batch_size, 8, ceil(input_size/8)] Example: [batch_size, 8, 2]

        # shape: [batch_size, 8, ceil(input_size/8)] --> [batch_size, output_size, 1]
        # Example: [batch_size, 8, 2] --> [batch_size, output_size, 1]
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.conv8_4 = nn.Conv1d(in_channels=8, out_channels=4, kernel_size=1, stride=1, padding=0, bias=bias)
        self.conv4_o = nn.Conv1d(in_channels=4, out_channels=output_size, kernel_size=1, stride=1, padding=0, bias=bias)

        self.bottleneck2 = Bottleneck(in_channels=4, out_channels=4, activation=activation, shortcut=True, width_factor=0.5)
        self.bottleneck3 = Bottleneck(in_channels=8, out_channels=8, activation=activation, shortcut=True, width_factor=0.5)

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量，维度为 [batch_size, input_size]。
        :return: 输出张量，维度为 [batch_size, output_size]。
        """
        x = x.unsqueeze(1)  # shape: [batch_size, 1, input_size]
        x = self.dropout(self.bottleneck2(self.conv2(self.conv1(x))))  # shape: [batch_size, 4, ceil(input_size/4)]
        x = self.dropout(self.bottleneck3(self.conv3(x)))  # shape: [batch_size, 8, ceil(input_size/8)]
        x = self.conv4_o(self.conv8_4(self.maxpool(x)))  # shape: [batch_size, output_size, 1]
        return x.squeeze(2)  # shape: [batch_size, output_size]
