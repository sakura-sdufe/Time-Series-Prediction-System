# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation
# @Time     : 2025/1/22 20:54
# @Author   : 张浩
# @FileName : Attentions.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

from torch import nn
from Base import EnsembleBase, get_activation_fn, get_activation_nn


class Attention(nn.Module):
    def __init__(self, embed_dim=8, num_heads=4, dropout=0.0, bias=True):
        """
        多头注意力机制。
        :param dropout: dropout 概率。默认值为 0.0。（删的是注意力）
        """
        super(Attention, self).__init__()
        self.attn_weight = None  # 保存注意力权重
        self.attn_output = None  # 保存注意力输出

        self.q_conv = nn.Conv1d(in_channels=1, out_channels=embed_dim, kernel_size=3, stride=1, padding=1)
        self.k_conv = nn.Conv1d(in_channels=1, out_channels=embed_dim, kernel_size=3, stride=1, padding=1)
        self.v_conv = nn.Conv1d(in_channels=1, out_channels=embed_dim, kernel_size=3, stride=1, padding=1)
        self.output_conv = nn.Conv1d(in_channels=embed_dim, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=bias)

        # self.dropout = nn.Dropout(dropout)

    def forward(self, q, k ,v):
        """
        前向传播
        :param q: 查询张量，维度为 [batch_size, dim_q]。
        :param k: 键张量，维度为 [batch_size, dim_k]。
        :param v: 值张量，维度为 [batch_size, dim_k]。
        :return: 输出张量，维度为 [batch_size, dim_q]。
        """
        q, k, v = self.q_conv(q.unsqueeze(1)), self.k_conv(k.unsqueeze(1)), self.v_conv(v.unsqueeze(1))  # shape: [batch_size, embed_dim, dim_q]
        q, k, v = q.permute(2, 0, 1), k.permute(2, 0, 1), v.permute(2, 0, 1)  # shape: [dim_q, batch_size, embed_dim]
        self.attn_output, self.attn_weight = self.attn(q, k, v)  # shape: [dim_q, batch_size, embed_dim]
        self.attn_output = self.output_conv(self.attn_output.permute(1, 2, 0)).squeeze(1)  # shape: [batch_size, dim_q]

        # q, v = q.unsqueeze(2), v.unsqueeze(2)  # shape: [batch_size, dim_q, 1]  [batch_size, dim_k, 1]
        # k = k.unsqueeze(1)  # shape: [batch_size, 1, dim_k]
        # # 计算注意力权重
        # self.attn_weight = torch.softmax(torch.log(torch.pow(q-k, 2)), dim=-1)  # shape: [batch_size, dim_q, dim_k]
        # self.attn_weight = self.dropout(self.attn_weight)
        # # 计算输出
        # self.attn_output = torch.matmul(self.attn_weight, v).squeeze(-1)  # shape: [batch_size, dim_q]
        return self.attn_output, self.attn_weight


class AttentionEnsemble(EnsembleBase):
    def __init__(self, input_size, output_size, *, dropout=0.1, bias=True, activation='relu', **kwargs):
        """
        将多个特征使用多头注意力机制进行融合。注意：这个过程无时序性，也没有位置编码，因此不能用于预测。
        :param input_size: 每个特征的维度，在这里指的是预测器的个数。
        :param output_size: 输出的维度。
        :param dropout: dropout 概率。默认值为 0.0。
        :param bias: 多头注意力是否使用偏置。默认值为 True。
        :param activation: 激活函数。默认值为 'relu'，可选值 'relu', 'gelu'。
        :param kwargs: nn.MultiheadAttention() 的其他参数。
        """
        super(AttentionEnsemble, self).__init__()
        self.attn_weight = None  # 保存注意力权重
        self.attention = Attention(dropout=dropout, bias=bias)
        self.output_project = nn.Linear(input_size, output_size)
        self.activation = get_activation_fn(activation)  # 激活函数

    def forward(self, X):
        """
        前向传播
        :param X: 输入张量，维度为 [batch_size, input_size]。
        :return: 输出张量，维度为 [batch_size, output_size]。
        """
        attn_output, self.attn_weight = self.attention(X, X, X)
        attn_output = self.activation(attn_output + X)  # shape: [batch_size, input_size]
        output = self.output_project(attn_output)  # shape: [batch_size, output_size]
        return output


class AttentionProEnsemble(EnsembleBase):
    def __init__(self, input_size, output_size, project_size, *, feedforward=2048, dropout=0.1, bias=True,
                 activation='relu', **kwargs):
        """
        将多个特征使用具有高维映射的 Attention 进行融合。注意：这个过程无时序性，也没有位置编码，因此不能用于预测。
        :param input_size: 每个特征的维度，在这里指的是预测器的个数。
        :param output_size: 输出的维度。
        :param project_size: 参与注意力计算的特征维度。
        :param feedforward: 前馈神经网络的隐藏层维度。
        :param dropout: dropout 概率。默认值为 0.0。
        :param bias: 多头注意力是否使用偏置。默认值为 True。
        :param activation: 激活函数。默认值为 'relu'，可选值 'relu', 'gelu'。
        :param kwargs: nn.MultiheadAttention() 的其他参数。
        """
        super(AttentionProEnsemble, self).__init__()
        self.attn_weight = None  # 保存注意力权重
        self.activation = get_activation_fn(activation)  # 激活函数

        self.input_project = nn.Linear(input_size, project_size)  # 将特征维度映射到嵌入维度
        self.attention = Attention(embed_dim=4, num_heads=2, dropout=dropout, bias=bias)  # 注意力机制
        self.output_project = nn.Sequential(
            nn.Linear(project_size, feedforward),
            get_activation_nn(activation),
            nn.Linear(feedforward, output_size),
        )  # 将特征维度映射到输出层维度层

    def forward(self, X):
        """
        前向传播
        :param X: 输入张量，维度为 [batch_size, input_size]。
        :return: 输出张量，维度为 [batch_size, output_size]。
        """
        X = self.activation(self.input_project(X))  # shape: [batch_size, project_size]
        attn_output, self.attn_weight = self.attention(X, X, X)
        attn_output = self.activation(attn_output + X)  # shape: [batch_size, project_size]
        output = self.output_project(attn_output)  # shape: [batch_size, output_size]
        return output
