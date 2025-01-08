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

import os
import time
import pandas as pd
import torch
import torch.nn as nn

from Class_Function.Accumulator import Accumulator
from Class_Function.Animator import Animator
from Class_Function.device_gpu import try_gpu


def predict_RNNs(model, dataloader, device):
    """
    预测 RNNs 系列模型
    :param model: 需要预测的 RNNs 系列模型
    :param dataloader: DataLoader 数据加载器。X的维度：[batch_size, steps, inputs_node]，Y的维度：[batch_size, outputs_node]
    :param device: 运算设备
    :return: 预测结果。维度为：torch.Size([batch_size, outputs_node])
    """
    model.to(device)
    model.eval()
    predict_result = torch.empty((0,1), dtype=torch.float32)
    with torch.no_grad():
        for X, Y in dataloader:
            batch_size = X.shape[0]
            state = model.begin_state(batch_size, device)
            # 将state从上一个计算图中剔除，防止之前计算图的梯度影响下一次的梯度更新。（是否可以删除？）
            if isinstance(model, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
            # RNNs 开始推理
            X, Y = X.to(device), Y.to(device)
            X = X.permute(1, 0, 2)  # 将输入张量的维度调整为：[time_step, batch_size, input_size]
            Y_hat, hidden_state = model(X, state)
            predict_result = torch.cat((predict_result, Y_hat.cpu()), dim=0)
    return predict_result


def evaluate_RNNs(model, dataloader, monitors, device):
    """
    评估 RNNs 系列模型
    :param model: 需要评估的 RNNs 系列模型
    :param dataloader: DataLoader 数据加载器。需要有 dataloader.dataset.targets 属性
    :param monitors: Union[List|Tuple] 监控器指标函数，用于监测预测任务的表现，监控器函数 reduction 应当为 'mean'。
    :param device: 运算设备
    :return: true_predict_result, monitors_result
        true_predict_result [pd.DataFrame]: 预测结果，columns=['true', 'predict']
        monitors_result [Tuple]: 每个监控器平均样本指标值
    """
    monitors_result = []
    predict_result = predict_RNNs(model, dataloader, device)
    for monitor in monitors:
        monitors_result.append(monitor(predict_result, dataloader.dataset.target.unsqueeze(dim=1)).item())
    true_predict_result = pd.DataFrame({'true': dataloader.dataset.target.numpy(), 'predict': predict_result.squeeze().numpy()})
    return true_predict_result, tuple(monitors_result)


def train_epoch_RNNs(model, dataloader, criterion, optimizer, monitor, device, clip_norm=None):
    """
    训练 RNNs 系列模型一个迭代周期
    :param model: 需要训练的 RNNs 系列模型
    :param dataloader: DataLoader 数据加载器
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param monitor: 监控器指标函数，用于监测预测任务的表现。监控器函数 reduction 应当为 'mean'。
    :param clip_norm: 梯度裁剪时最大的梯度值（L2）。默认为 None，即不进行梯度裁剪。
    :return: 训练平均损失函数值（all_loss/samples），训练平均监控器指标值，每秒样本数（all_sample/sec）
    """
    model.to(device)
    model.train()
    start_time = time.perf_counter()
    train_metric = Accumulator(3)  # 累加器：（总损失函数值, 总监测函数值, 总样本数）
    for X, Y in dataloader:
        # 处理隐藏层状态
        batch_size = X.shape[0]
        state = model.begin_state(batch_size=batch_size, device=device)
        # 将state从上一个计算图中剔除，防止之前计算图的梯度影响下一次的梯度更新。
        if isinstance(model, nn.Module) and not isinstance(state, tuple):
            state.detach_()
        else:
            for s in state:
                s.detach_()

        X, Y = X.to(device), Y.to(device)  # 将数据转移到设备上
        X = X.permute(1, 0, 2)  # 将输入张量的维度调整为：[time_step, batch_size, input_size]
        Y = Y.unsqueeze(dim=1)  # 将目标张量的维度调整为：[batch_size, output_size]
        Y_hat, hidden_state = model(X, state)
        loss_value = criterion(Y_hat, Y.float())
        optimizer.zero_grad()
        loss_value.backward()
        if clip_norm:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm, norm_type=2)
        optimizer.step()
        train_metric.add(loss_value.item()*Y.numel(), monitor(Y_hat, Y.float()).item()*Y.numel(), Y.numel())
    sample_sec = train_metric[2] / (time.perf_counter() - start_time)
    return train_metric[0] / train_metric[2], train_metric[1] / train_metric[2], sample_sec


def train_RNNs(model, train_loader, valid_loader, *, epochs, criterion, optimizer, scheduler, monitor, clip_norm=None,
               device=None, best_model_dir=None, loss_figure_path=None, monitor_figure_path=None, loss_result_path=None,
               monitor_result_path=None, monitor_name="Monitor", loss_title="Loss", monitor_title=None,
               loss_yscale="linear", monitor_yscale="linear"):
    """
    训练 RNNs 系列模型
    :param model: 需要训练的 RNNs 系列模型
    :param train_loader: 训练集 DataLoader 数据加载器
    :param valid_loader: 验证集 DataLoader 数据加载器
    :param epochs: 迭代次数
    :param criterion: 损失函数。reduction 应当设置为 'mean'。
    :param optimizer: 参数优化器
    :param scheduler: 学习率调度器，支持 ReduceLROnPlateau 学习率调度器。
    :param monitor: 监控器指标函数，用于监测预测任务的表现，最小化监测器值（在预测任务中可以选择监测 MAPE, sMAPE, MAE, -log(R2) 等指标中的一个）
        ，保存最好模型将会参考监测器指标。监控器函数 reduction 应当设置为 'mean'。
    :param clip_norm: 梯度裁剪时最大的梯度值（L2）。默认为 None，即不进行梯度裁剪。
    :param device: 运算设备。默认为 None，如果有 GPU 则使用 GPU 进行运算，否则使用 CPU 进行运算。
    :param best_model_dir: 最佳模型保存目录，保存整个模型。默认为 None，即不保存模型。
    :param loss_figure_path: 损失函数图像保存路径。默认为 None，即不保存损失函数图像。
    :param monitor_figure_path: 监控器指标图像保存路径。默认为 None，即不保存监控器指标图像。
    :param loss_result_path: 损失函数结果保存路径。默认为 None，即不保存损失函数结果。
    :param monitor_result_path: 监控器指标结果保存路径。默认为 None，即不保存监控器指标结果。
    :param monitor_name: 监控器指标名称。默认为 "Monitor"。
    :param loss_title: 损失函数绘图标题。默认为 "Loss"。
    :param monitor_title: 监控器指标绘图标题。默认为 None，表示使用 monitor_name。
    :param loss_yscale: 损失函数绘图 Y 轴刻度。默认为 "linear"。
    :param monitor_yscale: 监控器指标绘图 Y 轴刻度。默认为 "linear"。
    :return: best_monitor, run_time, best_model_path
        best_monitor: 最佳监控器指标值
        run_time: 总运行时间
        best_model_path: 最佳模型保存路径。如果 best_model_dir 为 None，则返回 None。
    """
    start_time = time.perf_counter()  # 记录开始时间
    if device is None:
        device = try_gpu()
    if monitor_title is None:
        monitor_title = monitor_name

    loss_result, monitor_result = [], []  # 保存损失函数和监控器指标结果
    loss_animators = Animator(xlabel='epoch', ylabel='loss', title=loss_title, legend=['train', 'valid'],
                              xlim=[0, epochs], yscale=loss_yscale)
    monitor_animators = Animator(xlabel='epoch', ylabel=monitor_name, title=monitor_title, legend=['train', 'valid'],
                                 xlim=[0, epochs], yscale=monitor_yscale)
    best_monitor = float('inf')  # 初始化最佳监控器指标值
    best_model_filename = None  # 初始化最佳模型文件名

    for epoch in range(epochs):
        train_loss, train_monitor, sample_sec = train_epoch_RNNs(
            model, train_loader, criterion, optimizer, monitor, device, clip_norm=clip_norm)
        _, monitors_result = evaluate_RNNs(model, valid_loader, monitors=[criterion, monitor], device=device)
        valid_loss, valid_monitor = monitors_result
        scheduler.step(valid_monitor)  # 更新学习率并监测验证集上的性能
        print(f"epoch：{epoch+1}，学习率：{optimizer.param_groups[0]['lr']}，每秒样本数：{sample_sec:.2f}；\n",
              f"\t\t训练集损失值：{train_loss:.5f}，训练集监测值：{train_monitor:.5f}；\n",
              f"\t\t验证集损失值：{valid_loss:.5f}，验证集监测值：{valid_monitor:.5f}。\n", sep='')
        loss_result.append([train_loss, valid_loss])
        monitor_result.append([train_monitor, valid_monitor])
        if valid_monitor < best_monitor:
            best_monitor = valid_monitor
            if best_model_dir:
                best_model_filename = f'epoch={epoch+1}, monitor={valid_monitor:.5f}.pth'
                torch.save(model, os.path.join(best_model_dir, best_model_filename))
        loss_animators.add(epoch+1, [train_loss, valid_loss])
        monitor_animators.add(epoch+1, [train_monitor, valid_monitor])
    loss_animators.show(loss_figure_path)
    monitor_animators.show(monitor_figure_path)
    torch.save(model, os.path.join(best_model_dir, f'epoch={epochs}, final.pth')) if best_model_dir else None
    if loss_result_path:
        pd.DataFrame(loss_result, columns=['train', 'valid'], index=range(1, len(loss_result)+1)).to_csv(loss_result_path)
    if monitor_result_path:
        pd.DataFrame(monitor_result, columns=['train', 'valid'], index=range(1, len(monitor_result)+1)).to_csv(monitor_result_path)
    best_model_path = os.path.join(best_model_dir, best_model_filename) if best_model_dir else None  # 最佳模型保存路径
    run_time = time.perf_counter() - start_time
    print(f"训练结束，最佳模型监控器指标值为：{best_monitor:.5f}，总运行时间：{run_time:.2f}秒。")
    return best_monitor, run_time, best_model_path


class RNNModelBase(nn.Module):
    def __init__(self, rnn_layer, output_size):
        """
        定义 RNNs 系列模型的基模型。
        :param rnn_layer: 声明所需的 RNNs 系列模型的层。可以为 nn.RNN, nn.LSTM, nn.GRU
        :param output_size: 输出层的特征维度，即输出节点个数
        """
        super().__init__()
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
        :param inputs: 输入张量。维度为：[time_step, batch_size, input_size]。
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

    def to_device(self, device):
        """将模型转移到指定设备"""
        self.to(device)

    def fit(self, train_loader, valid_loader, epochs, criterion, optimizer, scheduler, monitor, **kwargs):
        """
        训练 RNNs 系列模型
        :param train_loader: 训练集 DataLoader 数据加载器
        :param valid_loader: 验证集 DataLoader 数据加载器
        :param epochs: 迭代次数
        :param criterion: 损失函数。reduction 应当设置为 'mean'。
        :param optimizer: 优化器
        :param scheduler: 学习率调度器
        :param monitor: 监控器指标函数，用于监测预测任务的表现，最小化监测器值。
        :param kwargs: 其他参数，具体参数见 train_RNNs 函数。
        :return: best_monitor（最佳监控器指标值），run_time（总运行时间）
        """
        best_monitor, run_time, best_model_path = train_RNNs(
            self, train_loader=train_loader, valid_loader=valid_loader, epochs=epochs, criterion=criterion,
            optimizer=optimizer, scheduler=scheduler, monitor=monitor, **kwargs
        )
        if best_model_path:
            best_model = torch.load(best_model_path)
            self.load_state_dict(best_model.state_dict())  # 加载最佳模型参数
        return best_monitor, run_time

    def predict(self, dataloader, device=None):
        """
        预测 RNNs 系列模型
        :param dataloader: DataLoader 数据加载器
        :param device: 运算设备。默认为 None，如果有 GPU 则使用 GPU 进行运算，否则使用 CPU 进行运算。
        :return: 预测结果。维度为：torch.Size([batch_size, outputs_node])
        """
        if device is None:
            device = try_gpu()
        predict_result = predict_RNNs(self, dataloader, device=device)
        return predict_result

    def evaluate(self, dataloader, monitors, device=None):
        """
        评估 RNNs 系列模型
        :param dataloader: DataLoader 数据加载器
        :param monitors: Union[List|Tuple] 监控器指标函数，用于监测预测任务的表现，监控器函数 reduction 应当为 'mean'。
        :param device: 运算设备。默认为 None，如果有 GPU 则使用 GPU 进行运算，否则使用 CPU 进行运算。
        :return: true_predict_result, monitors_result
            true_predict_result [pd.DataFrame]: 预测结果，columns=['true', 'predict']
            monitors_result [Tuple]: 每个监控器平均样本指标值
        """
        if device is None:
            device = try_gpu()
        true_predict_result, monitors_result = evaluate_RNNs(self, dataloader, monitors, device)
        return true_predict_result, monitors_result

    def save(self, model_path):
        """保存模型"""
        torch.save(self, model_path)


class RNNModel(RNNModelBase):
    def __init__(self, input_size, hidden_size, output_size, *, num_layers=1, bidirectional=False, **kwargs):
        rnn_layer = nn.RNN(input_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, **kwargs)
        super(RNNModel, self).__init__(rnn_layer, output_size)


class LSTMModel(RNNModelBase):
    def __init__(self, input_size, hidden_size, output_size, *, num_layers=1, bidirectional=False, **kwargs):
        lstm_layer = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, **kwargs)
        super(LSTMModel, self).__init__(lstm_layer, output_size)


class GRUModel(RNNModelBase):
    def __init__(self, input_size, hidden_size, output_size, *, num_layers=1, bidirectional=False, **kwargs):
        gru_layer = nn.GRU(input_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, **kwargs)
        super(GRUModel, self).__init__(gru_layer, output_size)

