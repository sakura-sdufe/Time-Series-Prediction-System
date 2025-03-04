# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation
# @Time     : 2025/1/9 13:16
# @Author   : 张浩
# @FileName : Base.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import os
import copy
import time
import warnings
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.Animator import Animator
from utils.device_gpu import try_gpu


def evaluate_DL(model, dataloader, monitors, device):
    """
    评估深度学习模型
    :param model: 需要评估的深度学习模型。需要有 predict.predict() 方法。
    :param dataloader: DataLoader 数据加载器。需要有 dataloader.dataset.targets 属性
    :param monitors: Union[List|Tuple] 监控器指标函数，用于监测预测任务的表现，监控器函数 reduction 应当为 'mean'。
    :param device: 运算设备
    :return: true_predict_result, monitors_result
        true_predict_result [pd.DataFrame]: 预测结果，columns=['true', 'predict']
        monitors_result [Tuple]: 每个监控器平均样本指标值
    """
    monitors_result = []
    predict_result = model.predict(dataloader, device)
    for monitor in monitors:
        monitors_result.append(monitor(predict_result, dataloader.dataset.target.unsqueeze(dim=1)).item())
    true_predict_result = pd.DataFrame({'true': dataloader.dataset.target.numpy(), 'predict': predict_result.squeeze().numpy()})
    return true_predict_result, tuple(monitors_result)


def train_DL(model, train_trainer, train_evaler, valid_evaler, *, epochs, criterion, optimizer, scheduler, monitor,
             clip_norm=None, device=None, best_model_dir=None, loss_figure_path=None, monitor_figure_path=None,
             loss_result_path=None, monitor_result_path=None, lr_sec_path=None, monitor_name="Monitor", loss_title="Loss",
             monitor_title=None, loss_yscale="linear", monitor_yscale="linear", train_back=True):
    """
    训练深度学习模型
    :param model: 需要训练的深度学习模型，需要有 predict.train_epoch() 方法。
    :param train_trainer: 训练集 DataLoader 数据加载器（用于训练）。
    :param train_evaler: 训练集 DataLoader 数据加载器（用于评估）。
    :param valid_evaler: 验证集 DataLoader 数据加载器。
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
    :param loss_result_path: 损失函数结果保存路径。默认为 None，即不保存损失函数结果。后缀名为 .csv
    :param monitor_result_path: 监控器指标结果保存路径。默认为 None，即不保存监控器指标结果。后缀名为 .csv
    :param lr_sec_path: 学习率与每秒样本数结果保存路径。默认为 None，即不保存学习率与每秒样本数结果。后缀名为 .csv
    :param monitor_name: 监控器指标名称。默认为 "Monitor"。
    :param loss_title: 损失函数绘图标题。默认为 "Loss"。
    :param monitor_title: 监控器指标绘图标题。默认为 None，表示使用 monitor_name。
    :param loss_yscale: 损失函数绘图 Y 轴刻度。默认为 "linear"。
    :param monitor_yscale: 监控器指标绘图 Y 轴刻度。默认为 "linear"。
    :param train_back: 是否在训练过程中，如果学习率发生变化，是否读取之前的最优模型，在最优模型上继续训练。默认为 True。
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

    loss_result, monitor_result, lr_sec_result = [], [], []  # 保存损失函数结果、监控器函数结果、学习率与每秒样本数结果
    loss_animators = Animator(xlabel='epoch', ylabel='loss', title=loss_title, legend=['train', 'valid'],
                              xlim=[0, epochs], yscale=loss_yscale)
    monitor_animators = Animator(xlabel='epoch', ylabel=monitor_name, title=monitor_title, legend=['train', 'valid'],
                                 xlim=[0, epochs], yscale=monitor_yscale)
    best_monitor = float('inf')  # 初始化最佳监控器指标值
    best_model_filename = None  # 初始化最佳模型文件名
    current_lr = optimizer.param_groups[0]['lr']  # 初始化当前学习率

    for epoch in range(epochs):
        epoch_start_time = time.perf_counter()
        model.train_epoch(dataloader=train_trainer, criterion=criterion, optimizer=optimizer,
                          device=device, clip_norm=clip_norm)
        sample_sec = len(train_trainer.dataset) / (time.perf_counter() - epoch_start_time)  # 计算每秒样本数
        _, train_evaluate_result = evaluate_DL(model, train_evaler, monitors=[criterion, monitor], device=device)
        _, valid_evaluate_result = evaluate_DL(model, valid_evaler, monitors=[criterion, monitor], device=device)
        train_loss, train_monitor = train_evaluate_result
        valid_loss, valid_monitor = valid_evaluate_result
        scheduler.step(valid_monitor)  # 更新学习率并监测验证集上的性能
        if train_back and current_lr != optimizer.param_groups[0]['lr']:  # 如果学习率发生变化，读取之前的最优模型，在最优模型上继续训练
            try:
                model.load_state_dict(torch.load(os.path.join(best_model_dir, best_model_filename)).state_dict())
                print("\n" + "-"*100 + "\n学习率发生变化，读取之前的最优模型，在最优模型上继续训练。\n" + "-"*100 + "\n")
            except Exception:
                warnings.warn(f"读取最优模型失败，将使用学习率更新前的上一个状态继续训练。")
        print(f"epoch：{epoch+1}，学习率：{optimizer.param_groups[0]['lr']}，每秒样本数：{sample_sec:.2f}；\n",
              f"\t\t训练集损失值：{train_loss:.5f}，训练集监测值：{train_monitor:.5f}；\n",
              f"\t\t验证集损失值：{valid_loss:.5f}，验证集监测值：{valid_monitor:.5f}，最佳验证集监测值：{best_monitor:.5f}。\n",
              sep='')
        loss_result.append([train_loss, valid_loss])  # 保存损失函数结果
        monitor_result.append([train_monitor, valid_monitor, best_monitor])  # 保存监控器函数结果
        lr_sec_result.append([optimizer.param_groups[0]['lr'], sample_sec])  # 保存学习率与每秒样本数结果
        if valid_monitor < best_monitor:  # 保存最佳模型
            best_monitor = valid_monitor
            if best_model_dir:
                best_model_filename = f'epoch={epoch+1}, monitor={valid_monitor:.5f}.pth'
                torch.save(model, os.path.join(best_model_dir, best_model_filename))
        loss_animators.add(epoch+1, [train_loss, valid_loss])  # 绘制损失函数动态图
        monitor_animators.add(epoch+1, [train_monitor, valid_monitor])  # 绘制监控器指标动态图
    loss_animators.show(loss_figure_path)  # 显示（保存）损失函数动态图
    monitor_animators.show(monitor_figure_path)  # 显示（保存）监控器指标动态图
    torch.save(model, os.path.join(best_model_dir, f'epoch={epochs}, final.pth')) if best_model_dir else None

    if loss_result_path:
        pd.DataFrame(
            loss_result, columns=['train', 'valid'], index=range(1, len(loss_result)+1)
        ).to_csv(loss_result_path)
    if monitor_result_path:
        pd.DataFrame(
            monitor_result, columns=['train', 'valid', 'best(valid)'], index=range(1, len(monitor_result)+1)
        ).to_csv(monitor_result_path)
    if lr_sec_path:
        pd.DataFrame(
            lr_sec_result, columns=['lr', 'sample/sec'], index=range(1, len(lr_sec_result)+1)
        ).to_csv(lr_sec_path)
    best_model_path = os.path.join(best_model_dir, best_model_filename) if best_model_dir else None  # 最佳模型保存路径
    run_time = time.perf_counter() - start_time
    print(f"训练结束，最佳模型监控器指标值为：{best_monitor:.5f}，总运行时间：{run_time:.2f}秒。")
    return best_monitor, run_time, best_model_path


class ModelBase(nn.Module):
    def __init__(self, **kwawrgs):
        """
        定义深度学习模型的基模型。需要重写 __init__ 和 forward 方法。
        若子类中 forward 方法不满足标记的条件，那么需要重写 train_epoch 和 predict 方法。
        note:
            1. DataLoader类型数据满足 X 的维度：[batch_size, steps, input_size]，Y的维度：[batch_size]。
            2. _trainer 表示用于训练的 DataLoader 数据加载器，_evaler 表示用于评估的 DataLoader 数据加载器。主要差异是：
                _trainer 中的 shuffle=True 和 batch_size=predictor_batch_size；
                _evaler 中的 shuffle=False 和 batch_size=eval_batch_size。
            3. input_size 和 time_step 参数会自动解析为输入特征的维度和时间步数。
                input_size 是定义模型必须参数，但是模型参数无需指定。time_step 根据模型需求选择是否引入。
                input_size 为必须参数，即使没有使用也需要在模型中定义。
        """
        super(ModelBase, self).__init__()

    def forward(self, **kwargs):
        """
        深度学习模型向前计算，需要在子类中重写。
        note:
            1. 输入应当是一个 Tensor，且维度应为：[time_step, batch_size, input_size]；
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
            X, Y = X.to(device), Y.to(device)  # 将数据转移到设备上
            X = X.permute(1, 0, 2)  # 将输入张量的维度调整为：[time_step, batch_size, input_size]
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
            for X, Y in dataloader:
                X, Y = X.to(device), Y.to(device)
                X = X.permute(1, 0, 2)  # 将输入张量的维度调整为：[time_step, batch_size, input_size]
                Y_hat = self(X)
                predict_list.append(Y_hat.cpu())
        predict_result = torch.cat(predict_list, dim=0)
        return predict_result

    def fit(self, train_trainer, train_evaler, valid_evaler, epochs, criterion, optimizer, scheduler, monitor, **kwargs):
        """
        训练深度学习模型
        :param train_trainer: 训练集 DataLoader 数据加载器（用于训练）。
        :param train_evaler: 训练集 DataLoader 数据加载器（用于评估）。
        :param valid_evaler: 验证集 DataLoader 数据加载器。
        :param epochs: 迭代次数
        :param criterion: 损失函数。reduction 应当设置为 'mean'。
        :param optimizer: 优化器
        :param scheduler: 学习率调度器
        :param monitor: 监控器指标函数，用于监测预测任务的表现，最小化监测器值。
        :param kwargs: 其他参数，具体参数见 train_DL 函数。
        :return: best_monitor（最佳监控器指标值），run_time（总运行时间）
        """
        best_monitor, run_time, best_model_path = train_DL(
            self, train_trainer=train_trainer, train_evaler=train_evaler, valid_evaler=valid_evaler, epochs=epochs,
            criterion=criterion, optimizer=optimizer, scheduler=scheduler, monitor=monitor, **kwargs)
        if best_model_path:
            best_model = torch.load(best_model_path)
            self.load_state_dict(best_model.state_dict())  # 加载最佳模型参数
        return best_monitor, run_time

    def evaluate(self, dataloader, monitors, device=None):
        """
        评估深度学习模型
        :param dataloader: DataLoader 数据加载器。需要有 dataloader.dataset.targets 属性。
        :param monitors: Union[List|Tuple] 监控器指标函数，用于监测预测任务的表现，监控器函数 reduction 应当为 'mean'。
        :param device: 运算设备。默认为 None，如果有 GPU 则使用 GPU 进行运算，否则使用 CPU 进行运算。
        :return: true_predict_result, monitors_result
            true_predict_result [pd.DataFrame]: 预测结果，columns=['true', 'predict']
            monitors_result [Tuple]: 每个监控器平均样本指标值
        """
        if device is None:
            device = try_gpu()
        true_predict_result, monitors_result = evaluate_DL(self, dataloader, monitors, device)
        return true_predict_result, monitors_result

    def to_device(self, device):
        """将模型转移到指定设备"""
        self.to(device)

    def save(self, model_path):
        """保存模型"""
        torch.save(self, model_path)


class RepeatLayer(nn.Module):
    def __init__(self, layer, num_layers):
        """
        将多个 layer 层堆叠在一起。
        :param layer: torch.nn 模块支持的层或者模块（可以直接调用）。
        :param num_layers: layer 重复层数。
        """
        super(RepeatLayer, self).__init__()
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers

    def forward(self, X):
        output = X
        for sublayer in self.layers:
            output = sublayer(output)
        return output


def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == 'silu':
        return F.silu


def get_activation_nn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == 'silu':
        return nn.SiLU()


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
