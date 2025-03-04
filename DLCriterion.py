# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation
# @Time     : 2025/1/6 20:49
# @Author   : 张浩
# @FileName : DLCriterion.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import torch
from torch import Tensor
import torch.nn as nn


class MSELoss_scale(nn.MSELoss):
    def __init__(self, scale=1.0, **kwargs):
        super(MSELoss_scale, self).__init__(**kwargs)
        self.scale = scale

    def forward(self, inputs:Tensor, targets:Tensor) -> Tensor:
        return super(MSELoss_scale, self).forward(inputs, targets) * self.scale


class MSELoss_sqrt(nn.MSELoss):
    def __init__(self, power=1.0, **kwargs):
        super(MSELoss_sqrt, self).__init__(**kwargs)
        self.power = power

    def forward(self, inputs:Tensor, targets:Tensor) -> Tensor:
        return torch.pow(torch.sqrt(super(MSELoss_sqrt, self).forward(inputs, targets)), self.power)


class sMAPELoss(nn.Module):
    def __init__(self, reduction:str='mean'):
        super(sMAPELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction

    def forward(self, inputs:Tensor, targets:Tensor) -> Tensor:
        epsilon = 1e-6  # 防止分母为 0
        Numerator = torch.abs(targets - inputs)
        Denominator = (torch.abs(targets) + torch.abs(inputs)) / 2 + epsilon
        loss = (Numerator / Denominator)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MAPELoss(nn.Module):
    def __init__(self, reduction:str='mean'):
        super(MAPELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction

    def forward(self, inputs:Tensor, targets:Tensor) -> Tensor:
        epsilon = 1e-6  # 防止分母为 0
        Numerator = torch.abs(targets - inputs)
        Denominator = torch.abs(targets) + epsilon
        loss = (Numerator / Denominator)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
