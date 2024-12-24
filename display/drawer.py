# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation-Project
# @Time     : 2024/11/19 11:04
# @Author   : 张浩
# @FileName : drawer.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import os
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline  # 设置图片以SVG格式显示
from .cprint import cprint


def use_svg_display():
    """使用矢量图(SVG)打印图片"""
    backend_inline.set_matplotlib_formats('svg')


# 绘制图像
class Drawer:
    def __init__(self, save_dir, fmts=None, figsize=None, **kwargs):
        self.save_figures = os.path.join(save_dir, 'figures')
        if not os.path.exists(self.save_figures):
            os.makedirs(self.save_figures)

        if fmts is None:
            fmts = ['C0-', 'C1--', 'C2-.', 'C3:', 'C4-', 'C5--', 'C6-.', 'C7:', 'C8-', 'C9--']
        if figsize is None:
            figsize = (7, 5)
        self.fmts, self.figsize = fmts, figsize
        self.init_axes_set = kwargs

    def draw(self, x, y=None, fmts=None, figsize=None, filename=None, show=True, **kwargs):
        """
        绘制图像
        :param x: 1D 类型的序列数据，例如：list、tuple、ndarray等（要求有 __len__ 魔法方法）；该参数也可以省略，自动匹配长度。
        :param y: 1D 或 2D 类型的序列数据，例如：list、tuple、ndarray等（要求有 __len__ 魔法方法）。
        :param fmts: 图像的格式，例如：['b-', 'm--', 'g-.', 'r:']。
        :param figsize: 图窗大小，例如：(7, 5)。
        :param filename: 图片保存名称（带后缀）
        :param show: 是否展示图像，默认为 True。
        :param kwargs: 其他图窗相关参数。
        :return: None
        """
        # 更新默认配置
        current_axes_set = deepcopy(self.init_axes_set)
        current_axes_set.update(kwargs)
        current_fmts = fmts if fmts is not None else self.fmts
        current_figsize = figsize if figsize is not None else self.figsize
        # 如果 y 是 None，则表示需要把 x 设置为 None，y 设置为纵坐标数据。
        if y is None:
            x, y = None, x
        # 如果 y 是 1D 类型的序列数据，则将其转换为 2D 类型的序列数据。
        if isinstance(y, (list, tuple)) and not isinstance(y[0], (list, tuple, np.ndarray)):
            y = [y]
        elif isinstance(y, np.ndarray) and len(y.shape) == 1:
            y = y.reshape(1, -1)

        # 创建画布和子图
        use_svg_display()  # 使用svg格式的矢量图绘制图片
        nrows, ncols = 1, 1
        fig, axes = plt.subplots(nrows, ncols, figsize=current_figsize)
        # 绘制图像
        for i, seq in enumerate(y):
            if x is None:
                axes.plot(seq, current_fmts[i])
            else:
                axes.plot(x, seq, current_fmts[i])
        # 设置图窗
        for key, value in current_axes_set.items():
            if key == "legend":
                getattr(axes, f"{key}")(value, loc='upper right')
            else:
                getattr(axes, f"set_{key}")(value)
        axes.grid()
        # 展示图像 和 保存图像
        if filename and show:
            plt.savefig(os.path.join(self.save_figures, filename), dpi=None, facecolor='w', edgecolor='w')
            plt.show()
            cprint(f"绘制图像，图片已保存至 {os.path.join(self.save_figures, filename)}。", text_color="白色", end='\n')
        elif filename and not show:
            plt.savefig(os.path.join(self.save_figures, filename), dpi=None, facecolor='w', edgecolor='w')
            plt.close()
            cprint(f"未绘制图像，图片已保存至 {os.path.join(self.save_figures, filename)}", text_color="红色", end='\n')
        elif not filename and show:
            plt.show()
            cprint("图像绘制完成，但未保存图像！", text_color="红色", end='\n')
        elif not filename and not show:
            plt.close()
            cprint("未绘制图像，也未保存图像！", text_color="红色", end='\n')
        else:
            raise ValueError("出现未预知的错误！")
