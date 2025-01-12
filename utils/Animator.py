# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :40_class_ImageNet
# @Time     :2023/6/23 11:18
# @Author   :张浩
# @FileName :xb_Animator.py
# @Software :PyCharm

定义动态画图类 Accumulator。每次对类的实例进行添加操作时，都会清除之前的图并且绘制新的图。
-------------------------------------------------
"""

from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline  # 设置图片以SVG格式显示
from IPython import display


def use_svg_display():
    """使用矢量图(SVG)打印图片"""
    backend_inline.set_matplotlib_formats('svg')


def set_axes(axes, xlabel, ylabel, xlim, ylim, title, xscale, yscale, legend):
    """设置matplotlib中的axes"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    axes.set_title(title)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    if legend:
        axes.legend(legend)
    axes.grid()


# 动态画图类，并且可以往其中添加数据更新图片
class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, title=None, xscale="linear", yscale="linear",
                 fmts=('b-', 'm--', 'g-.', 'r:'), figsize=(7, 5)):
        # 使用None代替可变的默认参数。防止运行函数的结果收到上一次运行结果的影响
        if legend is None:
            legend = []  # legend无论是否传入值，都是list类型数据

        # 使用svg格式的矢量图绘制图片
        use_svg_display()

        # 给出(子)图的画布 和 图窗配置
        # ax can be either a single Axes object, or an array of Axes objects if more than one subplot was created.
        # 单图的axes：axis。多图的axes：array([[axis1, axis2], [axis3, axis4]])
        nrows, ncols = 1, 1  # 指定nrows和ncols为1，并且只画一个图
        # 这里的画布设置是规范形式的，使用plt设置坐标轴其实是在其基础上继续封装的。
        self.fig, self.ax = plt.subplots(nrows, ncols, figsize=figsize)

        # 使用lambda生成一个图窗的一个配置文件。self.config_axes本质是一个匿名函数。self.axes只有一个图窗。
        self.config_axes = lambda: set_axes(self.ax, xlabel, ylabel, xlim, ylim, title,
                                            xscale, yscale, legend)

        # 将传入的fmt放到类内部，并且初始化画图数据（总的数据，每次都往其中添加x和y）
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y, clear_display=False):
        if clear_display:
            # 清除所有输出。其中，wait=True 表示在清除输出区域之前，先等待新输出的产生。在这里清除上一次输出。在 Jupyter 中很有用。
            display.clear_output(wait=True)

        # hasattr(object, name) 函数用于判断对象是否包含对应的属性。
        # 对y赋值
        if not hasattr(y, "__len__"):
            # 表示y是一个数字，不是一个list或者tuple
            y = [y]
        n = len(y)  # 需要画图的个数

        # 对x赋值。x可以传入一个数字，x的赋值可以自动求出y中有多少个数，需要多少个x，然后对x赋值。
        if not hasattr(x, "__len__"):
            x = [x] * n

        # 初始化X坐标，画图的X坐标，几个图list下几个list
        if not self.X:
            self.X = [[] for _ in range(n)]

        # 初始化Y坐标，画图的Y坐标，几个图list下几个list
        if not self.Y:
            self.Y = [[] for _ in range(n)]

        # 将x和y 追加到 self.X和self.Y。其中x, y和self.X, self.Y位置一一对应
        # i表示第i个 图(list)
        for i, (a, b) in enumerate(zip(x, y)):
            # 这里的x和y是输入的，它们需要添加到self.X和self.Y中
            # 这里的if很巧妙。如果a=0，那么if a 返回False，if is not None 返回True
            if (a is not None) and (b is not None):
                self.X[i].append(a)
                self.Y[i].append(b)
        self.ax.cla()  # 清除当前的坐标轴

        # 对每个self.X, self.Y下的list都绘图。并且指定线型。
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            # 这里的x和y是来自于self.X和self.Y的，它们表示的是一条曲线的横坐标和纵坐标
            self.ax.plot(x, y, fmt)  # 在axes上绘制图像和标记。
        self.config_axes()  # 对图像配置坐标轴信息

        # 显示图像。也可以 用于显示一个字符串、显示一个 Pandas 数据框、显示一张图片、显示一段 HTML 代码
        display.display(self.fig)
        # 在PyCharm中不正常显示图像，需要重绘图像
        plt.draw()
        plt.pause(1)

    def show(self, figure_path=None):
        # 对每个self.X, self.Y下的list都绘图。并且指定线型。
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            # 这里的x和y是来自于self.X和self.Y的，它们表示的是一条曲线的横坐标和纵坐标
            self.ax.plot(x, y, fmt)  # 在axes上绘制图像和标记。
        self.config_axes()  # 对图像配置坐标轴信息
        self.ax.grid()  # 生成网格

        # 保存图像
        if figure_path:
            print(f"图像已保存至{figure_path}")
            self.fig.savefig(figure_path, dpi=None, facecolor='w', edgecolor='w')
        # 在PyCharm中生成图片，并且block program
        # print("图像绘制完成，请查看图像，并且手动结束program block！")
        # plt.show()


if __name__ == "__main__":
    # 设置训练动态图
    loss_animator = Animator(xlabel='epoch', ylabel='loss', legend=['train loss', 'valid loss'], title="Model",
                             xlim=[0, 10])
    acc_animator = Animator(xlabel='epoch', ylabel='accuracy', legend=['train acc', 'valid acc'], title="Model",
                            xlim=[0, 10])
    for i in range(5):
        loss_animator.add(i, [i+1, -i+1])
        acc_animator.add(i, [i+2, -i+2])
    # 保存绘制的图片。先保存再做展示。
    loss_animator.show("../test loss.png")
    acc_animator.show("../test acc.png")
