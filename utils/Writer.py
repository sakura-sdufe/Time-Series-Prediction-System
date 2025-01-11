# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation-Project
# @Time     : 2024/11/20 15:24
# @Author   : 张浩
# @FileName : Writer.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import os
import shutil
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline  # 设置图片以SVG格式显示

from .Cprint import cprint


class Writer:
    def __init__(self, save_dir, is_delete=False, *, fmts=None, figsize=None, **kwargs):
        """
        写入指标、参数、数据、日志；保存模型
        :param save_dir: 保存的根目录，要求为绝对地址。
        :param is_delete: 是否删除原有的 save_dir 文件夹。
        :param fmts: 图像的格式，例如：['b-', 'm--', 'g-.', 'r:']。
        :param figsize: 图窗大小，例如：(7, 5)。
        :param kwargs: 其他图窗相关参数。
        """
        if not os.path.isabs(save_dir):
            raise ValueError(f"输入的根目录参数 {save_dir} 不是绝对路径！")

        # draw 绘图方法参数
        if fmts is None:
            fmts = ['C0-', 'C1--', 'C2-.', 'C3:', 'C4-', 'C5--', 'C6-.', 'C7:', 'C8-', 'C9--']
        if figsize is None:
            figsize = (7, 5)
        self.fmts, self.figsize = fmts, figsize
        self.init_axes_set = kwargs

        # 指标、参数、数据的索引
        self.metric_key_index, self.parameter_key_index, self.data_key_index, self.log_key_index = 0, 0, 0, 0
        self.metric = dict() # 指标（key 为关键词，value:pd.DataFrame 为所存储的指标）
        self.parameters = dict()  # 参数（key 为关键词，value:str 为所存储的参数）
        self.data = dict()  # 数据（key 为关键词，value:pd.DataFrame 为所存储的数据）
        self.log = dict()  # 日志（key 为关键词，value:str 为所存储的日志）

        self.save_dir = save_dir  # 保存的根目录
        self.save_documents_path = os.path.join(save_dir, 'documents')  # 保存文档路径
        self.save_models_path = os.path.join(save_dir, 'models_trained')  # 保存模型路径
        self.save_results_path = os.path.join(save_dir, 'results')  # 保存结果路径
        self.save_figures_path = os.path.join(save_dir, 'figures')  # 保存图片路径（Write）
        self.sub_paths = [self.save_documents_path, self.save_models_path, self.save_results_path, self.save_figures_path]

        # 删除文件夹
        if os.path.exists(save_dir) and is_delete:
            self._delete_dir()
        # 创建文件夹
        for path in self.sub_paths:
            if not os.path.exists(path):
                os.makedirs(path)

    def _delete_dir(self):
        """
        删除 save_dir 文件夹。如果文件夹内存在文件的话，先验证是否存在 documents、models_trained、results、figures 文件夹。
        如果不同时存在这些文件夹，则抛出 OSError 错误，因为您很有可能删除了错误的文件夹。
        如果同时存在这些文件夹，那么将会删除这四个文件夹内的所有文件（包括这四个文件夹）。
        最后，检查 self.save_dir 文件夹中是否有其他文件，如果有，则提示用户到资源管理器中核验，是否继续删除。
        """
        # Step 1: 验证是否存在 documents、models_trained、results、figures 文件夹
        raise_info = f"您很有可能删除了错误的目录或该目录已经进行了修改。当前的根目录路径为：{self.save_dir}，请仔细检查！"
        for path in self.sub_paths:
            if not os.path.exists(path):
                raise OSError(raise_info)

        # Step 2: 删除这四个文件夹及其内部文件
        for path in self.sub_paths:
            while True:
                verify = input(f"您确定要删除 '{path}' 文件夹及其内部文件吗？删除将无法恢复，请谨慎选择！(y/n)")
                if verify.lower() == 'y':
                    shutil.rmtree(path)
                    print(f"删除 '{path}' 文件夹及其内部文件成功！")
                    break
                elif verify.lower() == 'n':
                    print(f"取消删除 '{path}' 文件夹及其内部文件！")
                    break
                else:
                    print("输入错误，请重新输入！")
                    continue
        # Step 3: 检查 self.save_dir 文件夹中是否有其他文件
        if os.listdir(self.save_dir):
            os.startfile(self.save_dir)
            print(f"根目录 '{self.save_dir}' 中存在非 Writer 类产生的文件，已在资源管理器中打开该目录，请仔细核验！")
            verify = input("请到资源管理器中仔细核验当前目录下的文件，删除将无法恢复，是否继续删除（摁下除 'y' 以外所有键取消删除）？")
            if verify.lower() == 'y':
                shutil.rmtree(self.save_dir)
                print(f"根目录 '{self.save_dir}' 删除成功！")
            else:
                print(f"取消删除根目录 '{self.save_dir}'！")
        else:
            os.rmdir(self.save_dir)
            print(f"根目录 '{self.save_dir}' 删除成功！")

    def check_key(self, new_key, ignore_dict):
        """
        检查新的 key 是否已经存在于除 ignore_dict 之外的所有字典中。检查范围：self.metric, self.parameters, self.data, self.log。
        如果存在相同的 key，则返回 True，否则返回 False。当返回结果为 True 时，请不要使用该 key。
        """
        all_dict = [self.metric, self.parameters, self.data, self.log]
        check_result = any([new_key in d for d in all_dict if d is not ignore_dict])
        return check_result

    def add_metrics(self, metrics_df, axis, key=None):
        """
        添加指标到 self.metric 中。
        :param metrics_df: pd.DataFrame 类型，需要添加的指标结果。
        :param axis: 指定拼接的方向，0 为纵向拼接，1 为横向拼接。
        :param key: 指定的 key，如果为 None，则自动生成一个 key。
                    如果之前已经存在相同的 key，则将新的指标结果追加到原有的指标结果中；否则，新建一个 key。
        :return: None
        """
        if key is None:
            key = f"metric_{self.metric_key_index}"
            self.metric_key_index += 1
        if self.check_key(key, self.metric):
            raise ValueError("指标名称不能与参数名称或者数据名称相同！")
        elif key not in self.metric:
            self.metric[key] = metrics_df
        else:
            self.metric[key] = pd.concat([self.metric[key], metrics_df], axis=axis)

    def add_parameters(self, content_text, content_dict, key=None, sep='\n', end='\n\n'):
        """
        添加参数到 self.parameters 中。
        :param content_text: 参数的文本内容。
        :param content_dict: 参数的字典内容。
        :param key: 指定的 key，如果为 None，则自动生成一个 key。
                    如果之前已经存在相同的 key，则将新的参数追加到原有的参数中；否则，新建一个 key。
        :param sep: 参数文本和参数字典、参数字典之间的分隔符。
        :param end: 每次调用 add_parameters 时，在最后添加的分隔符。
        :return: None
        """
        if key is None:
            key = f"parameter_{self.parameter_key_index}"
            self.parameter_key_index += 1
        # 当前文本情况
        if content_text and content_dict:
            current_content = content_text + sep + \
                              sep.join([f"\t{(key+':').ljust(30, ' ')} {value}" for key, value in content_dict.items()])
        elif content_text and not content_dict:
            current_content = content_text
        elif not content_text and content_dict:
            current_content = sep.join([f"\t{(key+':').ljust(30, ' ')} {value}" for key, value in content_dict.items()])
        else:
            raise ValueError("content_text 和 content_dict 不能同时为空！")
        # 添加内容
        if self.check_key(key, self.parameters):
            raise ValueError("参数名称不能与指标名称或者数据名称相同！")
        elif key not in self.parameters:
            self.parameters[key] = current_content
        else:
            self.parameters[key] = self.parameters[key] + end + current_content

    def add_data(self, data_df, axis, key=None):
        """
        添加数据到 self.data 中。
        :param data_df: pd.DataFrame 类型，需要添加的数据。
        :param axis: 指定拼接的方向，0 为纵向拼接，1 为横向拼接。
        :param key: 指定的 key，如果为 None，则自动生成一个 key。
                    如果之前已经存在相同的 key，则将新的数据追加到原有的数据中；否则，新建一个 key。
        :return: None
        """
        if key is None:
            key = f"data_{self.data_key_index}"
            self.data_key_index += 1
        if self.check_key(key, self.data):
            raise ValueError("数据名称不能与指标名称或者参数名称相同！")
        elif key not in self.data:
            self.data[key] = data_df
        else:
            self.data[key] = pd.concat([self.data[key], data_df], axis=axis)

    def add_log(self, context, key=None, end='\n'):
        """
        添加日志到 self.log 中。
        :param context: 日志内容。
        :param key: 指定的 key，如果为 None，则自动生成一个 key。
                    如果之前已经存在相同的 key，则将新的日志追加到原有的日志中；否则，新建一个 key。
        :param end: 每次调用 add_log 时，在最后添加的分隔符。
        :return: None
        """
        if key is None:
            key = f"log_{self.log_key_index}"
            self.log_key_index += 1
        if self.check_key(key, self.log):
            raise ValueError("数据名称不能与指标名称或者参数名称或者数据名称相同！")
        elif key not in self.log:
            self.log[key] = context
        else:
            self.log[key] = self.log[key] + end + context

    def write_model(self, model, filename):
        """
        保存模型。模型保存目录：self.save_models_path
        :param model: 模型
        :param filename: 文件名
        :return: None
        """
        with open(os.path.join(self.save_models_path, filename), 'wb') as f:
            pickle.dump(model, f)

    def write(self, save_mode='w+'):
        """
        将暂存的指标、参数、数据、日志导出到指定的文件夹中。
        :param save_mode: 保存模式，默认为 'w+'，即覆盖写入。如果为 'a+'，则为追加写入（可实现不同 Writer 写入同一个文件）。
        - 指标保存目录：self.save_results_path
        - 数据保存目录：self.save_results_path
        - 参数保存目录：self.save_documents_path
        - 日志保存目录：self.save_documents_path
        :return: None
        """
        assert save_mode in ['w+', 'a+'], "保存模式只能为 'w+' 或 'a+'！"
        # 写入指标类文件
        for key, value in self.metric.items():
            if (save_mode == 'a+') and os.path.exists(os.path.join(self.save_results_path, f"{key}.xlsx")):
                exist_excel = pd.read_excel(os.path.join(self.save_results_path, f"{key}.xlsx"), index_col=0)
                new_csv = pd.concat([exist_excel, value], axis=0)
                new_csv.to_excel(os.path.join(self.save_results_path, f"{key}.xlsx"), index=True)
            else:
                value.to_excel(os.path.join(self.save_results_path, f"{key}.xlsx"), index=True)
        # 写入数据类文件
        for key, value in self.data.items():
            if (save_mode == 'a+') and os.path.exists(os.path.join(self.save_results_path, f"{key}.csv")):
                exist_csv = pd.read_csv(os.path.join(self.save_results_path, f"{key}.csv"), index_col=0)
                new_csv = pd.concat([exist_csv, value], axis=1)
                new_csv.to_csv(os.path.join(self.save_results_path, f"{key}.csv"), index=True)
            else:
                value.to_csv(os.path.join(self.save_results_path, f"{key}.csv"), index=True)
        # 写入参数类文件
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for key, value in self.parameters.items():
            with open(os.path.join(self.save_documents_path, f"{key}.txt"), save_mode, encoding='utf-8') as f:
                if save_mode == 'w+':
                    f.write(f"{current_time}\n{value}")
                elif save_mode == 'a+':
                    f.write(f"\n\n\n[{current_time}]\n{value}")
        # 写入日志类文件
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for key, value in self.log.items():
            with open(os.path.join(self.save_documents_path, f"{key}.txt"), save_mode, encoding='utf-8') as f:
                if save_mode == 'w+':
                    f.write(f"{current_time}\n{value}")
                elif save_mode == 'a+':
                    f.write(f"\n\n\n[{current_time}]\n{value}")

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
        def use_svg_display():
            """使用矢量图(SVG)打印图片"""
            backend_inline.set_matplotlib_formats('svg')

        # 更新默认配置
        current_axes_set = deepcopy(self.init_axes_set)
        current_axes_set.update(kwargs)
        current_fmts = fmts if fmts is not None else self.fmts
        current_figsize = figsize if figsize is not None else self.figsize
        # 如果 y 是 None，则表示需要把 X 设置为 None，y 设置为纵坐标数据。
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
            plt.savefig(os.path.join(self.save_figures_path, filename), dpi=None, facecolor='w', edgecolor='w')
            plt.show()
            cprint(f"绘制图像，图片已保存至 {os.path.join(self.save_figures_path, filename)}。", text_color="白色", end='\n')
        elif filename and not show:
            plt.savefig(os.path.join(self.save_figures_path, filename), dpi=None, facecolor='w', edgecolor='w')
            plt.close()
            cprint(f"未绘制图像，图片已保存至 {os.path.join(self.save_figures_path, filename)}", text_color="红色", end='\n')
        elif not filename and show:
            plt.show()
            cprint("图像绘制完成，但未保存图像！", text_color="红色", end='\n')
        elif not filename and not show:
            plt.close()
            cprint("未绘制图像，也未保存图像！", text_color="红色", end='\n')
        else:
            raise ValueError("出现未预知的错误！")
