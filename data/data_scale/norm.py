# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation-Project
# @Time     : 2024/11/10 21:23
# @Author   : 张浩
# @FileName : norm.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import numpy as np
import pandas as pd


class Norm:
    """对数据进行使用 Norm 方法进行归一化和反归一化"""
    def __init__(self, data):
        """
        :param data: data 可以是 2Darray, DataFrame, Series。对列进行归一化，即对 axis=0 方向进行归一化操作（有几列就执行几次归一化操作）。
            Note：2Darray 的数据类型应当为数值型，DataFrame 应当存在数值型的列。
            Note：标准化结果存放在 norm_result 属性中，可以通过 get_norm_result 方法获取。
        """
        self.mean = None  # 存储原始数据的均值
        self.std = None  # 存储原始数据的标准差
        self.norm_array = None  # 刚开始存储的是原始数据的数值型 numpy 数组，然后会调用 norm 方法进行归一化操作。
        self.norm_result = None  # 存储归一化后的结果（与输入的形式保持一致，主要用于处理 DataFrame）

        self.is_series = False  # 判断 data 是否为 Series
        if isinstance(data, pd.Series):
            self.is_series = True
            data = data.to_frame()

        self.is_dataframe = False  # 判断 data 是否为 DataFrame，下面参数只有在 data 为 DataFrame 时才有意义。
        self.data_columns = None
        data_index = None
        self.df_columns = None
        if isinstance(data, pd.DataFrame):
            self.is_dataframe = True
            self.data_columns = data.columns  # 保存原始数据的列名
            data_index = data.index  # 保存原始数据的索引
            df_norm = data.select_dtypes(include=[np.number])  # 选择数值型的列
            self.df_columns = df_norm.columns
            self.norm_array = df_norm.to_numpy()
        elif isinstance(data, np.ndarray) and np.ndim(data)==2:
            self.norm_array = data
        else:
            raise ValueError('data 只能是 2Darray, DataFrame 中的一种。')

        # 对数据进行归一化操作
        self._norm_private()

        # 将归一化后的 array 转换为 DataFrame（按照原来的列排序）
        if self.is_dataframe:
            self.norm_result = pd.DataFrame(columns=self.data_columns, index=data_index)
            for col in self.data_columns:
                if col in self.df_columns:
                    self.norm_result[col] = self.norm_array[:, self.df_columns.get_loc(col)]
                else:
                    self.norm_result[col] = data[col]
        else:
            self.norm_result = self.norm_array
        # 如果是 Series，将结果转换为 Series
        if self.is_series:
            self.norm_result = self.norm_result.squeeze()
        del data

    def _norm_private(self):
        """对数据进行归一化操作"""
        self.mean = np.mean(self.norm_array, axis=0)
        self.std = np.std(self.norm_array, axis=0)
        self.norm_array = (self.norm_array - self.mean) / self.std

    def get_norm_result(self):
        """获取归一化后的数据"""
        return self.norm_result

    def norm(self, data, number=None, column=None):
        """
        对给定的数据进行归一化操作。
        第一个参数你需要传入你想标准化的数据，输入的数据类型可以是数值型的 array 也可以是数值型的 DataFrame。
        第二个参数你可以传入整数，这个整数表示你所输入的数据使用第几列的参数进行反标准化（默认为第 0 列）。
            当你初始化类时使用的是 DataFrame 时，你也可以输入一个列名（string），这样就可以使用列名进行标准化。
            如果你的列名为整数，你也可以选择使用 column 参数传入列名。
        Note：当你第二个参数传入整数时，column 参数将会失效。DataFrame 会转换成 array 进行处理，不考虑列名的影响。
        """
        if isinstance(number, int):
            pass
        elif isinstance(number, str):
            number = self.data_columns.get_loc(number)
        elif (number is None) and (column is None):
            number = 0
        elif (number is None) and (column is not None) and self.is_dataframe:
            number = self.data_columns.get_loc(column)
        elif (number is None) and (column is not None) and (not self.is_dataframe):
            raise ValueError('当初始化类使用的不是 DataFrame 或 Series 类型数据时，column 参数无效。')
        else:
            raise ValueError('请输入合适的参数指定列。')

        if self.is_dataframe:
            # 将 self.data_columns 对应位置转换为列名，然后根据列名找到在 self.df_columns 对应的位置，最后提取对应的均值和方差。
            number = self.df_columns.get_loc(self.data_columns[number])

        is_dataframe = True if isinstance(data, pd.DataFrame) else False
        is_series = True if isinstance(data, pd.Series) else False
        cols, inds = None, None
        if is_dataframe:
            # 获取列名和索引
            cols, inds = data.columns, data.index
            # 转换为 numpy 数组
            data = data.to_numpy()
        elif is_series:
            # 获取列名和索引
            cols, inds = data.name, data.index
            # 转换为 numpy 数组
            data = data.to_numpy()

        # 从 self.mean 和 self.std 获取对应的均值和方差
        mean = self.mean[number]
        std = self.std[number]
        # 归一化
        data = (data - mean) / std

        # 转换为输出格式
        if is_dataframe:
            data = pd.DataFrame(data, columns=cols, index=inds)
        elif is_series:
            data = pd.Series(data, name=cols, index=inds)
        return data

    def denorm(self, data, number=None, column=None):
        """
        对给定的数据进行反归一化操作。
        第一个参数你需要传入你想反标准化的数据，输入的数据类型可以是数值型的 array 也可以是数值型的 DataFrame。
        第二个参数你可以传入整数，这个整数表示你所输入的数据使用第几列的参数进行反标准化（默认为第 0 列）。
            当你初始化类时使用的是 DataFrame 时，你也可以输入一个列名（string），这样就可以使用列名进行反标准化。
            如果你的列名为整数，你也可以选择使用 column 参数传入列名。
        Note：当你第二个参数传入整数时，column 参数将会失效。DataFrame 会转换成 array 进行处理，不考虑列名的影响。
        """
        if isinstance(number, int):
            pass
        elif isinstance(number, str):
            number = self.data_columns.get_loc(number)
        elif (number is None) and (column is None):
            number = 0
        elif (number is None) and (column is not None) and self.is_dataframe:
            number = self.data_columns.get_loc(column)
        elif (number is None) and (column is not None) and (not self.is_dataframe):
            raise ValueError('当初始化类使用的不是 DataFrame 或 Series 类型数据时，column 参数无效。')
        else:
            raise ValueError('请输入合适的参数指定列。')

        if self.is_dataframe:
            # 将 self.data_columns 对应位置转换为列名，然后根据列名找到在 self.df_columns 对应的位置，最后提取对应的均值和方差。
            number = self.df_columns.get_loc(self.data_columns[number])

        is_dataframe = True if isinstance(data, pd.DataFrame) else False
        is_series = True if isinstance(data, pd.Series) else False
        cols, inds = None, None
        if is_dataframe:
            # 获取列名和索引
            cols, inds = data.columns, data.index
            # 转换为 numpy 数组
            data = data.to_numpy()
        elif is_series:
            # 获取列名和索引
            cols, inds = data.name, data.index
            # 转换为 numpy 数组
            data = data.to_numpy()

        # 从 self.mean 和 self.std 获取对应的均值和方差
        mean = self.mean[number]
        std = self.std[number]
        # 反归一化
        data = data * std + mean

        # 转换为输出格式
        if is_dataframe:
            data = pd.DataFrame(data, columns=cols, index=inds)
        elif is_series:
            data = pd.Series(data, name=cols, index=inds)
        return data
