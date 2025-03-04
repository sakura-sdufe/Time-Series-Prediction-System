# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 21:15:12 2023

@author: 张浩
"""

import sys
import os


def join_syspath_subfolder(subfolder, top_dir_join=True):
    """
    将指定路径添加到系统路径中。直接修改 sys.path（暂时添加）。
    如果需要添加的路径在系统路径中已经存在，那么不再重复添加。（sys.path是list类型数据，重复添加会出现两个相同的路径）
    
    说明：
    添加为系统文件后可以直接调用该文件夹内的所有函数
    os.path.realpath(__file__): 获取当前执行脚本的绝对路径。
    os.path.dirname: 去掉文件名，返回目录。注意：只返回目录，如果需要返回文件名使用 basename
    
    Parameters
    ----------
    subfolder : 当前文件以上两层的文件夹 的 某个子文件夹，将其添加到系统路径。
    top_dir_join : 是否将 当前文件以上两层的文件夹 添加到系统路径中。默认为True表示添加。

    Returns
    -------
    无返回值，直接将路径添加到系统路径中。

    """
    # 添加 当前文件以上两层的文件夹 为系统文件。
    if top_dir_join:
        top_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        if top_dir not in sys.path:
            sys.path.append(top_dir)
    
    # 添加 当前文件以上两层的文件夹的某个子文件夹 为系统文件。其中子文件夹由subfolder给出。
    subfolder_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), subfolder)
    if subfolder_dir not in sys.path:
        sys.path.append(subfolder_dir)


def remove_duplicates_order(li):
    """
    不改变顺序对List去重，注意：sys.path也是一个list类型数据

    Parameters
    ----------
    li : 需要去重的list

    Returns
    -------
    new_li : 不改变顺序去重后的list
    
    """
    new_li=list(set(li))
    new_li.sort(key=li.index)
    return new_li


if __name__ == "__main__":
    # 导入 ../53_语言模型/d_reading_long_sequence_data.py
    subfolder = "53_语言模型"
    join_syspath_subfolder(subfolder)

