# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation-Project
# @Time     : 2024/11/14 15:37
# @Author   : 张浩
# @FileName : predictor_parameters.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

class PredictorParameters:
    def __init__(self):
        """类内变量名需与模型类名保持一致，否则无法正常解析。"""
        # Persistence 模型参数
        self.Persistence = {}
        # SVR 模型参数
        self.SVR = {
            'kernel': 'rbf',  # 核函数，默认为 'rbf'。
            'C': 1.0,  # 正则化参数，默认为 1.0。
            'epsilon': 0.1,  # 不敏感区间，默认为 0.1。
            'gamma': 'scale',  # 核系数，默认为 'scale'。
            'degree': 3,  # 多项式核函数的次数，默认为 3。
            'coef0': 0.0,  # 核函数中的常数项，默认为 0.0。
            'shrinking': True,  # 是否使用启发式方法，默认为 True。
            'tol': 1e-3,  # 容忍误差的阈值，默认为 1e-3。
            'cache_size': 200  # 缓存大小，默认为 200。
        }
        # Ridge 模型参数
        self.Ridge = {
            'alpha': 1.0,  # 正则化强度，默认为 1.0。
            'solver': 'auto',  # 求解器，默认为 'auto'。
            'fit_intercept': True,  # 是否计算截距，默认为 True。
            'max_iter': None,  # 最大迭代次数，默认为 None。
            'random_state': None,  # 随机种子，默认为 None。
            'tol': 1e-3  # 容忍误差的阈值，默认为 1e-3。
        }
        # RandomForest 模型参数
        self.RandomForestRegressor = {
            'n_estimators': 100,  # 决策树的数量，默认为 100。
            'max_depth': None,  # 决策树的最大深度，默认为 None。
            'min_samples_split': 2,  # 内部节点再划分所需最小样本数，默认为 2。
            'min_samples_leaf': 1,  # 叶子节点最少样本数，默认为 1。
            'max_features': 'auto',  # 寻找最佳分割时的特征数，默认为 'auto'。
            'bootstrap': True,  # 是否使用自助法，默认为 True。
            'random_state': None,  # 随机种子，默认为 None。
        }
        # GradientBoostingRegressor 模型参数
        self.GradientBoostingRegressor = {
            # 梯度提升树相关参数
            'loss': 'squared_error',  # 损失函数，默认为 'squared_error'。
            'learning_rate': 0.1,  # 学习率，默认为 0.1。
            'n_estimators': 100,  # 决策树的数量，默认为 100。
            'subsample': 1.0,  # 子采样率，默认为 1.0。
            'criterion': 'friedman_mse',  # 分裂节点的评价标准，默认为 'friedman_mse'。
            'random_state': None,  # 随机种子，默认为 None。
            # 树结构相关参数
            'max_depth': 3,  # 决策树的最大深度，默认为 3。
            'min_samples_split': 2,  # 内部节点再划分所需最小样本数，默认为 2。
            'min_samples_leaf': 1,  # 叶子节点最少样本数，默认为 1。
            'max_features': None,  # 寻找最佳分割时的特征数，默认为 None。
        }
        # AdaBoostRegressor 模型参数
        self.AdaBoostRegressor = {
            'base_estimator': None,  # 基本估计器，默认为 None。
            'n_estimators': 50,  # 决策树的数量（等于弱学习器的最大迭代次数），默认为 50。
            'learning_rate': 1.0,  # 学习率，默认为 1.0。
            'loss': 'linear',  # 损失函数，默认为 'linear'。
            'random_state': None,  # 随机种子，默认为 None。
        }
        # BaggingRegressor 模型参数
        self.BaggingRegressor = {
            'base_estimator': None,  # 基础学习器。如果为 None，则默认使用 DecisionTreeRegressor。默认为 None。
            'n_estimators': 10,  # 集成中基学习器的数量。数量越多，效果可能越好，但计算成本更高。默认为 10。
            'max_samples': 1.0,  # 每个基学习器训练时使用样本的比例或数量。如果为浮点数（如 0.8），表示比例；整数表示样本数。默认为 1.0。
            'max_features': 1.0,  # 每个基学习器使用特征的比例或数量。类似于 max_samples，支持浮点数和整数。默认为 1.0。
            'bootstrap': True,  # 是否对样本进行有放回抽样。设置为 False 表示无放回抽样。默认为 True。
            'bootstrap_features': False,  # 是否对特征进行有放回抽样。默认为 False，使用所有特征。默认为 False。
            'oob_score': False,  # 是否使用袋外样本评估模型性能，仅在 bootstrap=True 时有效。默认为 False。
            'warm_start': False,  # 如果为 True，可以增量增加基学习器，而无需重新拟合模型。默认为 False。
            'n_jobs': None,  # 并行工作的数量。-1 表示使用所有可用 CPU 核心。默认为 None，表示 1。
            'random_state': None,  # 随机种子。默认为 None。
            'verbose': 0,  # 控制日志的输出量，0 表示无输出，1 表示适量输出，>1 表示详细输出。默认为 0。
        }
        # RNN 模型参数
        self.RNNModel = {
            'hidden_size': 128,  # 隐藏层节点数，默认为 128。
            'output_size': 1,  # 输出层节点数，默认为 1。
            'num_layers': 2,  # 网络层数，默认为 2。
            'bidirectional': False,  # 是否使用双向RNN，默认为 False。
        }
        # LSTM 模型参数
        self.LSTMModel = {
            'hidden_size': 128,  # 隐藏层节点数，默认为 128。
            'output_size': 1,  # 输出层节点数，默认为 1。
            'num_layers': 2,  # 网络层数，默认为 2。
            'bidirectional': False,  # 是否使用双向RNN，默认为 False。
        }
        # GRU 模型参数
        self.GRUModel = {
            'hidden_size': 128,  # 隐藏层节点数，默认为 128。
            'output_size': 1,  # 输出层节点数，默认为 1。
            'num_layers': 2,  # 网络层数，默认为 2。
            'bidirectional': False,  # 是否使用双向RNN，默认为 False。
        }
        # TransformerWithLinear 模型参数
        self.TransformerWithLinear = {
            'output_size': 1,  # 输出层节点数，默认为 1。
            'encoder_model_dim': 128,  # 编码器 TransformerEncoderLayer 模型维度，默认为 128。
            'encoder_head_num': 8,  # 编码器 TransformerEncoderLayer 多头注意力机制的头数，默认为 8。
            'encoder_feedforward_dim': 2048,  # 编码器 TransformerEncoderLayer 前馈神经网络的隐藏层维度，默认为 2048。
            'encoder_layer_num': 2,  # 编码器 TransformerEncoderLayer 层数，默认为 2。
            'decoder_hidden_sizes': [1024, 256, 32],  # 解码器全连接层的隐藏层维度列表，默认为 None，表示直接映射到输出维度。
            'activation': 'relu',  # 编码器和解码器的激活函数，默认为 'relu'。
            'dropout': 0.1,  # 编码器 TransformerEncoderLayer 和 解码器全连接层的 dropout 概率，默认为 0.1。
            'max_length': 100,  # 位置编码的最大长度，默认为 1000。主要用于位置编码。注：该参数值必须要大于时间步。
        }
        # TransformerWithAttention 模型参数
        self.TransformerWithAttention = {
            'output_size': 1,  # 输出层节点数，默认为 1。
            'encoder_model_dim': 128,  # 编码器 TransformerEncoderLayer 模型维度，默认为 128。
            'encoder_head_num': 8,  # 编码器 TransformerEncoderLayer 多头注意力机制的头数，默认为 8。
            'encoder_feedforward_dim': 2048,  # 编码器 TransformerEncoderLayer 前馈神经网络的隐藏层维度，默认为 2048。
            'encoder_layer_num': 2,  # 编码器 TransformerEncoderLayer 层数，默认为 2。
            'decoder_model_dim': 128,  # 解码器 MultiHeadAttention 模型维度，默认值为 128。
            'decoder_head_num': 8,  # 解码器 MultiHeadAttention 多头注意力机制的头数，默认值为 8。
            'decoder_feedforward_dim': 2048,  # 解码器 MultiHeadAttention 前馈神经网络的隐藏层维度，默认值为 2048。
            'decoder_layer_num': 2,  # 解码器 MultiHeadAttention 层数，默认值为 2。
            'activation': 'relu',  # 编码器和解码器的激活函数，默认为 'relu'。
            'dropout': 0.1,  # 编码器 TransformerEncoderLayer 和 解码器 MultiHeadAttention 的 dropout 概率，默认为 0.1。
            'max_length': 100,  # 位置编码的最大长度，默认为 1000。主要用于位置编码。注：该参数值必须要大于时间步。
        }
        self.DL_train = {
            'epochs': 150,  # 训练轮数，默认为 150。
            'learning_rate': 1e-3,  # 学习率，默认为 1e-3。
            'clip_norm': None,  # 梯度裁剪阈值，默认为 None，表示不裁剪。
            'ReduceLROnPlateau_factor': 0.5,  # 学习率衰减因子，默认为 0.5。
            'ReduceLROnPlateau_patience': 10,  # 监测器函数不再减小的累计次数，默认为 10。
            'ReduceLROnPlateau_threshold': 1e-4,  # 只关注超过阈值的显著变化，默认为 1e-4。
        }

    def __getitem__(self, item):
        return getattr(self, item)

    def __iter__(self):
        return iter(self.__dict__.items())

    def items(self):
        return self.__dict__.items()
