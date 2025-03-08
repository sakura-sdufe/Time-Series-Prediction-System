- # Time-Series-Prediction-System

  ## 项目介绍

  **该项目是一个支持机器学习和深度学习的多因素时间序列预测系统，旨在能够根据历史数据和可能影响时间序列的外部因素预测时间序列的未来值。**

  ### 项目特点

  - 本项目提供了丰富的接口和简便的实现方法，特别是针对深度学习模型的构建，用户只需提供模型类就可以自动执行训练、评估、绘图、保存等功能。
  - 本项目已经内置了 sklearn 常用回归模型（SVR、Ridge、RandomForestRegressor、GradientBoostingRegressor、AdaBoostRegressor、BaggingRegressor）、多层感知机（MLP）、循环神经网络系列模型（RNN、LSTM、GRU）和 Transformer系列模型（TransformerWithLinear、TransformerWithAttention）。用户可以提供自定义的模型类（需继承自 ModelBase 类），无需考虑训练、评估、展示、保存等繁杂工作，该项目可以自动实现以上功能（**仅支持来自 sklearn 和 pytorch 的模型、仅支持未来一步的时间序列预测任务**）。
  - 本项目提供了 ensemble 模块，用户可以通过传入预测输出结果的路径就可以将预测结果进一步组合，得到更稳定的预测结果。同样地，用户只需提供模型类（需继承自 EnsembleBase 类）就可以自动执行训练、评估、绘图、保存等功能。
  - 本项目已经内置了基于卷积神经网络（CNNs）和注意力机制（Attention）的两种组合模型。
  - 本项目用户可以自己决定是否输出信息到控制台、是否保存结果以及保存什么结果，具有较高的可拓展性和个性化设计。

  ### 环境要求

  本项目中的程序大部分都是基于第三方模块的基础功能，没有严格的版本要求，以下为测试环境仅供参考：

  Python = 3.10 </br>
  numpy = 1.26.3 </br>
  pandas = 2.2.3 </br>
  matplotlib = 3.10.0 </br>
  scikit-learn = 1.6.1 </br>
  torch = 2.5.1+cu124 </br>
  tqdm = 4.67.1 </br>

  ### 参数介绍

  1. 该项目将外部因素特征分为时变已知特征（可以提前知道真实值的特征）和时变未知特征（无法提前预知的特征），需要用户根据自身需求在 **data_parameters.py** 文件中的 **DataParameters** 类设置参数。该项目将会根据设置的特征信息处理数据，所有未设置的特征将会被忽略。
  2. 该项目可以一次性训练并推理多个机器学习模型和多个深度学习模型，所有的模型参数都存放在 **predictor_parameters.py** 文件中 **PredictorParameters** 类中。我已经预设了一些常用的机器学习模型和深度学习模型，用户可以选择使用这些模型并且修改参数，也可以建立自己的模型并且把参数存放到 **PredictorParameters** 类中。
  3. 该项目可以一次性运行多个组合模型并保存结果，组合模型的参数都存放在 **ensemble_parameters.py** 文件中**EnsembleParameters**类中。我已经预设了基于卷积神经网络和注意力机制的两种组合模型，用户可以选择使用这些模型并且修改参数，也可以建立自己的模型并且把参数存到 **EnsembleParameters** 类中。
  4. 该项目会自动保存运行日志、用户设置的参数、预测走势和真实值图像、训练后模型、预测结果、评价指标；如果是深度学习模型，那么还会保存每次迭代的损失函数值、监控函数值、学习率和每秒样本数，以及对应的图像。在 **project_parameters.py** 文件中的 **ProjectParameters** 类中用户可以选择每次运行时是否删除同名的运行结果（包含安全检查）以及运行结果的写入形式（如果用户想要在同一个运行结果中继续追加新的运行结果并汇入一个表中，这个功能是非常有用的）

  ### 生成预测结果

  1. 生成的结果将会保存到 **result** 目录下，结果的根目录由 **ProjectParameters** 类中的 ***"self.save_dir"*** 变量决定。保存的目录包括 **"data", "documents", "figures", "models", "results" 和 ".result"**，其中： 
     - **"data"** 保存机器学习模型的特征和目标，深度学习模型的 DataLoader；
     - **"documents"** 保存日志信息和用户设置的参数信息（如果运行出现错误也会保存错误信息）；
     - **"figures"** 保存预测走势图、深度学习损失函数图和监控函数图；
     - **"models"** 保存训练后的模型；
     - **"results"** 保存预测结果、评估指标结果，如果是深度学习模型还会保存损失函数值、监控器值、学习率和训练每秒样本数。
     - **".result"** 用于识别当前目录是否由该系统创建，用于识别预测结果和删除保护。
  	
  2. 在预测结果中 **True** 列表示目标值，为了防止机器学习数据划分和深度学习数据划分没有对齐，本系统还引入了检测功能，如果出现未对齐情况会抛出 assert 错误（正常传入特征和目标不会出现这种情况）。
  3. 在深度学习模型中，最终用于预测和评估的模型不是最后一次迭代的模型，而是整个训练周期中，验证集在监控函数中表现最好（监控器取最低值）的模型。
     - 每次验证集在监控函数中出现下降情况时，都会保存该模型（文件名记录当前迭代次数和验证集的监控值）。
     - 来自 sklearn 的机器学习模型保存格式为 **.pkl**，来自 pytorch 的深度学习模型保存格式为 **.pth**。

  ### 运行介绍

  1. 打开 **data_parameters.py** 文件中的 **DataParameters** 类，自定义数据处理相关参数。尤其要将 **self.target, self.time_unknown_variables 和 self.time_known_variables** 三个参数修改为自己文件的形式，否则会找不到特征和目标。
  2. 打开 **predictor_parameters.py** 文件中的 **PredictorParameters** 类，自定义预测模型相关参数。
  3. 打开 **ensemble_parameters.py** 文件中的 **EnsembleParameters** 类，自定义组合模型相关参数。
  4. 打开 **project_parameters.py** 文件中的 **ProjectParameters** 类，自定义保存相关参数。尤其检查保存路径，防止后续找不到运行结果，必要时可以修改为自定义的文件路径。
  5. 打开 **predictor_main.py** 文件，将 "读取文件" 和 "获取特征（DataFrame）和目标（Series）" 相关代码（**第48行至50行**）修改为自己的文件及路径。用户可以选择将 data 直接读取为 DataFrame 类型数据。
  6. 【可选】打开 **DLCriterion.py** 文件，自定义损失类和监控器类，并在 **main.py** 文件中深度学习模型参数 **criterion** 和 **monitor** 修改为自定义的类。
  7. 打开 **predictor_main.py** 文件，选择运行单个模型还是运行所有模型（**第67行至第89行**）。
  8. 运行项目。

  ## 项目模块介绍

  如果您想深度适配您的数据或工作，您可以阅读下面的内容，以便您更好的了解整个项目的逻辑。我将尽我最大可能介绍主要模块的内容与实现的功能。

  ### 基础模块（Base）

  - **Base.py** 文件中的 **BaseModel** 类是所有自定义深度模型的基类（用于预测），它内部定义了 **train_epoch**（训练过程中一个迭代周期）、**fit**（深度学习模型训练）、**predict**（深度学习模型推理）、**evaluate**（深度学习模型评估）、**to_device**（模型转移到指定设备）、**save**（保存模型）方法。
    - 在子类中需要**重写 \_\_init\_\_ 和 forward 方法**。
    - 如果自定义的模型的 forward 方法输出不是预测结果或包含其他内容，那么需要**重写 train_epoch 和 predict 方法。**
    - **BaseModel** 类中的 **fit** 方法会**自动训练深度学习模型**、**保存最优模型**、**控制台打印并保存损失值和监测值**、**绘制动态损失值图像和监测值图像**、以及**保存所有训练结果**等操作（这些过程用户都可以通过传入参数自行决定是否执行）。
  - **Ensemble.py** 文件中的 **EnsembleBase** 类是所有自定义深度模型的基类（用于集成），它继承自 **BaseModel** 类。由于集成数据集样式和预测数据集样式不同，所以 EnsembleBase 类重写了 **train_epoch** 和 **predict** 方法，以便后续集成模型直接调用。（如果自己定义不同的数据输入输出格式，也可以采用重写 **train_epoch** 和 **predict** 方法解决）

  ### 数据封装模块（data）

  - **data_encapsulation** 模块实现了对机器学习数据和深度学习数据的封装。
    - **predictor_ML.py** 文件中的 **SeqSplit** 类实现了对机器学习数据的划分和封装，该类支持添加目标的历史数据（可选是否添加特征的历史数据）、标准化数据、划分数据集、生成可用于机器学习的数据集等功能。
    - **predictor_DL.py** 文件中的 **SeqLoader** 类实现了对深度学习数据的封装，该类支持添加的历史数据、标准化数据、划分数据集、生成可用于深度学习的 dataloader 等功能。
    - **ensemble_ML.py** 文件中的 **EnsembleSplit** 类实现了集成数据在机器学习上的封装。
    - **ensemble_DL.py** 文件中的 **EnsembleLoader** 类实现了集成数据在深度学习上的封装。
    - _Note 1_: **SeqLoader** 类支持 **sample_gap** 参数，可以为用于训练的训练集设置采样间隔，以降低过拟合风险。但是，用于评估的训练集、验证集和测试集并不会设置采样间隔，以保证评估的准确性。例如上一个样本的特征时间步为 [n, n+1, $\cdots$, n+time_step-1]，则下一个样本的特征时间步为 [n+sample_gap, n+sample_gap+1, $\cdots$, n+sample_gap+time_step-1]。
    - _Note 2_：预测器封装类（SeqSplit 和 SeqLoader）适用于具有时间特征的数据；集成器封装类（EnsembleSplit 和 EnsembleLoader）适用于不具时间特征的数据。
  - **data_scale** 模块实现了对数据的标准化。
    - **Norm.py** 文件中的 **Norm** 类实现了对 2Darray 和 DataFrame 的标准化，该类支持对数据的标准化和反标准化。与此同时，该类还支持应用指定列的标准化信息到新传入的数据中。
    - ~~**min_max.py** 文件中的 **MinMax** 类暂未实现。~~
  - **data_selection.py** 文件将 **sklearn.SelectKBest** 模块进一步封装。

  ### 预测器模块（predictor）

  - **DLModel** 模块实现了深度学习系列模型的训练、预测和评估。当前仅支持：<u>**多层感知机（MLP）、RNNs 系列模型（包括：RNN、LSTM 和 GRU）和 Transformers 系列模型（包括：TransformerWithLinear 和 TransformerWithAttention）**</u>。用户可以在本目录下定义自己的模型（继承 ModelBase 类）。
    - **MLP.py** 文件中的 **MLPModel** 类继承自 ModelBase，通过传入不同隐藏层的大小可以生成多层感知机模型。
    - **RNNs.py** 文件中的 **RNNModelBase** 类继承自 ModelBase，通过重写 forward、train_epoch、predict 方法和添加 begin_state 方法实现循环神经网络系列模型。**RNNModel、LSTMModel、GRUModel** 类分别实现 RNN、LSTM 和 GRU 模型，这些模型均继承自 RNNModelBase 类。
    - **Transformers.py** 文件包含 **TransformerWithLinear** 和 **TransformerWithAttention** 两个模型。
      - TransformerWithLinear  的编码器为多层 Transformer 编码器，解码器为多个可自定义的全连接层（包含残差结构）。
      - TransformerWithAttention 的编码器为多层 Transformer 编码器，解码器由多个多头自注意力和前馈全连接层构成（包括残差结构）的 block 组成。
      - Transformers.py 文件中的 PositionalEncoding 类实现位置编码操作，AttentionLinearLayer 类实现由多头自注意力和前馈全连接层构成的 block，Decoder 类将解码器层构建成解码器。
    - ~~**TCN.py** 文件暂未实现时域卷积网络。~~
  - **predictor.py** 文件包含 **Predictors 类**，**<u>它支持来自 sklearn 的机器学习模型也支持来自 pytorch 的深度学习模型的训练、评估和保存</u>**。主要的方法有：
    - **ML** 方法实例化、训练和预测机器学习模型。
    - **DL** 方法实例化、训练和预测深度学习模型。
    - **model** 方法训练和评估一个模型，并保存结果和评估指标。
    - **persistence** 方法计算 Persistence 模型的预测结果和评估指标。Persistence 模型表示使用前一时间步的结果作为当前时间步的预测结果。
    - **all_models** 方法训练和评估多个模型，并保存结果和评估指标（会自动添加 Persistence 模型，用户无需显式添加）。

  ### 集成器模块（ensemble）

  - **DLModel** 模块实现了深度学习系列模型的训练、预测和评估（用于集成部分）。当前仅支持：<u>**基于卷积神经网络和基于注意力机制的两类模型**</u>。用户可以在本目录下定义自己的模型（继承 EnsembleBase 类）。
    - **CNNs.py** 文件中的 **C2L** 模型由两个卷积和一个线性层构成；**C3B2H** 模型是由 CBA、Bottleneck 和 Conv1d 构成。（如果集成模型的输入数据没有标准化操作，推荐使用不带 BatchNorm1d 的 C2L 模型）
    - **Attentions.py** 文件中的 **Attention** 类通过一维卷积对多头注意力的 embed_dim 维度进行映射，从而使得注意力机制获得更佳的结果；**AttentionEnsemble** 模型由 Attention 类和输出层线性映射构成；**AttentionProEnsemble** 模型由 输入层线性映射、Attention 类、ResNet 结构和 FeedForward 构成。
  - **ensemble.py** 文件包含 **Ensembles类，它支持来自 sklearn 的回归模型也支持来自 pytorch 的深度学习模型的训练、评估和保存**。主要的方法有：
    - **ML** 方法实例化、训练和预测机器学习模型。
    - **DL** 方法实例化、训练和预测深度学习模型。
    - **model** 方法训练和评估一个模型，并保存结果和评估指标。
    - **all_models** 方法训练和评估多个模型，并保存结果和评估指标。
    - **Note：未测试 sklearn 的回归模型**

  ### 指标模块（metrics）

  - **metrics.py** 文件中的 **calculate_metrics** 函数实现了多种不同的评估指标，已经内置了 **sMAPE、MAPE、RMSE、MSE、MAE、R2** 指标。默认输出所有指标。

  ### 工具模块（utils）

  - **Accumulator.py** 文件中的 Accumulator 类是一个累加器。
  - **Animator.py** 文件中的 Animator 类是一个动态绘图类。
  - **Cprint.py** 文件中的 cprint 函数是基于 ANSI 转义序列实现的控制台彩色打印函数。
  - **device_gpu.py** 文件中的 try_gpu 函数可以判断当前 'cuda' 是否可用。
  - **Timer.py** 文件中的 Timer 类用于记录时间。
  - **Writer.py** 文件中的 Writer 类可以暂存和保存表数据、文本数据、参数数据和文件数据，也可以用于保存机器学习和深度学习模型，还可以绘制和保存图像。Writer 类支持删除上次输出结果，也可以在上次输出结果中继续追加新数据。

  ### 其他文件

  - **data_parameters.py** 文件中的 **DataParameters** 类记录了所有与数据处理相关的参数，用户可以修改这些参数自定义数据处理的方式。

  - **predictor_parameters.py** 文件中的 **PredictorParameters** 类记录了所有与预测模型相关的参数，用户可以修改这些参数以更好地适应不同的数据集。
    - **类内变量名必须要与模型类名保持一致**，否则使用 Predictors 类的 all_models 方法无法解析。
    - **self.DL_train** 变量存放的是深度学习模型训练参数，所有深度学习模型共用一套训练参数。
  - **ensemble_parameters.py** 文件中的 **EnsembleParameters** 类记录了所有与集成模型相关的参数，用户可以修改模型参数和训练参数适应不同的数据集。（修改方式和注意事项与 **PredictorParameters** 一致）

  - **project_parameters.py** 文件中的 **ProjectParameters** 类存放运行结果的保存形式和地址。
  - **parameters.py** 文件导入 data_parameters.py 文件中的 **DataParameters** 类、predictor_parameters.py 文件中的 **PredictorParameters** 类、ensemble_parameters.py 文件中的 **EnsembleParameters** 和 project_parameters.py 中的 **ProjectParameters** 类。
  - **DLCriterion.py** 文件自定义了四个损失函数（监测器函数）。
    - **MSELoss_scale** 类在 nn.MSELoss 上乘以了一个放缩系数。
    - **MSELoss_sqrt** 类在 nn.MSELoss 上做了个幂操作。
    - **sMAPELoss** 类自定义了 sMAPE 监测器函数（损失函数）。
    - **MAPELoss** 类自定义了 MAPE 监测器函数（损失函数）。
  - **predictor_main.py** 为预测主程序。主要内容包括：初始化参数类、初始化 Writer 类、读取文件、获取 DataFrame 类型的特征和 Series 类型的目标、执行特征选择并添加到日志、实例化机器学习数据封装类和实例化深度学习数据封装类、实例化预测器类、选择预测模型和是否标准化操作、将所有暂存内容全部写入本地文件。（程序运行过程中如果遇到任何继承于 Exception 的错误，都将会保存错误至 **documents\\Logs.log** 下，然后将所有暂存的结果全部保存到本地，防止数据丢失）
  - **ensemble_main.py** 为集成主程序。主要内容和错误捕获与 **predictor_main.py** 类似，但是该文件的数据封装部分只需要传入预测器结果目录即可。

  ## 注意事项

  当您需要创建自己的深度学习模型用于未来一步的时间序列预测，您需要做以下工作：

  1. 准备好您的数据，并尝试该项目是否能在您的数据上运行。
  2. 【可选】在 **predictor/DLModel** 目录下自定义用于预测的深度学习模型（继承 ModelBase 类，重写 \_\_init\_\_ 和 forward 方法；如果 forward 方法的返回值不是预测结果或包含其他内容，那么您还需要重写 train_epoch 和 predict 方法）。<u>如果仅需重写 \_\_init\_\_ 和 forward 方法，您可以参考 TransformerWithLinear 和 TransformerWithAttention 的实现过程；如果您还需要重写 train_epoch 和 predict 方法或者需要添加新的方法，您可以参考 RNNModelBase、RNNModel、LSTMModel、GRUModel 的实现过程。</u>
  3. 【可选】在 **ensemble/DLModel** 目录下自定义用于集成的深度学习模型。
  4. 【可选】在 **metrics/metrics.py** 文件下的 **calculate_metrics** 函数中自定义评价指标。
  5. 在 **data_parameters.py** 文件中的 **DataParameters** 类中调整数据处理（数据封装）相关的参数。
  6. 在 **predictor_parameters.py** 文件中的 **PredictorParameters** 类中调整模型相关的参数。**如果有自定义模型，一定要在这里添加模型参数，类内变量名应当与类名保持一致。**
  7. 在 **ensemble_parameters.py** 文件中的 **EnsembleParameters** 类中调整模型相关的参数。**如果有自定义模型，一定要在这里添加模型参数，类内变量名应当与类名保持一致。**
  8. 在 **project_parameters.py** 文件中的 **ProjectParameters** 类中调整保存设置和路径设置。
  9. 在 **predictor_main.py** 文件下选择您需要训练的模型，可以是单个模型也可以是所有模型。
  10. 运行 **predictor_main.py** 文件。
  11. 打开保存路径查看预测结果。
  12. 在 **ensemble_main.py** 文件下选择您需要训练的模型，可以是单个模型也可以是所有模型。
  13. 运行 **ensemble_main.py** 文件。
  14. 打开保存路径查看集成结果。
