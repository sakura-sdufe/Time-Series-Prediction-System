# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Graduation
# @Time     : 2025/3/21 22:20
# @Author   : 张浩
# @FileName : main.py
# @Software : PyCharm
# @Function :
Note: 请运行完预测器再运行集成器；如果运行集成器后再想添加新的预测器，请先删除集成器结果，然后再调用 Runer.ensemble_data 方法。（未测试）
-------------------------------------------------
"""

from RUNER import Runer

from regression import SVR, Ridge, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from predictor import MLPModel, RNNModel, LSTMModel, GRUModel, TransformerWithLinear, TransformerWithAttention
from ensemble import AttentionEnsemble, AttentionProjEnsemble, C3B2H

from DLCriterion import MSELoss_scale, MSELoss_sqrt, sMAPELoss, MAPELoss

criterion = MSELoss_scale
monitor = sMAPELoss

REGRESSION_MODELS = [SVR, Ridge, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor]
REGRESSION_NORMALIZATIONS = [True, True, False, False, False, False]
PREDICTOR_MODELS = [MLPModel, RNNModel, LSTMModel, GRUModel, TransformerWithLinear, TransformerWithAttention]
PREDICTOR_NORMALIZATIONS = [True, True, True, True, True, True]
ENSEMBLE_MODELS = [AttentionEnsemble, AttentionProjEnsemble, C3B2H]
ENSEMBLE_NORMALIZATIONS = [False, False, False]


runer = Runer()  # 实例化 Runer 类


# "测试1：只运行预测器（单个模型，机器学习）"
# predictor_rf = runer.run_predictor(
#     models=SVR,
#     normalizations=True,
# )

# "测试2：只运行预测器（单个模型，深度学习，追加形式）"
# predictor_lstm = runer.run_predictor(
#     models=LSTMModel,
#     normalizations=True,
#     criterion=criterion,
#     monitor=monitor,
# )

# "测试3：只运行集成器（单个模型，机器学习，需要修改 project_parameters.py 中的 ProjectParameters.read_predictor_dir）"
# ensemble_rf = runer.run_ensemble(
#     models=GradientBoostingRegressor,
#     normalizations=False,
# )

# "测试4：只运行集成器（单个模型，深度学习，追加形式，需要修改 project_parameters.py 中的 ProjectParameters.read_predictor_dir）"
# ensemble_attnj = runer.run_ensemble(
#     models=AttentionProjEnsemble,
#     normalizations=False,
#     criterion=criterion,
#     monitor=monitor,
# )

# "测试5：只运行预测器（所有模型）"
# trained_predictors = runer.run_predictor(
#     models=REGRESSION_MODELS+PREDICTOR_MODELS,
#     normalizations=REGRESSION_NORMALIZATIONS+PREDICTOR_NORMALIZATIONS,
#     criterion=criterion,
#     monitor=monitor,
# )

# "测试6：只运行集成器（所有模型，需要修改 project_parameters.py 中的 ProjectParameters.read_predictor_dir）"
# trained_ensembles = runer.run_ensemble(
#     models=REGRESSION_MODELS+ENSEMBLE_MODELS,
#     normalizations=REGRESSION_NORMALIZATIONS+ENSEMBLE_NORMALIZATIONS,
#     criterion=criterion,
#     monitor=monitor,
# )

"测试7：预测器（所有模型）和训练器（所有模型）一起运行"
trained_predictors = runer.run_predictor(
    models=REGRESSION_MODELS+PREDICTOR_MODELS,
    normalizations=REGRESSION_NORMALIZATIONS+PREDICTOR_NORMALIZATIONS,
    criterion=criterion,
    monitor=monitor,
)
trained_ensembles = runer.run_ensemble(
    models=REGRESSION_MODELS+ENSEMBLE_MODELS,
    normalizations=REGRESSION_NORMALIZATIONS+ENSEMBLE_NORMALIZATIONS,
    criterion=criterion,
    monitor=monitor,
)


