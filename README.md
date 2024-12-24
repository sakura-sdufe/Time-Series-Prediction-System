# 毕业论文程序



## 项目架构

|——main.py

|——preprocessing.py



|——data

|——|——data_scale

|——|——|——norm.py

|——|——|——min_max.py

|——|——feature_select

|——|——|——algorithms.py

|——|——data_encapsulation



|——predictor

|——|——multi_feature

|——|——|——models.py			# 记录所有模型多特征模型

|——|——single_feature

|——|——|——models.py			# 记录所有模型单特征模型

|——|——model_select



|——ensemble



|——validation



## 第四章 基于多因素集成学习的风电功率预测

### Step 1：数据清洗

- 删除没有变化的列（GenTemp）。
- 对风电机维修部分数据进行处理，将原始数据分成维修前数据和维修后数据。并且只保留后半段数据（2019/7/28 15:50 至 2020/3/12 12:20，共有 31573 数据点）

### Step 2：特征选择

- 使用 SelectKBest(score_func=mutual_info_regression) 方法进行特征选择，从19个特征中选择出15个特征，其中未被选择的特征为：Index(['NacelTemp', 'GenBearTemp', 'WindDirAbs', 'EnvirTemp']

### Step 3：多因素预测器

- **SARIMAX**： statsmodels.tsa.statespace.sarimax.SARIMAX
- **Bagging**： sklearn.ensemble.BaggingRegressor
  - **AdaBoost**： sklearn.ensemble.AdaBoostRegressor
  - **Gradient Boosting**： sklearn.ensemble.GradientBoostingRegressor
  - **XGBoost**： xgboost.XGBRegressor
  - **LightGBM**： lightgbm.LGBMRegressor
  - **CatBoost**： catboost.CatBoostRegressor
- **Stacking**： sklearn.ensemble.StackingRegressor
- **Voting**： sklearn.ensemble.VotingRegressor
- MLP、TCN、LSTM、GRU 多因素预测

### Step 4：预测器（选择和）集成

- 使用 self-attention 进行集成



