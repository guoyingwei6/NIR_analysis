#!/usr/bin/env python
# coding: utf-8

# ## 环境配置，函数定义

# In[1]:


import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import CARS


# In[5]:


# ---------------------------
# SPXY 样本划分函数
# ---------------------------
def spxy_partition(X, y, train_fraction=0.8):
    """
    SPXY 样本集合划分，基于自变量 (X) 和因变量 (y) 联合标准化后的欧氏距离。
    
    参数:
      X: 自变量数组（二维），例如预处理后的光谱数据，每行代表一个样本。
      y: 因变量数组（一维），例如水分含量。
      train_fraction: 训练集比例（默认 0.8）。
      
    返回:
      train_indices: 训练集样本索引列表。
      test_indices: 测试集样本索引列表。
    """
    n_samples = X.shape[0]
    scaler_X = StandardScaler()
    X_norm = scaler_X.fit_transform(X)
    scaler_y = StandardScaler()
    y_norm = scaler_y.fit_transform(y.reshape(-1, 1))
    
    # 拼接自变量和因变量
    Z = np.hstack((X_norm, y_norm))
    # 计算欧氏距离矩阵
    dist_matrix = squareform(pdist(Z, metric='euclidean'))
    
    # 初始选择距离最大的两个样本
    i, j = np.unravel_index(np.argmax(dist_matrix, axis=None), dist_matrix.shape)
    train_indices = [i, j]
    n_train = int(np.floor(train_fraction * n_samples))
    
    while len(train_indices) < n_train:
        remaining = list(set(range(n_samples)) - set(train_indices))
        min_dists = [np.min(dist_matrix[r, train_indices]) for r in remaining]
        selected = remaining[np.argmax(min_dists)]
        train_indices.append(selected)
    
    test_indices = list(set(range(n_samples)) - set(train_indices))
    return train_indices, test_indices


# In[6]:


# ---------------------------
# SNV 预处理函数
# ---------------------------
def SNV_transform(X):
    """
    对光谱数据进行标准正态变换（Standard Normal Variate, SNV）
    
    参数:
      X: numpy 数组, 大小为 (n_samples, n_features)
    返回:
      X_snv: SNV 处理后的光谱数据
    """
    # 每个样本独立标准化: 减去每行均值，再除以每行标准差（ddof=1 使用样本标准差）
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, ddof=1, keepdims=True)
    X_snv = (X - mean) / std
    return X_snv


# In[7]:


def MSC_transform(X):
    """
    对光谱数据进行多元散射校正 (MSC)
    
    参数:
      X: numpy 数组，形状为 (n_samples, n_features)，光谱数据
      
    返回:
      X_msc: MSC 处理后的光谱数据，形状与 X 相同
    """
    # 计算所有样本的平均光谱作为参考光谱
    mean_spectrum = np.mean(X, axis=0)
    X_msc = np.zeros_like(X)
    
    # 对每个样本进行线性回归拟合参考光谱
    for i in range(X.shape[0]):
        # 用 np.polyfit 拟合一次线性关系：x_i ~ b * mean_spectrum + a
        b, a = np.polyfit(mean_spectrum, X[i, :], 1)
        # MSC 校正： (x - a) / b
        X_msc[i, :] = (X[i, :] - a) / b
    return X_msc


# In[32]:


# ---------------------------
# 选择最佳因子数（n_components） - 使用交叉验证
# ---------------------------
def select_best_n_components(X_train, y_train, max_components=20):
    """
    选择最佳的因子数（n_components）来最小化RMSECV。
    
    参数:
      X_train: 训练集的光谱数据
      y_train: 训练集的水分含量
      max_components: 最大因子数（默认 20）
    
    返回:
      best_n_components: 最佳的因子数
      rmsecvs: 各因子数对应的RMSECV
    """
    rmsecvs = []
    
    # 使用 KFold 交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5折交叉验证

    # 计算不同因子数下的RMSECV
    for n in range(1, max_components+1):
        pls = PLSRegression(n_components=n)
        
        # 使用交叉验证计算RMSE
        rmse = np.sqrt(-cross_val_score(pls, X_train, y_train, cv=kf, scoring='neg_mean_squared_error'))
        rmsecvs.append(rmse.mean())  # 取每个因子数下RMSE的均值

    # 找到RMSECV最小的因子数
    best_n_components = np.argmin(rmsecvs) + 1
    return best_n_components, rmsecvs


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from scipy.signal import savgol_filter
import pandas as pd


def mccv_remove_outliers_and_plot(X_raw, y, n_iterations=100, test_size=0.3, threshold_factor=2, window_length=11, polyorder=2, deriv=1):
    """
    利用蒙特卡洛交叉验证(MCCV)方法剔除异常样本，同时计算残差均值与方差，并进行可视化。
    
    参数:
      X_raw: 光谱数据（自变量）
      y: 目标变量（如水分含量）
      n_iterations: MCCV 迭代次数
      test_size: 每次随机拆分时测试集比例
      threshold_factor: 阈值因子，用于计算异常样本
      window_length, polyorder, deriv: SG滤波器参数
      
    返回:
      X_raw_clean: 剔除异常样本后的光谱数据
      y_clean: 剔除异常样本后的目标变量
      mean_residuals: 每个样本的残差均值
      var_residuals: 每个样本的残差方差
    """
    sample_indices = np.arange(len(y))  # 样本索引
    
    # 用于存储每个样本的残差
    residuals_dict = {idx: [] for idx in sample_indices}
    
    # 存储每个样本的误差
    errors = {idx: [] for idx in sample_indices}

    for i in range(n_iterations):
        # 随机划分训练集和测试集
        X_train_raw, X_test_raw, y_train, y_test, idx_train, idx_test = train_test_split(
            X_raw, y, sample_indices, test_size=test_size, random_state=i
        )
        
        # 对训练集和测试集进行Savitzky-Golay预处理
        X_train_proc = savgol_filter(X_train_raw, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)
        X_test_proc = savgol_filter(X_test_raw, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)
        
        # 训练PLS模型
        pls = PLSRegression(n_components=20)
        pls.fit(X_train_proc, y_train)
        
        # 测试集预测
        y_pred = pls.predict(X_test_proc).ravel()
        
        # 记录测试集中每个样本的残差
        for idx, real_val, pred_val in zip(idx_test, y_test, y_pred):
            residual = abs(real_val - pred_val)
            residuals_dict[idx].append(residual)
            
            # 记录每个样本的误差
            errors[idx].append(abs(real_val - pred_val))
    
    # 计算每个样本的残差均值和方差
    mean_residuals = []
    var_residuals = []
    
    for idx in sample_indices:
        res_list = residuals_dict[idx]
        if len(res_list) == 0:
            # 如果该样本从未出现在测试集中，可以设定为0或NaN
            mean_residuals.append(np.nan)
            var_residuals.append(np.nan)
        else:
            mean_val = np.mean(res_list)
            var_val = np.var(res_list, ddof=1)  # 使用样本方差
            mean_residuals.append(mean_val)
            var_residuals.append(var_val)
    
    # 计算异常样本阈值
    avg_errors_series = pd.Series({idx: np.mean(err_list) for idx, err_list in errors.items()})
    thresh = avg_errors_series.mean() + threshold_factor * avg_errors_series.std()
    print("MCCV异常判断阈值: {:.5f}".format(thresh))
    
    # 找到异常样本
    outlier_indices = avg_errors_series[avg_errors_series > thresh].index
    print("检测到异常样本数量:", len(outlier_indices))
    print("异常样本索引:", outlier_indices.tolist())
    
    # 剔除异常样本
    X_raw_clean = np.delete(X_raw, outlier_indices, axis=0)
    y_clean = np.delete(y, outlier_indices, axis=0)
    
    # 绘制残差均值与方差的散点图
    plot_mean_variance(mean_residuals, var_residuals, sample_indices, outlier_indices)
    
    return X_raw_clean, y_clean, mean_residuals, var_residuals


def plot_mean_variance(mean_residuals, var_residuals, sample_indices, outlier_indices):
    """
    绘制残差均值-方差散点图，并标注异常样本
    
    参数:
      mean_residuals: 每个样本残差均值的数组
      var_residuals: 每个样本残差方差的数组
      sample_indices: 样本索引（原始DataFrame的index）
      outlier_indices: 异常样本的索引
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(mean_residuals, var_residuals, color='blue', alpha=0.6, label='Samples')
    
    # 标记异常样本
    for idx in outlier_indices:
        i = np.where(sample_indices == idx)[0][0]  # 获取索引位置
        plt.scatter(mean_residuals[i], var_residuals[i], color='red')
        plt.text(mean_residuals[i], var_residuals[i], str(idx), color='red', fontsize=9)
    
    plt.xlabel("Mean of Residuals")
    plt.ylabel("Variance of Residuals")
    plt.title("Mean vs. Variance of Residuals (MCCV)")
    plt.legend()
    plt.show()


# In[30]:


'''import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from scipy.signal import savgol_filter

def mccv_remove_outliers_and_plot(df, n_iterations=100, test_size=0.3, threshold_factor=2, window_length=11, polyorder=2, deriv=1):
    """
    利用蒙特卡洛交叉验证(MCCV)方法剔除异常样本，同时计算残差均值与方差，并进行可视化。
    
    参数:
      df: DataFrame，已去除第一列（样本名称）的数据，要求：第0列为水分含量（目标变量），第3列开始为光谱数据（自变量）。
      n_iterations: MCCV 迭代次数
      test_size: 每次随机拆分时测试集比例
      threshold_factor: 阈值因子，用于计算异常样本
      window_length, polyorder, deriv: SG滤波器参数
      
    返回:
      df_clean: 剔除异常样本后的 DataFrame
      mean_residuals: 每个样本的残差均值
      var_residuals: 每个样本的残差方差
      sample_indices: 样本索引，用于可视化时标注
    """
    # 提取光谱数据和目标变量
    X_raw = df.iloc[:, 3:].values  # 光谱数据从第4列开始
    y = df.iloc[:, 0].values  # 水分含量（目标变量）
    sample_indices = df.index.values  # 保留原始数据的索引
    
    # 用于存储每个样本的残差
    residuals_dict = {idx: [] for idx in sample_indices}
    
    # 存储每个样本的误差
    errors = {idx: [] for idx in sample_indices}

    for i in range(n_iterations):
        # 随机划分训练集和测试集
        X_train_raw, X_test_raw, y_train, y_test, idx_train, idx_test = train_test_split(
            X_raw, y, sample_indices, test_size=test_size, random_state=i
        )
        
        # 对训练集和测试集进行Savitzky-Golay预处理
        X_train_proc = savgol_filter(X_train_raw, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)
        X_test_proc = savgol_filter(X_test_raw, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)
        
        # 训练PLS模型
        pls = PLSRegression(n_components=20)
        pls.fit(X_train_proc, y_train)
        
        # 测试集预测
        y_pred = pls.predict(X_test_proc).ravel()
        
        # 记录测试集中每个样本的残差
        for idx, real_val, pred_val in zip(idx_test, y_test, y_pred):
            residual = abs(real_val - pred_val)
            residuals_dict[idx].append(residual)
            
            # 记录每个样本的误差
            errors[idx].append(abs(real_val - pred_val))
    
    # 计算每个样本的残差均值和方差
    mean_residuals = []
    var_residuals = []
    
    for idx in sample_indices:
        res_list = residuals_dict[idx]
        if len(res_list) == 0:
            # 如果该样本从未出现在测试集中，可以设定为0或NaN
            mean_residuals.append(np.nan)
            var_residuals.append(np.nan)
        else:
            mean_val = np.mean(res_list)
            var_val = np.var(res_list, ddof=1)  # 使用样本方差
            mean_residuals.append(mean_val)
            var_residuals.append(var_val)
    
    # 计算异常样本阈值
    avg_errors_series = pd.Series({idx: np.mean(err_list) for idx, err_list in errors.items()})
    thresh = avg_errors_series.mean() + threshold_factor * avg_errors_series.std()
    print("MCCV异常判断阈值: {:.5f}".format(thresh))
    
    # 找到异常样本
    outlier_indices = avg_errors_series[avg_errors_series > thresh].index
    print("检测到异常样本数量:", len(outlier_indices))
    print("异常样本索引:", outlier_indices.tolist())
    
    # 剔除异常样本
    df_clean = df.drop(index=outlier_indices)
    
    # 绘制残差均值与方差的散点图
    plot_mean_variance(mean_residuals, var_residuals, sample_indices, outlier_indices)
    
    return df_clean, mean_residuals, var_residuals, sample_indices


def plot_mean_variance(mean_residuals, var_residuals, sample_indices, outlier_indices):
    """
    绘制残差均值-方差散点图，并标注异常样本
    
    参数:
      mean_residuals: 每个样本残差均值的数组
      var_residuals: 每个样本残差方差的数组
      sample_indices: 样本索引（原始DataFrame的index）
      outlier_indices: 异常样本的索引
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(mean_residuals, var_residuals, color='blue', alpha=0.6, label='Samples')
    
    # 标记异常样本
    for idx in outlier_indices:
        i = np.where(sample_indices == idx)[0][0]  # 获取索引位置
        plt.scatter(mean_residuals[i], var_residuals[i], color='red')
        plt.text(mean_residuals[i], var_residuals[i], str(idx), color='red', fontsize=9)
    
    plt.xlabel("Mean of Residuals")
    plt.ylabel("Variance of Residuals")
    plt.title("Mean vs. Variance of Residuals (MCCV)")
    plt.legend()
    plt.show()


# 使用修改后的mccv_remove_outliers_and_plot函数
#df_clean, mean_res, var_res, idx_all = mccv_remove_outliers_and_plot(data, n_iterations=100, test_size=0.3, threshold_factor=2)
'''


# # 水分

# ## 剔除异常样本

# In[9]:


# ---------------------------
# 主流程
# ---------------------------
# 1. 读取数据并删除第一列（样本名称）
data_raw = pd.read_csv('NIR_WT_CF_CP.average.csv', header=0)
data = data_raw.iloc[:, 1:]  # 删除第一列
print("数据尺寸（删除样本名称后）：", data.shape)
#print("列名：", data.columns.tolist())
# 数据结构说明：
# 第0列：水分含量（目标变量）
# 第1-2列：其他营养物质（不用于建模）
# 第3列到最后：光谱数据（自变量）


# In[ ]:


X_raw = data.iloc[:, 3:].values 
y = data.iloc[:, 0].values


# In[ ]:


# 假设 X_raw 是光谱数据，y 是目标变量
X_raw_clean, y_clean, mean_res, var_res = mccv_remove_outliers_and_plot(X_raw, y, n_iterations=100, test_size=0.3, threshold_factor=2)
print("剔除异常样本后数据尺寸:", X_raw_clean.shape)


# In[10]:


# 2. 利用 MCCV 剔除异常样本
#data_clean, outlier_indices = mccv_remove_outliers(data, n_iterations=100, test_size=0.3, threshold_factor=2)
#print("剔除异常样本后数据尺寸:", data_clean.shape)


# In[31]:


#df_clean, mean_res, var_res, idx_all = mccv_remove_outliers_and_plot(data, n_iterations=100, test_size=0.3, threshold_factor=2)


# In[12]:


# 3. 提取目标变量与光谱数据
#y_clean = data_clean.iloc[:, 0].values      # 水分含量作为目标（第0列）
#X_raw_clean = data_clean.iloc[:, 3:].values   # 光谱数据（从第4列开始）


# ### 预处理前的光谱曲线和平均光谱

# In[13]:


# 绘制预处理前的光谱曲线和平均光谱
plt.figure(figsize=(10,6))
for i in range(X_raw_clean.shape[0]):
    plt.plot(X_raw_clean[i, :], alpha=0.5)
plt.title("Raw Spectral Curves")
plt.xlabel("Wavelength Index")
plt.ylabel("Signal")
plt.show()
plt.figure(figsize=(10,6))
plt.plot(np.mean(X_raw_clean, axis=0), color='magenta', linewidth=2)
plt.title("Raw Average Spectrum")
plt.xlabel("Wavelength Index")
plt.ylabel("Average Signal")
plt.show()


# ## 光谱数据预处理

# ### snv

# In[14]:


# 4. 对光谱数据进行 SNV 预处理
X_snv = SNV_transform(X_raw_clean)
X_proc = X_snv


# ### msc

# In[15]:


X_msc = MSC_transform(X_raw_clean)
X_proc = X_msc


# ### sg

# In[16]:


# 4. 对光谱数据进行 Savitzky–Golay 导数预处理
X_sg = savgol_filter(X_raw_clean, window_length=11, polyorder=2, deriv=1, axis=1)
X_proc = X_sg


# ### 预处理后光谱数据的可视化

# In[17]:


plt.figure(figsize=(10,6))
for i in range(X_proc.shape[0]):
    plt.plot(X_proc[i, :], alpha=0.5)
plt.title("SNV Processed Spectral Curves")
plt.xlabel("Wavelength Index")
plt.ylabel("Signal")
plt.show()
plt.figure(figsize=(10,6))
plt.plot(np.mean(X_proc, axis=0), color='magenta', linewidth=2)
plt.title("SNV Processed Average Spectrum")
plt.xlabel("Wavelength Index")
plt.ylabel("Average Signal")
plt.show()


# ## 划分数据集

# In[18]:


# 5. 利用 SPXY 方法划分样本集
train_idx, test_idx = spxy_partition(X_proc, y_clean, train_fraction=0.8)
print("SPXY划分 - 训练集样本数:", len(train_idx))
print("SPXY划分 - 测试集样本数:", len(test_idx))

# 6. 根据 SPXY 划分结果构建训练集和测试集
X_train = X_proc[train_idx, :]
X_test = X_proc[test_idx, :]
y_train = y_clean[train_idx]
y_test = y_clean[test_idx]



# 打印训练集和测试集的样本数，最大值，最小值，平均值，标准差
print("\n训练集 y 值统计:")
print("训练集样本数:", len(y_train))
print("训练集最大值:", np.max(y_train))
print("训练集最小值:", np.min(y_train))
print("训练集平均值:", np.mean(y_train))
print("训练集标准差:", np.std(y_train))

print("\n测试集 y 值统计:")
print("测试集样本数:", len(y_test))
print("测试集最大值:", np.max(y_test))
print("测试集最小值:", np.min(y_test))
print("测试集平均值:", np.mean(y_test))
print("测试集标准差:", np.std(y_test))


# ## 特征选择

# In[19]:


import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 或其他中文字体，如'Noto Sans CJK SC'


# In[20]:


import importlib

# Assuming CARSS.py is in your current working directory or is on the Python path
import CARSS

# Reload the CARSS module to pick up any changes made
importlib.reload(CARSS)


# In[21]:


import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress the specific warning about the coef_ attribute in PLSRegression
warnings.filterwarnings("ignore", category=FutureWarning, message=".*coef_.*")


# In[22]:


# 4. 使用 CARS 进行特征选择
OptWave = CARSS.CARS_Cloud(X_train, y_train, N=50, f=20, cv=10)
print("选择的波长索引：", OptWave)

# 5. 使用 CARS 选择的特征进行拟合模型
X_train_selected = X_train[:, OptWave]
X_test_selected = X_test[:, OptWave]


# In[23]:


data_clean.iloc[:, 3:].columns[OptWave]


# In[24]:


data_clean.iloc[:, 3:].columns[OptWave].shape


# ## 确定最佳因子数

# In[25]:


# 5. 选择最佳的因子数（n_components）用所有的数据吗
best_n_components, rmsecvs = select_best_n_components(X_train_selected, y_train, max_components=50)

# 打印最佳的因子数
print("最佳的因子数 (n_components):", best_n_components)


# In[26]:


# 绘制因子数与RMSECV的关系图
plt.figure(figsize=(16,8))

# 绘制RMSECV曲线，动态设定x轴的范围
plt.plot(range(1, len(rmsecvs) + 1), rmsecvs, marker='o', linestyle='-', color='b')
plt.title('RMSECV vs Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('RMSECV')
plt.xticks(range(1, len(rmsecvs) + 1))  # 动态设置x轴的标签
plt.grid(True)
plt.show()


# ## 拟合模型

# In[27]:


# 7. 拟合 PLS 回归模型（这里设置 n_components=10，可根据需要调整）
pls = PLSRegression(n_components=best_n_components)
pls.fit(X_train_selected, y_train)


# ## 模型评估

# In[28]:


# 8. 在测试集上预测
y_pred = pls.predict(X_test_selected).ravel()

# 9. 模型评估指标计算
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
n_samples_test = len(y_test)
SEP = np.sqrt(np.sum((y_test - y_pred) ** 2) / (n_samples_test - 1))
std_y_test = np.std(y_test, ddof=1)
RPD = std_y_test / SEP

print("\nPLS模型评估结果（经SG预处理、MCCV剔除异常且SPXY划分样本后）：")
print("R²  : {:.5f}".format(r2))
print("RMSE: {:.5f}".format(rmse))
print("SEP : {:.5f}".format(SEP))
print("RPD : {:.5f}".format(RPD))



# 计算绝对误差（Absolute Error, AE）和相对误差（Relative Error, RE）
absolute_error = np.abs(y_test - y_pred)  # 计算每个样本的绝对误差
relative_error = (absolute_error / np.abs(y_test)) * 100  # 计算每个样本的相对误差，转换为百分比

# 输出前几个样本的绝对误差和相对误差（保留四位小数）
#print("绝对误差 (AE) 的前 10 个样本：", np.round(absolute_error[:10], 4))
#print("相对误差 (RE) 的前 10 个样本：", np.round(relative_error[:10], 4))

# 输出总体的统计数据（保留四位小数）
print("\n绝对误差的统计信息：")
print("最大绝对误差:", np.round(np.max(absolute_error), 4))
print("最小绝对误差:", np.round(np.min(absolute_error), 4))
print("平均绝对误差:", np.round(np.mean(absolute_error), 4))
print("标准差的绝对误差:", np.round(np.std(absolute_error), 4))

print("\n相对误差的统计信息：")
print("最大相对误差:", np.round(np.max(relative_error), 4), "%")
print("最小相对误差:", np.round(np.min(relative_error), 4), "%")
print("平均相对误差:", np.round(np.mean(relative_error), 4), "%")
print("相对误差的标准差:", np.round(np.std(relative_error), 4), "%")


# ## 散点对比图

# In[29]:


# 绘制真实值与预测值对比图
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, )  #label='Predicted vs True'
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', ) # label='Ideal'
plt.xlabel("True Content")
plt.ylabel("Predicted Content")
plt.title("Predicted vs True (Water)")

# 构造显示指标的文本信息，保留五位小数
metrics_text = f"R²: {r2:.2f}\nRMSE: {rmse:.2f}\nSEP: {SEP:.2f}\nRPD: {RPD:.2f}"
# 在图的左上角（坐标轴单位）添加文本框
plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

plt.legend()
plt.show()


# # CF

# ## 剔除异常样本

# In[9]:


# ---------------------------
# 主流程
# ---------------------------
# 1. 读取数据并删除第一列（样本名称）
data_raw = pd.read_csv('NIR_WT_CF_CP.average.csv', header=0)
data = data_raw.iloc[:, 1:]  # 删除第一列
print("数据尺寸（删除样本名称后）：", data.shape)
#print("列名：", data.columns.tolist())
# 数据结构说明：
# 第0列：水分含量（目标变量）
# 第1-2列：其他营养物质（不用于建模）
# 第3列到最后：光谱数据（自变量）


# In[67]:


X_raw = data.iloc[:, 3:].values 
y = data.iloc[:, 1].values


# In[71]:


# 假设 X_raw 是光谱数据，y 是目标变量
X_raw_clean, y_clean, mean_res, var_res = mccv_remove_outliers_and_plot(X_raw, y, n_iterations=100, test_size=0.2, threshold_factor=1)
print("剔除异常样本后数据尺寸:", X_raw_clean.shape)


# ### 预处理前的光谱曲线和平均光谱

# In[72]:


# 绘制预处理前的光谱曲线和平均光谱
plt.figure(figsize=(10,6))
for i in range(X_raw_clean.shape[0]):
    plt.plot(X_raw_clean[i, :], alpha=0.5)
plt.title("Raw Spectral Curves")
plt.xlabel("Wavelength Index")
plt.ylabel("Signal")
plt.show()
plt.figure(figsize=(10,6))
plt.plot(np.mean(X_raw_clean, axis=0), color='magenta', linewidth=2)
plt.title("Raw Average Spectrum")
plt.xlabel("Wavelength Index")
plt.ylabel("Average Signal")
plt.show()


# ## 光谱数据预处理

# ### snv

# In[14]:


# 4. 对光谱数据进行 SNV 预处理
X_snv = SNV_transform(X_raw_clean)
X_proc = X_snv


# ### msc

# In[15]:


X_msc = MSC_transform(X_raw_clean)
X_proc = X_msc


# ### sg

# In[73]:


# 4. 对光谱数据进行 Savitzky–Golay 导数预处理
X_sg = savgol_filter(X_raw_clean, window_length=11, polyorder=2, deriv=1, axis=1)
X_proc = X_sg


# ### 预处理后光谱数据的可视化

# In[74]:


plt.figure(figsize=(10,6))
for i in range(X_proc.shape[0]):
    plt.plot(X_proc[i, :], alpha=0.5)
plt.title("SNV Processed Spectral Curves")
plt.xlabel("Wavelength Index")
plt.ylabel("Signal")
plt.show()
plt.figure(figsize=(10,6))
plt.plot(np.mean(X_proc, axis=0), color='magenta', linewidth=2)
plt.title("SNV Processed Average Spectrum")
plt.xlabel("Wavelength Index")
plt.ylabel("Average Signal")
plt.show()


# ## 划分数据集

# In[75]:


# 5. 利用 SPXY 方法划分样本集
train_idx, test_idx = spxy_partition(X_proc, y_clean, train_fraction=0.8)
print("SPXY划分 - 训练集样本数:", len(train_idx))
print("SPXY划分 - 测试集样本数:", len(test_idx))

# 6. 根据 SPXY 划分结果构建训练集和测试集
X_train = X_proc[train_idx, :]
X_test = X_proc[test_idx, :]
y_train = y_clean[train_idx]
y_test = y_clean[test_idx]



# 打印训练集和测试集的样本数，最大值，最小值，平均值，标准差
print("\n训练集 y 值统计:")
print("训练集样本数:", len(y_train))
print("训练集最大值:", np.max(y_train))
print("训练集最小值:", np.min(y_train))
print("训练集平均值:", np.mean(y_train))
print("训练集标准差:", np.std(y_train))

print("\n测试集 y 值统计:")
print("测试集样本数:", len(y_test))
print("测试集最大值:", np.max(y_test))
print("测试集最小值:", np.min(y_test))
print("测试集平均值:", np.mean(y_test))
print("测试集标准差:", np.std(y_test))


# ## 特征选择

# In[76]:


import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 或其他中文字体，如'Noto Sans CJK SC'


# In[77]:


import importlib

# Assuming CARSS.py is in your current working directory or is on the Python path
import CARSS

# Reload the CARSS module to pick up any changes made
importlib.reload(CARSS)


# In[78]:


import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress the specific warning about the coef_ attribute in PLSRegression
warnings.filterwarnings("ignore", category=FutureWarning, message=".*coef_.*")


# In[79]:


# 4. 使用 CARS 进行特征选择
OptWave = CARSS.CARS_Cloud(X_train, y_train, N=50, f=20, cv=10)
print("选择的波长索引：", OptWave)

# 5. 使用 CARS 选择的特征进行拟合模型
X_train_selected = X_train[:, OptWave]
X_test_selected = X_test[:, OptWave]


# In[80]:


data_clean.iloc[:, 3:].columns[OptWave]


# In[81]:


data_clean.iloc[:, 3:].columns[OptWave].shape


# ## 确定最佳因子数

# In[82]:


# 5. 选择最佳的因子数（n_components）用所有的数据吗
best_n_components, rmsecvs = select_best_n_components(X_train_selected, y_train, max_components=50)

# 打印最佳的因子数
print("最佳的因子数 (n_components):", best_n_components)

# 绘制因子数与RMSECV的关系图
plt.figure(figsize=(16,8))

# 绘制RMSECV曲线，动态设定x轴的范围
plt.plot(range(1, len(rmsecvs) + 1), rmsecvs, marker='o', linestyle='-', color='b')
plt.title('RMSECV vs Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('RMSECV')
plt.xticks(range(1, len(rmsecvs) + 1))  # 动态设置x轴的标签
plt.grid(True)
plt.show()


# ## 拟合模型

# In[83]:


# 7. 拟合 PLS 回归模型（这里设置 n_components=10，可根据需要调整）
pls = PLSRegression(n_components=best_n_components)
pls.fit(X_train_selected, y_train)


# ## 模型评估

# In[84]:


# 8. 在测试集上预测
y_pred = pls.predict(X_test_selected).ravel()

# 9. 模型评估指标计算
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
n_samples_test = len(y_test)
SEP = np.sqrt(np.sum((y_test - y_pred) ** 2) / (n_samples_test - 1))
std_y_test = np.std(y_test, ddof=1)
RPD = std_y_test / SEP

print("\nPLS模型评估结果（经SG预处理、MCCV剔除异常且SPXY划分样本后）：")
print("R²  : {:.5f}".format(r2))
print("RMSE: {:.5f}".format(rmse))
print("SEP : {:.5f}".format(SEP))
print("RPD : {:.5f}".format(RPD))



# 计算绝对误差（Absolute Error, AE）和相对误差（Relative Error, RE）
absolute_error = np.abs(y_test - y_pred)  # 计算每个样本的绝对误差
relative_error = (absolute_error / np.abs(y_test)) * 100  # 计算每个样本的相对误差，转换为百分比

# 输出前几个样本的绝对误差和相对误差（保留四位小数）
#print("绝对误差 (AE) 的前 10 个样本：", np.round(absolute_error[:10], 4))
#print("相对误差 (RE) 的前 10 个样本：", np.round(relative_error[:10], 4))

# 输出总体的统计数据（保留四位小数）
print("\n绝对误差的统计信息：")
print("最大绝对误差:", np.round(np.max(absolute_error), 4))
print("最小绝对误差:", np.round(np.min(absolute_error), 4))
print("平均绝对误差:", np.round(np.mean(absolute_error), 4))
print("标准差的绝对误差:", np.round(np.std(absolute_error), 4))

print("\n相对误差的统计信息：")
print("最大相对误差:", np.round(np.max(relative_error), 4), "%")
print("最小相对误差:", np.round(np.min(relative_error), 4), "%")
print("平均相对误差:", np.round(np.mean(relative_error), 4), "%")
print("相对误差的标准差:", np.round(np.std(relative_error), 4), "%")


# ## 散点对比图

# In[85]:


# 绘制真实值与预测值对比图
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, )  #label='Predicted vs True'
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', ) # label='Ideal'
plt.xlabel("True Content")
plt.ylabel("Predicted Content")
plt.title("Predicted vs True (Crude Fat)")

# 构造显示指标的文本信息，保留五位小数
metrics_text = f"R²: {r2:.2f}\nRMSE: {rmse:.2f}\nSEP: {SEP:.2f}\nRPD: {RPD:.2f}"
# 在图的左上角（坐标轴单位）添加文本框
plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

plt.legend()
plt.show()


# # CP

# ## 剔除异常值

# In[51]:


X_raw = data.iloc[:, 3:].values 
y = data.iloc[:, 2].values


# In[52]:


# 假设 X_raw 是光谱数据，y 是目标变量
X_raw_clean, y_clean, mean_res, var_res = mccv_remove_outliers_and_plot(X_raw, y, n_iterations=100, test_size=0.3, threshold_factor=2)
print("剔除异常样本后数据尺寸:", X_raw_clean.shape)


# ### 预处理前的光谱曲线和平均光谱

# In[53]:


# 绘制预处理前的光谱曲线和平均光谱
plt.figure(figsize=(10,6))
for i in range(X_raw_clean.shape[0]):
    plt.plot(X_raw_clean[i, :], alpha=0.5)
plt.title("Raw Spectral Curves")
plt.xlabel("Wavelength Index")
plt.ylabel("Signal")
plt.show()
plt.figure(figsize=(10,6))
plt.plot(np.mean(X_raw_clean, axis=0), color='magenta', linewidth=2)
plt.title("Raw Average Spectrum")
plt.xlabel("Wavelength Index")
plt.ylabel("Average Signal")
plt.show()


# ## 光谱数据预处理

# ### snv

# In[14]:


# 4. 对光谱数据进行 SNV 预处理
X_snv = SNV_transform(X_raw_clean)
X_proc = X_snv


# ### msc

# In[15]:


X_msc = MSC_transform(X_raw_clean)
X_proc = X_msc


# ### sg

# In[54]:


# 4. 对光谱数据进行 Savitzky–Golay 导数预处理
X_sg = savgol_filter(X_raw_clean, window_length=11, polyorder=2, deriv=1, axis=1)
X_proc = X_sg


# ### 预处理后光谱数据的可视化

# In[55]:


plt.figure(figsize=(10,6))
for i in range(X_proc.shape[0]):
    plt.plot(X_proc[i, :], alpha=0.5)
plt.title("SNV Processed Spectral Curves")
plt.xlabel("Wavelength Index")
plt.ylabel("Signal")
plt.show()
plt.figure(figsize=(10,6))
plt.plot(np.mean(X_proc, axis=0), color='magenta', linewidth=2)
plt.title("SNV Processed Average Spectrum")
plt.xlabel("Wavelength Index")
plt.ylabel("Average Signal")
plt.show()


# ## 划分数据集

# In[56]:


# 5. 利用 SPXY 方法划分样本集
train_idx, test_idx = spxy_partition(X_proc, y_clean, train_fraction=0.8)
print("SPXY划分 - 训练集样本数:", len(train_idx))
print("SPXY划分 - 测试集样本数:", len(test_idx))

# 6. 根据 SPXY 划分结果构建训练集和测试集
X_train = X_proc[train_idx, :]
X_test = X_proc[test_idx, :]
y_train = y_clean[train_idx]
y_test = y_clean[test_idx]



# 打印训练集和测试集的样本数，最大值，最小值，平均值，标准差
print("\n训练集 y 值统计:")
print("训练集样本数:", len(y_train))
print("训练集最大值:", np.max(y_train))
print("训练集最小值:", np.min(y_train))
print("训练集平均值:", np.mean(y_train))
print("训练集标准差:", np.std(y_train))

print("\n测试集 y 值统计:")
print("测试集样本数:", len(y_test))
print("测试集最大值:", np.max(y_test))
print("测试集最小值:", np.min(y_test))
print("测试集平均值:", np.mean(y_test))
print("测试集标准差:", np.std(y_test))


# ## 特征选择

# In[57]:


import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 或其他中文字体，如'Noto Sans CJK SC'


# In[58]:


# 4. 使用 CARS 进行特征选择
OptWave = CARSS.CARS_Cloud(X_train, y_train, N=50, f=20, cv=10)
print("选择的波长索引：", OptWave)

# 5. 使用 CARS 选择的特征进行拟合模型
X_train_selected = X_train[:, OptWave]
X_test_selected = X_test[:, OptWave]


# In[59]:


data_clean.iloc[:, 3:].columns[OptWave]


# In[60]:


data_clean.iloc[:, 3:].columns[OptWave].shape


# ## 确定最佳因子数

# In[62]:


# 5. 选择最佳的因子数（n_components）用所有的数据吗
best_n_components, rmsecvs = select_best_n_components(X_train_selected, y_train, max_components=50)

# 打印最佳的因子数
print("最佳的因子数 (n_components):", best_n_components)

# 绘制因子数与RMSECV的关系图
plt.figure(figsize=(16,8))

# 绘制RMSECV曲线，动态设定x轴的范围
plt.plot(range(1, len(rmsecvs) + 1), rmsecvs, marker='o', linestyle='-', color='b')
plt.title('RMSECV vs Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('RMSECV')
plt.xticks(range(1, len(rmsecvs) + 1))  # 动态设置x轴的标签
plt.grid(True)
plt.show()


# ## 拟合模型

# In[63]:


# 7. 拟合 PLS 回归模型（这里设置 n_components=10，可根据需要调整）
pls = PLSRegression(n_components=best_n_components)
pls.fit(X_train_selected, y_train)


# ## 模型评估

# In[64]:


# 8. 在测试集上预测
y_pred = pls.predict(X_test_selected).ravel()

# 9. 模型评估指标计算
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
n_samples_test = len(y_test)
SEP = np.sqrt(np.sum((y_test - y_pred) ** 2) / (n_samples_test - 1))
std_y_test = np.std(y_test, ddof=1)
RPD = std_y_test / SEP

print("\nPLS模型评估结果（经SG预处理、MCCV剔除异常且SPXY划分样本后）：")
print("R²  : {:.5f}".format(r2))
print("RMSE: {:.5f}".format(rmse))
print("SEP : {:.5f}".format(SEP))
print("RPD : {:.5f}".format(RPD))



# 计算绝对误差（Absolute Error, AE）和相对误差（Relative Error, RE）
absolute_error = np.abs(y_test - y_pred)  # 计算每个样本的绝对误差
relative_error = (absolute_error / np.abs(y_test)) * 100  # 计算每个样本的相对误差，转换为百分比

# 输出前几个样本的绝对误差和相对误差（保留四位小数）
#print("绝对误差 (AE) 的前 10 个样本：", np.round(absolute_error[:10], 4))
#print("相对误差 (RE) 的前 10 个样本：", np.round(relative_error[:10], 4))

# 输出总体的统计数据（保留四位小数）
print("\n绝对误差的统计信息：")
print("最大绝对误差:", np.round(np.max(absolute_error), 4))
print("最小绝对误差:", np.round(np.min(absolute_error), 4))
print("平均绝对误差:", np.round(np.mean(absolute_error), 4))
print("标准差的绝对误差:", np.round(np.std(absolute_error), 4))

print("\n相对误差的统计信息：")
print("最大相对误差:", np.round(np.max(relative_error), 4), "%")
print("最小相对误差:", np.round(np.min(relative_error), 4), "%")
print("平均相对误差:", np.round(np.mean(relative_error), 4), "%")
print("相对误差的标准差:", np.round(np.std(relative_error), 4), "%")


# ## 散点对比图

# In[66]:


# 绘制真实值与预测值对比图
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, )  #label='Predicted vs True'
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', ) # label='Ideal'
plt.xlabel("True Content")
plt.ylabel("Predicted Content")
plt.title("Predicted vs True (Crude protein)")

# 构造显示指标的文本信息，保留五位小数
metrics_text = f"R²: {r2:.2f}\nRMSE: {rmse:.2f}\nSEP: {SEP:.2f}\nRPD: {RPD:.2f}"
# 在图的左上角（坐标轴单位）添加文本框
plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top',
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

plt.legend()
plt.show()

