import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import CARS  # Assuming you have this module

# ---------------------------
# 数据处理与异常值剔除函数
# ---------------------------
def mccv_remove_outliers(X_raw, y, n_iterations=100, test_size=0.3, threshold_factor=2, window_length=11, polyorder=2, deriv=1):
    sample_indices = np.arange(len(y))  # 样本索引
    
    residuals_dict = {idx: [] for idx in sample_indices}
    errors = {idx: [] for idx in sample_indices}

    for i in range(n_iterations):
        X_train_raw, X_test_raw, y_train, y_test, idx_train, idx_test = train_test_split(
            X_raw, y, sample_indices, test_size=test_size, random_state=i
        )
        
        X_train_proc = savgol_filter(X_train_raw, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)
        X_test_proc = savgol_filter(X_test_raw, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)
        
        pls = PLSRegression(n_components=20)
        pls.fit(X_train_proc, y_train)
        
        y_pred = pls.predict(X_test_proc).ravel()
        
        for idx, real_val, pred_val in zip(idx_test, y_test, y_pred):
            residual = abs(real_val - pred_val)
            residuals_dict[idx].append(residual)
            errors[idx].append(abs(real_val - pred_val))
    
    mean_residuals, var_residuals = [], []
    for idx in sample_indices:
        res_list = residuals_dict[idx]
        mean_residuals.append(np.mean(res_list))
        var_residuals.append(np.var(res_list, ddof=1))
    
    avg_errors_series = pd.Series({idx: np.mean(err_list) for idx, err_list in errors.items()})
    thresh = avg_errors_series.mean() + threshold_factor * avg_errors_series.std()
    outlier_indices = avg_errors_series[avg_errors_series > thresh].index
    X_raw_clean = np.delete(X_raw, outlier_indices, axis=0)
    y_clean = np.delete(y, outlier_indices, axis=0)
    
    return X_raw_clean, y_clean, mean_residuals, var_residuals

# ---------------------------
# 光谱数据预处理
# ---------------------------
def SNV_transform(X):
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, ddof=1, keepdims=True)
    return (X - mean) / std

def MSC_transform(X):
    mean_spectrum = np.mean(X, axis=0)
    X_msc = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        b, a = np.polyfit(mean_spectrum, X[i, :], 1)
        X_msc[i, :] = (X[i, :] - a) / b
    return X_msc

def preprocess(X_raw, y, preprocess_method="SNV", **kwargs):
    """
    根据用户选择的预处理方法对光谱数据进行预处理。
    
    参数:
      X_raw: 原始光谱数据
      y: 目标变量
      preprocess_method: 选择预处理方法，可选 "SNV", "MSC", "SG"
    
    返回:
      预处理后的光谱数据
    """
    if preprocess_method == "SNV":
        X_proc = SNV_transform(X_raw)
    elif preprocess_method == "MSC":
        X_proc = MSC_transform(X_raw)
    elif preprocess_method == "SG":
        window_length = kwargs.get('window_length', 11)
        polyorder = kwargs.get('polyorder', 2)
        deriv = kwargs.get('deriv', 1)
        X_proc = savgol_filter(X_raw, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)
    else:
        raise ValueError("Invalid preprocess_method. Choose from 'SNV', 'MSC', or 'SG'.")
    
    return X_proc

# ---------------------------
# SPXY 样本划分
# ---------------------------
def spxy_partition(X, y, train_fraction=0.8):
    n_samples = X.shape[0]
    scaler_X = StandardScaler()
    X_norm = scaler_X.fit_transform(X)
    scaler_y = StandardScaler()
    y_norm = scaler_y.fit_transform(y.reshape(-1, 1))
    
    Z = np.hstack((X_norm, y_norm))
    dist_matrix = squareform(pdist(Z, metric='euclidean'))
    
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

# ---------------------------
# 特征选择：CARS
# ---------------------------
def select_best_n_components(X_train, y_train, max_components=20):
    rmsecvs = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for n in range(1, max_components+1):
        pls = PLSRegression(n_components=n)
        rmse = np.sqrt(-cross_val_score(pls, X_train, y_train, cv=kf, scoring='neg_mean_squared_error'))
        rmsecvs.append(rmse.mean())
    
    best_n_components = np.argmin(rmsecvs) + 1
    return best_n_components, rmsecvs

# ---------------------------
# 运行函数：执行整个流程
# ---------------------------
def run(X_raw, y, preprocess_method="SNV", n_iterations=100, test_size=0.3, threshold_factor=2, window_length=11, polyorder=2, deriv=1):
    # 数据处理与异常值剔除
    X_raw_clean, y_clean, mean_res, var_res = mccv_remove_outliers(X_raw, y, n_iterations, test_size, threshold_factor, window_length, polyorder, deriv)
    
    # 预处理
    X_proc = preprocess(X_raw_clean, y_clean, preprocess_method, window_length=window_length, polyorder=polyorder, deriv=deriv)

    # SPXY样本划分
    train_idx, test_idx = spxy_partition(X_proc, y_clean)
    X_train, X_test = X_proc[train_idx, :], X_proc[test_idx, :]
    y_train, y_test = y_clean[train_idx], y_clean[test_idx]

    # 使用CARS进行特征选择
    OptWave = CARS.CARS_Cloud(X_train, y_train, N=50, f=20, cv=10)
    X_train_selected = X_train[:, OptWave]
    X_test_selected = X_test[:, OptWave]

    # 选择最佳因子数
    best_n_components, rmsecvs = select_best_n_components(X_train_selected, y_train)
    
    # 拟合PLS模型
    pls = PLSRegression(n_components=best_n_components)
    pls.fit(X_train_selected, y_train)
    y_pred = pls.predict(X_test_selected).ravel()

    # 模型评估
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"R²: {r2:.2f}, RMSE: {rmse:.2f}")
    return pls, r2, rmse
