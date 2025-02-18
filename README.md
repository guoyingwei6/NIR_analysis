# NIR_analysis
Classification and regression using NIR data


# NIR_regression 使用说明

## 概述
这个模块实现了从光谱数据预处理到模型评估的完整流程。它包含以下步骤：
1. 数据加载与异常样本剔除 (MCCV)
2. 光谱数据预处理（支持 SNV、MSC、SG）
3. 数据集划分（使用 SPXY 方法）
4. 特征选择（使用 CARS 方法）
5. 使用 PLS 模型进行训练和评估

## 函数说明

### `mccv_remove_outliers(X_raw, y, n_iterations=100, test_size=0.3, threshold_factor=2, window_length=11, polyorder=2, deriv=1)`
剔除异常样本。

**参数**:
- `X_raw`: 光谱数据（自变量）
- `y`: 目标变量（如水分含量）
- `n_iterations`: MCCV 迭代次数（默认为 100）
- `test_size`: 每次拆分数据时测试集的比例（默认为 0.3）
- `threshold_factor`: 阈值因子，决定异常样本的判断标准（默认为 2）

**返回**:
- `X_raw_clean`: 剔除异常样本后的光谱数据
- `y_clean`: 剔除异常样本后的目标变量
- `mean_residuals`: 每个样本的残差均值
- `var_residuals`: 每个样本的残差方差

### `preprocess(X_raw, y, preprocess_method="SNV", **kwargs)`
根据用户选择的预处理方法对光谱数据进行预处理。
  
  **参数**:
  - `X_raw`: 原始光谱数据
  - `y`: 目标变量
  - `preprocess_method`: 选择预处理方法，支持 "SNV"、"MSC"、"SG"
    
    **返回**:
    - 预处理后的光谱数据

### `spxy_partition(X, y, train_fraction=0.8)`
使用 SPXY 方法划分样本集，基于光谱数据和目标变量的联合标准化欧氏距离。

### `select_best_n_components(X_train, y_train, max_components=20)`
通过交叉验证选择最佳的因子数（n_components）。

### `run(X_raw, y, preprocess_method="SNV", n_iterations=100, test_size=0.3, threshold_factor=2, window_length=11, polyorder=2, deriv=1)`
执行完整的流程：从异常值剔除、光谱数据预处理、特征选择到模型训练与评估。

**参数**:
- `X_raw`: 输入的光谱数据（自变量）
- `y`: 目标变量
- `preprocess_method`: 预处理方法，支持 "SNV"、"MSC"、"SG"
- `n_iterations`, `test_size`, `threshold_factor`, `window_length`, `polyorder`, `deriv`: 对应的参数，控制数据处理与预处理步骤。

**返回**:
- 拟合的 PLS 模型
- 模型评估指标：R² 和 RMSE

## 示例

```python
# 运行完整流程，执行数据加载、异常样本剔除、光谱预处理、特征选择和模型训练
pls_model, r2, rmse = run(X_raw, y, preprocess_method="SNV")

# 输出评估指标
print(f"R²: {r2:.2f}")
print(f"RMSE: {rmse:.2f}")
