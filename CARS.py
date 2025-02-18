import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import copy
from matplotlib import rcParams

def PC_Cross_Validation(X, y, pc, cv):
    '''
        x :光谱矩阵 nxm
        y :浓度阵 （化学值）
        pc:最大主成分数
        cv:交叉验证数量
    return :
        RMSECV:各主成分数对应的RMSECV
        PRESS :各主成分数对应的PRESS
        rindex:最佳主成分数
    '''
    kf = KFold(n_splits=cv)
    RMSECV = []
    for i in range(pc):
        RMSE = []
        for train_index, test_index in kf.split(X):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            pls = PLSRegression(n_components=i + 1)
            pls.fit(x_train, y_train)
            y_predict = pls.predict(x_test)
            RMSE.append(np.sqrt(mean_squared_error(y_test, y_predict)))
        RMSE_mean = np.mean(RMSE)
        RMSECV.append(RMSE_mean)
    rindex = np.argmin(RMSECV)
    return RMSECV, rindex

def Cross_Validation(X, y, pc, cv):
    '''
     x :光谱矩阵 nxm
     y :浓度阵 （化学值）
     pc:最大主成分数
     cv:交叉验证数量
     return :
            RMSECV:各主成分数对应的RMSECV
    '''
    kf = KFold(n_splits=cv)
    RMSE = []
    for train_index, test_index in kf.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        pls = PLSRegression(n_components=pc)
        pls.fit(x_train, y_train)
        y_predict = pls.predict(x_test)
        RMSE.append(np.sqrt(mean_squared_error(y_test, y_predict)))
    RMSE_mean = np.mean(RMSE)
    return RMSE_mean

def CARS_Cloud(X, y, N=50, f=20, cv=10):
    p = 0.8
    m, n = X.shape
    u = np.power((n/2), (1/(N-1)))
    k = (1/(N-1)) * np.log(n/2)
    cal_num = np.round(m * p)
    # val_num = m - cal_num
    b2 = np.arange(n)
    x = copy.deepcopy(X)
    D = np.vstack((np.array(b2).reshape(1, -1), X))
    WaveData = []
    # Coeff = []
    WaveNum =[]
    RMSECV = []
    r = []
    coeff_path = []  # Store the coefficient path
    max_coeff_len = 0  # To store the max length of coefficients for padding
    
    for i in range(1, N+1):
        r.append(u*np.exp(-1*k*i))
        wave_num = int(np.round(r[i-1]*n))
        WaveNum = np.hstack((WaveNum, wave_num))
        cal_index = np.random.choice    \
            (np.arange(m), size=int(cal_num), replace=False)
        wave_index = b2[:wave_num].reshape(1, -1)[0]
        xcal = x[np.ix_(list(cal_index), list(wave_index))]
        #xcal = xcal[:,wave_index].reshape(-1,wave_num)
        ycal = y[cal_index]
        x = x[:, wave_index]
        D = D[:, wave_index]
        d = D[0, :].reshape(1,-1)
        wnum = n - wave_num
        if wnum > 0:
            d = np.hstack((d, np.full((1, wnum), -1)))
        if len(WaveData) == 0:
            WaveData = d
        else:
            WaveData  = np.vstack((WaveData, d.reshape(1, -1)))

        if wave_num < f:
            f = wave_num

        pls = PLSRegression(n_components=f)
        pls.fit(xcal, ycal)
        beta = pls.coef_
        b = np.abs(beta)
        b2 = np.argsort(-b, axis=0)
        coef = copy.deepcopy(beta)
        coeff = coef[b2, :].reshape(len(b2), -1)
        
        # Store the coefficient path
        coeff_path.append(coeff.flatten())
        max_coeff_len = max(max_coeff_len, len(coeff.flatten()))  # Update max coefficient length
        
        rmsecv, rindex = PC_Cross_Validation(xcal, ycal, f, cv)
        RMSECV.append(Cross_Validation(xcal, ycal, rindex+1, cv))

    # CoeffData = Coeff.T

    WAVE = []

    for i in range(WaveData.shape[0]):
        wd = WaveData[i, :]
        WD = np.ones((len(wd)))
        for j in range(len(wd)):
            ind = np.where(wd == j)
            if len(ind[0]) == 0:
                WD[j] = 0
            else:
                WD[j] = wd[ind[0]]
        if len(WAVE) == 0:
            WAVE = copy.deepcopy(WD)
        else:
            WAVE = np.vstack((WAVE, WD.reshape(1, -1)))

    MinIndex = np.argmin(RMSECV)
    Optimal = WAVE[MinIndex, :]
    boindex = np.where(Optimal != 0)
    OptWave = boindex[0]

    # Pad coefficient paths to have the same length for plotting
    coeff_path_padded = np.array([np.pad(coeff, (0, max_coeff_len - len(coeff))) for coeff in coeff_path])

    fig = plt.figure(figsize=(10, 8))
    # Set a font that is available on your system (for example, DejaVu Sans)
    rcParams['font.family'] = 'DejaVu Sans'
    rcParams['axes.unicode_minus'] = False  # This will display minus signs correctly
    fonts = 12
    
    # Plot (a)
    plt.subplot(311)
    plt.xlabel('Monte Carlo iterations', fontsize=fonts)
    plt.ylabel('Number of selected wavelengths', fontsize=fonts)
    plt.title('Optimal number of iterations:' + str(MinIndex), fontsize=fonts)
    plt.plot(np.arange(N), WaveNum)
    
    # Plot (b)
    plt.subplot(312)
    plt.xlabel('Monte Carlo iterations', fontsize=fonts)
    plt.ylabel('RMSECV', fontsize=fonts)
    plt.plot(np.arange(N), RMSECV)
    
    # Plot (c) - Regression coefficients path
    plt.subplot(313)
    plt.xlabel('Monte Carlo iterations', fontsize=fonts)
    plt.ylabel('Regression coefficients path', fontsize=fonts)
    
    # Plot the regression coefficients path (without legend)
    for i in range(coeff_path_padded.shape[1]):
        plt.plot(np.arange(N), coeff_path_padded[:, i])
    
    plt.tight_layout()
    plt.show()

    return OptWave