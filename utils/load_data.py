import h5py
import xlrd
import math
import numpy as np
from scipy import sparse
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os

"""
load parameters
"""
def load_para(N, M, baseMVA, omega_s, net, adj_mode):
    """
    从.xlsx文件中导出参数及初始条件
    """
    # parameter = xlrd.open_workbook('/public/home/spy2018/swing/parameter/parameter%s.xlsx' %(N))
    parameter = xlrd.open_workbook('/home/duguyuan/Documents/Swing_in_Grid/IEEE/case%s/parameter/parameter.xlsx' %(N))
    # 功率矩阵
    P_sheet1 = parameter.sheet_by_index(0)
    nrows = P_sheet1.nrows
    ncols = P_sheet1.ncols
    P = np.zeros((N))
    for i in range(nrows):
        for j in range(ncols):
            P[i] = P_sheet1.cell_value(i, j)
    P = P * baseMVA
    P = [i - np.sum(P)/N for i in P]  # 功率补偿
    P = np.array([i/(M*omega_s) for i in P])
    # 导纳矩阵
    Y_sheet1 = parameter.sheet_by_index(1)
    nrows = Y_sheet1.nrows
    ncols = Y_sheet1.ncols
    Y = np.zeros((N, N))
    YY = np.zeros((N, N))
    for i in range(nrows):
        for j in range(ncols):
            Y[i, j] = Y_sheet1.cell_value(i, j)
            if Y[i, j] != 0:
                YY[i, j] = 1
    Y = np.array([i*baseMVA/(M*omega_s) for i in Y])
    # 参数合并
    PY = np.vstack((P, Y))
    PY = PY / 16

    Y /= 16
    P /= 16
    if net == 'GCN' or 'RGCN' or 'RGCN-TCN':
        if adj_mode == 2:
            Y = Y + np.diag(abs(P))
        else:
            pass
        # A = sparse.csr_matrix(Y)
    elif net == 'GAT':
        A = sparse.csr_matrix(YY)
    print('原始数据导入完毕')
    return Y, PY

"""
split data to train/val/test sets
"""
def classify_random_5(N, x_theta, x_omega, y_data, test_size, chosedlength, CHANNEL):
    """
    randomly classify data to train and test group
    (2, chosedlength, N)
    """

    x_data = np.dstack((x_omega,x_theta))
    x_train, X_test, y_train, Y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=0) # 随机分组
    X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=0) # 随机分组
    del x_data, y_data, x_train, y_train
    X_train = np.reshape(X_train, (X_train.shape[0], N, chosedlength, 2))
    X_val = np.reshape(X_val, (X_val.shape[0], N, chosedlength, 2))
    X_test = np.reshape(X_test, (X_test.shape[0], N, chosedlength, 2))

    return X_train[:,:,:,0], X_train[:,:,:,1], X_val[:,:,:,0], X_val[:,:,:,1], X_test[:,:,:,0], X_test[:,:,:,1], Y_train, Y_val, Y_test

"""
normalization
"""
def normalization(N, x_theta, x_omega):

    scaler = StandardScaler()
    x_omega_norm = scaler.fit_transform(x_omega.astype(np.float32))
    x_theta_norm = scaler.fit_transform(x_theta.astype(np.float32))

    return x_omega_norm, x_theta_norm

def standardization(N, chosedlength, x_theta, x_omega, relative, mode):
    
    if relative:

        if mode == 0:

            omega_range = np.max(x_omega, axis=1) - np.min(x_omega, axis=1)
            theta_range = np.max(x_theta, axis=1) - np.min(x_theta, axis=1)

            return (x_omega - np.min(x_omega, axis=1).reshape(-1, 1)) / omega_range.reshape(-1, 1), (x_theta - np.min(x_theta, axis=1).reshape(-1, 1)) / theta_range.reshape(-1, 1)
        
        elif mode == 1:

            return x_omega / np.max(abs(x_omega), axis=1).reshape(-1, 1), x_theta / np.max(abs(x_theta), axis=1).reshape(-1, 1)

    else:
        x_omega_std = np.zeros((x_theta.shape[0], N*chosedlength))
        x_theta_std = np.zeros((x_theta.shape[0], N*chosedlength))
        if mode == 0:
            for i in range(N):
                omega_range = np.max(x_omega[:,i*chosedlength:(i+1)*chosedlength], axis=1) - np.min(x_omega[:,i*chosedlength:(i+1)*chosedlength], axis=1)
                x_omega_std[:,i*chosedlength:(i+1)*chosedlength] = (x_omega[:,i*chosedlength:(i+1)*chosedlength] - np.min(x_omega[:,i*chosedlength:(i+1)*chosedlength], axis=0)) / omega_range

                theta_range = np.max(x_theta[:,i*chosedlength:(i+1)*chosedlength], axis=1) - np.min(x_theta[:,i*chosedlength:(i+1)*chosedlength], axis=1)
                x_theta_std[:,i*chosedlength:(i+1)*chosedlength] = (x_theta[:,i*chosedlength:(i+1)*chosedlength] - np.min(x_theta[:,i*chosedlength:(i+1)*chosedlength], axis=0)) / theta_range

        elif mode == 1:

            for i in range(N):
                omega_max = np.max(abs(x_omega[:,i*chosedlength:(i+1)*chosedlength]), axis=1)
                x_omega_std[:,i*chosedlength:(i+1)*chosedlength] = x_omega[:,i*chosedlength:(i+1)*chosedlength] / omega_max.reshape(-1, 1)

                theta_max = np.max(abs(x_theta[:,i*chosedlength:(i+1)*chosedlength]), axis=1)
                x_theta_std[:,i*chosedlength:(i+1)*chosedlength] = x_theta[:,i*chosedlength:(i+1)*chosedlength] / theta_max.reshape(-1, 1)
        
        return x_omega_std, x_theta_std

"""
moving average
"""
def smooth(a, WSZ):
    out0 = np.convolve(a, np.ones(WSZ, dtype=int),'valid')/WSZ
    r = np.arange(1, WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((start, out0, stop))

"""
load data
"""
def load_data_one_IEEE_l(N, omega, timelength, chosedlength, TEST_SIZE, CHANNEL, interval, relative, normalize, standard, mode, move, WSZ):
    """
    no oversampling
    """
    start = time.perf_counter()
    length = 1000  # load data 441*39
    mode = 12
    X_one_theta_2 = np.zeros((length, chosedlength*N))
    X_one_omega_2 = np.zeros((length, chosedlength*N))

    f = h5py.File('../data/swing_equation/one/1.h5', 'r')
    if chosedlength != timelength:
        for i in range(N):
            X_one_theta_2[:, i*chosedlength:(i+1)*chosedlength] = f['data_theta'][()][:, i*timelength:i*timelength+chosedlength]
            X_one_omega_2[:, i*chosedlength:(i+1)*chosedlength] = f['data_omega'][()][:, i*timelength:i*timelength+chosedlength]
    else:
        X_one_theta_2 = f['data_theta'][()]
        X_one_omega_2 = f['data_omega'][()]
        
    Y_one_2 = f['Y'][()]
    f.close()
    del f

    for i in range(N):
        if i == 0:
            pass
        else:
            X_theta = np.zeros((length, chosedlength*N))
            X_omega = np.zeros((length, chosedlength*N))
            f = h5py.File('../data/swing_equation/one/%s.h5' % (i+1), 'r')
            if chosedlength != timelength:
                for ii in range(N):
                    X_theta[:, ii*chosedlength:(ii+1)*chosedlength] = f['data_theta'][()][:, ii*timelength:ii*timelength+chosedlength]
                    X_omega[:, ii*chosedlength:(ii+1)*chosedlength] = f['data_omega'][()][:, ii*timelength:ii*timelength+chosedlength]
            else:
                X_theta = f['data_theta'][()]
                X_omega = f['data_omega'][()]
            Y = f['Y'][()]
            f.close()
            del f
            X_one_theta_2 = np.vstack((X_one_theta_2, X_theta))
            X_one_omega_2 = np.vstack((X_one_omega_2, X_omega))
            Y_one_2 = np.hstack((Y_one_2, Y))
            del X_theta, X_omega, Y

    end = time.perf_counter()
    print('所用时间为%ss' % (str(end - start)))

    X_one_theta_2 = np.float32(X_one_theta_2)
    X_one_omega_2 = np.float32(X_one_omega_2)

    if normalize:
        X_one_omega_norm, X_one_theta_norm = normalization(
            x_theta=X_one_theta_2, x_omega=X_one_omega_2
        )
        del X_one_theta_2, X_one_omega_2
        X_train, X_train_theta, X_val, X_val_theta, X_test, X_test_theta, Y_train, Y_val, Y_test = classify_random_5(
            x_theta=X_one_theta_norm,
            x_omega=X_one_omega_norm,
            y_data=Y_one_2,
            test_size=TEST_SIZE,
            # timelength=timelength,
            chosedlength=chosedlength,
            CHANNEL=CHANNEL,
            N=N
        )
    
    elif standard:
        X_one_omega_std, X_one_theta_std = standardization(
            N=N, chosedlength=chosedlength, x_theta=X_one_theta_2, x_omega=X_one_omega_2, relative=relative, mode=mode
        )
        del X_one_theta_2, X_one_omega_2
        X_train, X_train_theta, X_val, X_val_theta, X_test, X_test_theta, Y_train, Y_val, Y_test = classify_random_5(
            x_theta=X_one_theta_std,
            x_omega=X_one_omega_std,
            y_data=Y_one_2,
            test_size=TEST_SIZE,
            # timelength=timelength,
            chosedlength=chosedlength,
            CHANNEL=CHANNEL,
            N=N
        )
    
    else:
        X_train, X_train_theta, X_val, X_val_theta, X_test, X_test_theta, Y_train, Y_val, Y_test = classify_random_5(
            x_theta=X_one_theta_2,
            x_omega=X_one_omega_2,
            y_data=Y_one_2,
            test_size=TEST_SIZE,
            # timelength=timelength,
            chosedlength=chosedlength,
            CHANNEL=CHANNEL,
            N=N
        )
        del X_one_theta_2, X_one_omega_2
    
    if move:
        for i in range(N):
            for j in range(X_train.shape[0]):
                X_train[j, :, i] = smooth(a=X_train[j, :, i], WSZ=WSZ)
                # X_train[j, 1, :, i] = smooth(a=X_train[j, 1, :, i], WSZ=WSZ)
            for j in range(X_val.shape[0]):
                X_val[j, :, i] = smooth(a=X_val[j, :, i], WSZ=WSZ)
                # X_val[j, 1, :, i] = smooth(a=X_val[j, 1, :, i], WSZ=WSZ)
            for j in range(X_test.shape[0]):
                X_test[j, :, i] = smooth(a=X_test[j, :, i], WSZ=WSZ)
                # X_test[j, 1, :, i] = smooth(a=X_test[j, 1, :, i], WSZ=WSZ)
        del i, j

    a = len(Y_train)+len(Y_val)+len(Y_test)
    print('数据总数为:%s' %(a))
    b = int(np.sum(Y_train)+np.sum(Y_val)+np.sum(Y_test))

    print('同步状态数量:非同步状态=%s:%s' %(a-b, b))

    print('训练集：', X_train.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_train)-int(np.sum(Y_train)), int(np.sum(Y_train))))
    print('验证集：', X_val.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_val)-int(np.sum(Y_val)), int(np.sum(Y_val))))
    print('测试集：', X_test.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_test)-int(np.sum(Y_test)), int(np.sum(Y_test))))

    print('数据分组完毕，开始训练')

    return X_train, X_train_theta, X_val, X_val_theta, X_test, X_test_theta, Y_train, Y_val, Y_test, a, b

if __name__ == '__main__':

