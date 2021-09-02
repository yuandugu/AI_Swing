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

def load_para_h5(N, M, baseMVA, omega_s, net, adj_mode):
    '''
    description: load new enngland data

    input      :
    output     : PY, power and admittance matrix

                 initial, initial angle state

                 H, inertia constant
    '''
    # f = h5py.File('/public/home/spy2018/swing/parameter/parameter%s_1.h5' %(N),'r')
    # f = h5py.File('/home/duguyuan/Documents/Swing_in_Grid/IEEE/case%s/parameter/parameter_1.h5' %(N),'r')
    f = h5py.File(r'F:\Swing\parameter\case39\parameter_1.h5','r')
    Y = f['Y'][()]
    # H = f['H'][()]
    # H = np.array(H)
    P_sol = f['P_sol'][()]
    f.close()
    for i in range(N):
        if abs(P_sol[0, i]) <= 1e-3:
            P_sol[0, i] = 0
        else:
            pass
    index = np.argwhere(P_sol[0, 0:30])
    for i in index:
        P_sol[0, i] -= sum(P_sol[0, :]) / index.shape[1]
    P_sol = np.array([i*baseMVA/(M*omega_s) for i in P_sol])

    Y = np.array([i*baseMVA/(M*omega_s) for i in Y])
    # 参数合并
    PY = np.vstack((P_sol, Y))
    PY = PY / 16
    YY = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if Y[i, j] != 0.:
                YY[i, j] = 1
    
    Y /= 16
    P_sol /= 16
    if net == 'GCN' or 'RGCN' or 'RGCN-TCN':
        if adj_mode == 1:
            Y = YY + np.eye(N)
        elif adj_mode == 2:
            Y = Y + np.diag(abs(np.squeeze(P_sol)))
            Y = Y / np.amax(Y)
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
def classify_random_2(N, x_theta, x_omega, y_data, test_size, timelength, CHANNEL):
    """
    randomly classify data to train and test group
    (2, timelength, N)
    """
    if CHANNEL == 2:
        x_data = np.dstack((x_omega,x_theta))
        x_train, X_test, y_train, Y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=0) # 随机分组
        X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=0) # 随机分组
        print(X_train.shape)
        X_train = np.reshape(X_train, (X_train.shape[0], N, timelength, 2))
        X_train = np.swapaxes(X_train,1,3)
        X_val = np.reshape(X_val, (X_val.shape[0], N, timelength, 2))
        X_val = np.swapaxes(X_val,1,3)
        X_test = np.reshape(X_test, (X_test.shape[0], N, timelength, 2))
        X_test = np.swapaxes(X_test,1,3)
    elif CHANNEL ==1:
        x_train, X_test, y_train, Y_test = train_test_split(x_omega, y_data, test_size=test_size, random_state=0) # 随机分组
        X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=0) # 随机分组
        X_train = np.reshape(X_train, (X_train.shape[0], N, timelength, 1))
        X_train = np.swapaxes(X_train,1,3)
        X_val = np.reshape(X_val, (X_val.shape[0], N, timelength, 1))
        X_val = np.swapaxes(X_val,1,3)
        X_test = np.reshape(X_test, (X_test.shape[0], N, timelength, 1))
        X_test = np.swapaxes(X_test,1,3)
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def classify_random_4(N, x_theta, x_omega, y_data, test_size, timelength, chosedlength, CHANNEL):
    """
    randomly classify data to train and test group
    
    (N, chosedlength)
    """
    if CHANNEL == 2:
        x_data = np.dstack((x_omega,x_theta))
        x_train, X_test, y_train, Y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=0) # 随机分组
        X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=0) # 随机分组
        del x_data, y_data, x_train, y_train
        X_train = np.reshape(X_train, (X_train.shape[0], N, chosedlength, 2))
        X_train = np.swapaxes(X_train,1,3)
        X_val = np.reshape(X_val, (X_val.shape[0], N, chosedlength, 2))
        X_val = np.swapaxes(X_val,1,3)
        X_test = np.reshape(X_test, (X_test.shape[0], N, chosedlength, 2))
        X_test = np.swapaxes(X_test,1,3)
    elif CHANNEL ==1:
        x_train, X_test, y_train, Y_test = train_test_split(x_omega, y_data, test_size=test_size, random_state=0) # 随机分组
        X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=0) # 随机分组
        del y_data, x_train, y_train
        X_train = np.reshape(X_train, (X_train.shape[0], N, chosedlength))
        # X_train = np.swapaxes(X_train,1,3)
        X_val = np.reshape(X_val, (X_val.shape[0], N, chosedlength))
        # X_val = np.swapaxes(X_val,1,3)
        X_test = np.reshape(X_test, (X_test.shape[0], N, chosedlength))
        # X_test = np.swapaxes(X_test,1,3)
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

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
load large data
"""
## recurrent
def load_data_two_recurrent_l(N, omega, timelength, chosedlength, TEST_SIZE, CHANNEL):

    start = time.perf_counter()

    X_two_theta_2 = np.zeros((4000, chosedlength*N))
    X_two_omega_2 = np.zeros((4000, chosedlength*N))

    f = h5py.File('/public/home/spy2018/swing/result/recurrent/case%s/omega=%s/change_two_node_long/4000/1_2.h5' %(N, omega), 'r')

    for i in range(N):
        X_two_theta_2[:, i*chosedlength:(i+1)*chosedlength] = f['data_theta'][()][:, i*timelength:i*timelength+chosedlength]
        X_two_omega_2[:, i*chosedlength:(i+1)*chosedlength] = f['data_omega'][()][:, i*timelength:i*timelength+chosedlength]
    Y_two_2 = f['Y'][()]
    f.close()
    del f

    for i in range(N-1):
        for j in range(i+1, N):
            if i == 0 and j == 1:
                pass
            else:
                X_theta = np.zeros((4000, chosedlength*N))
                X_omega = np.zeros((4000, chosedlength*N))

                f = h5py.File('/public/home/spy2018/swing/result/recurrent/case%s/omega=%s/change_two_node_long/4000/%s_%s.h5' % (N, omega, i+1, j+1), 'r')
                for ii in range(N):
                    X_theta[:, ii*chosedlength:(ii+1)*chosedlength] = f['data_theta'][()][:, ii*timelength:ii*timelength+chosedlength]
                    X_omega[:, ii*chosedlength:(ii+1)*chosedlength] = f['data_omega'][()][:, ii*timelength:ii*timelength+chosedlength]
                Y = f['Y'][()]

                X_two_theta_2 = np.vstack((X_two_theta_2, X_theta))
                X_two_omega_2 = np.vstack((X_two_omega_2, X_omega))
                Y_two_2 = np.hstack((Y_two_2, Y))
                del f, X_theta, X_omega, Y
    end = time.perf_counter()
    print('所用时间为%ss' % (str(end - start)))
    X_train, X_val, X_test, Y_train, Y_val, Y_test = classify_random_4(x_theta=X_two_theta_2,
                                                                       x_omega=X_two_omega_2,
                                                                       y_data=Y_two_2,
                                                                       test_size=TEST_SIZE,
                                                                       chosedlength=chosedlength,
                                                                       CHANNEL=CHANNEL)

    a = len(Y_train)+len(Y_val)+len(Y_test)
    print('数据总数为:%s' %(a))
    b = int(np.sum(Y_train)+np.sum(Y_val)+np.sum(Y_test))

    print('同步状态数量:非同步状态=%s:%s' %(a-b, b))

    print('训练集：',X_train.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_train)-int(np.sum(Y_train)), int(np.sum(Y_train))))
    print('验证集：',X_val.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_val)-int(np.sum(Y_val)), int(np.sum(Y_val))))
    print('测试集：',X_test.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_test)-int(np.sum(Y_test)), int(np.sum(Y_test))))

    print('数据分组完毕，开始训练')
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, a, b

def load_data_three_recurrent_l(N, omega, timelength, chosedlength, TEST_SIZE, CHANNEL):
    
    start = time.perf_counter()

    ## load data 1000*364

    X_three_theta_2 = np.zeros((1000, chosedlength*N))
    X_three_omega_2 = np.zeros((1000, chosedlength*N))

    f = h5py.File('/public/home/spy2018/swing/result/recurrent/case%s/omega=%s/change_three_node_long/1000/1_2_3.h5' %(N, omega), 'r')
    for i in range(N):
        X_three_theta_2[:, i*chosedlength:(i+1)*chosedlength] = f['data_theta'][()][:, i*timelength:i*timelength+chosedlength]
        X_three_omega_2[:, i*chosedlength:(i+1)*chosedlength] = f['data_omega'][()][:, i*timelength:i*timelength+chosedlength]
    Y_three_2 = f['Y'][()]
    f.close()
    del f

    for i in range(N-2):
        for j in range(i+1, N-1):
            for k in range(j+1, N):
                if i == 0 and j == 1 and k == 2:
                    pass
                else:
                    X_theta = np.zeros((1000, chosedlength*N))
                    X_omega = np.zeros((1000, chosedlength*N))

                    f = h5py.File('/public/home/spy2018/swing/result/recurrent/case%s/omega=%s/change_three_node_long/1000/%s_%s_%s.h5' % (N, omega, i+1, j+1, k+1), 'r')
                    for ii in range(N):
                        X_theta[:, ii*chosedlength:(ii+1)*chosedlength] = f['data_theta'][()][:, ii*timelength:ii*timelength+chosedlength]
                        X_omega[:, ii*chosedlength:(ii+1)*chosedlength] = f['data_omega'][()][:, ii*timelength:ii*timelength+chosedlength]
                    Y = f['Y'][()]

                    X_three_theta_2 = np.vstack((X_three_theta_2, X_theta))
                    X_three_omega_2 = np.vstack((X_three_omega_2, X_omega))
                    Y_three_2 = np.hstack((Y_three_2, Y))
                    del f, X_theta, X_omega, Y

    end = time.perf_counter()
    print('所用时间为%ss' % (str(end - start)))
    X_train, X_val, X_test, Y_train, Y_val, Y_test = classify_random_4(x_theta=X_three_theta_2,
                                                                       x_omega=X_three_omega_2,
                                                                       y_data=Y_three_2,
                                                                       test_size=TEST_SIZE,
                                                                       chosedlength=chosedlength,
                                                                       CHANNEL=CHANNEL)

    a = len(Y_train)+len(Y_val)+len(Y_test)
    print('数据总数为:%s' %(a))
    b = int(np.sum(Y_train)+np.sum(Y_val)+np.sum(Y_test))

    print('同步状态数量:非同步状态=%s:%s' %(a-b, b))

    print('训练集：',X_train.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_train)-int(np.sum(Y_train)), int(np.sum(Y_train))))
    print('验证集：',X_val.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_val)-int(np.sum(Y_val)), int(np.sum(Y_val))))
    print('测试集：',X_test.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_test)-int(np.sum(Y_test)), int(np.sum(Y_test))))

    print('数据分组完毕，开始训练')
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, a, b

def load_data_four_recurrent_l(N, omega, timelength, chosedlength, TEST_SIZE, CHANNEL):
    
    start = time.perf_counter()

    ## load data 1000*1001

    X_four_theta_2 = np.zeros((400, chosedlength*N))
    X_four_omega_2 = np.zeros((400, chosedlength*N))

    f = h5py.File('/public/home/spy2018/swing/result/recurrent/case%s/omega=%s/change_four_node_long/400/1_2_3_4.h5' %(N, omega), 'r')
    for i in range(N):
        X_four_theta_2[:, i*chosedlength:(i+1)*chosedlength] = f['data_theta'][()][:, i*timelength:i*timelength+chosedlength]
        X_four_omega_2[:, i*chosedlength:(i+1)*chosedlength] = f['data_omega'][()][:, i*timelength:i*timelength+chosedlength]
    Y_four_2 = f['Y'][()]
    f.close()
    del f

    for i in range(N-3):
        for j in range(i+1, N-2):
            for k in range(j+1, N -1):
                for l in range(k+1, N):
                    if i == 0 and j == 1 and k == 2 and l == 3:
                        pass
                    else:
                        X_theta = np.zeros((400, chosedlength*N))
                        X_omega = np.zeros((400, chosedlength*N))

                        f = h5py.File('/public/home/spy2018/swing/result/recurrent/case%s/omega=%s/change_four_node_long/400/%s_%s_%s_%s.h5' % (N, omega, i+1, j+1, k+1, l+1), 'r')
                        for ii in range(N):
                            X_theta[:, ii*chosedlength:(ii+1)*chosedlength] = f['data_theta'][()][:, ii*timelength:ii*timelength+chosedlength]
                            X_omega[:, ii*chosedlength:(ii+1)*chosedlength] = f['data_omega'][()][:, ii*timelength:ii*timelength+chosedlength]
                        Y = f['Y'][()]

                        X_four_theta_2 = np.vstack((X_four_theta_2, X_theta))
                        X_four_omega_2 = np.vstack((X_four_omega_2, X_omega))
                        Y_four_2 = np.hstack((Y_four_2, Y))
                        del f, X_theta, X_omega, Y

    end = time.perf_counter()
    print('所用时间为%ss' % (str(end - start)))
    X_train, X_val, X_test, Y_train, Y_val, Y_test = classify_random_4(x_theta=X_four_theta_2,
                                                                       x_omega=X_four_omega_2,
                                                                       y_data=Y_four_2,
                                                                       test_size=TEST_SIZE,
                                                                       chosedlength=chosedlength,
                                                                       CHANNEL=CHANNEL)

    a = len(Y_train)+len(Y_val)+len(Y_test)
    print('数据总数为:%s' %(a))
    b = int(np.sum(Y_train)+np.sum(Y_val)+np.sum(Y_test))

    print('同步状态数量:非同步状态=%s:%s' %(a-b, b))

    print('训练集：',X_train.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_train)-int(np.sum(Y_train)), int(np.sum(Y_train))))
    print('验证集：',X_val.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_val)-int(np.sum(Y_val)), int(np.sum(Y_val))))
    print('测试集：',X_test.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_test)-int(np.sum(Y_test)), int(np.sum(Y_test))))

    print('数据分组完毕，开始训练')
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, a, b

def load_data_two_three_four_recurrent_l(N, omega, timelength, chosedlength, TEST_SIZE, CHANNEL):

    start = time.perf_counter()

    X_two_train, X_two_val, X_two_test, Y_two_train, Y_two_val, Y_two_test, a_two, b_two = load_data_two_recurrent_l(N=N,
                                                                                                                     omega=omega,
                                                                                                                     timelength=timelength,
                                                                                                                     chosedlength=chosedlength,
                                                                                                                     TEST_SIZE=TEST_SIZE,
                                                                                                                     CHANNEL=CHANNEL)
    
    X_three_train, X_three_val, X_three_test, Y_three_train, Y_three_val, Y_three_test, a_three, b_three = load_data_three_recurrent_l(N=N,
                                                                                                                                      omega=omega,
                                                                                                                                      timelength=timelength,
                                                                                                                                      chosedlength=chosedlength,
                                                                                                                                      TEST_SIZE=TEST_SIZE,
                                                                                                                                      CHANNEL=CHANNEL)
    
    X_four_train, X_four_val, X_four_test, Y_four_train, Y_four_val, Y_four_test, a_four, b_four = load_data_four_recurrent_l(N=N,
                                                                                                                             omega=omega,
                                                                                                                             timelength=timelength,
                                                                                                                             chosedlength=chosedlength,
                                                                                                                             TEST_SIZE=TEST_SIZE,
                                                                                                                             CHANNEL=CHANNEL)
    
    a = a_two + a_three + a_four
    b = b_two + b_three + b_four

    ## dataset
    X_all_train = np.concatenate((X_two_train, X_three_train, X_four_train),axis=0)
    del X_two_train, X_three_train, X_four_train

    X_all_val = np.concatenate((X_two_val, X_three_val, X_four_val),axis=0)
    del X_two_val, X_three_val, X_four_val

    X_all_test = np.concatenate((X_two_test, X_three_test, X_four_test),axis=0)
    del X_two_test, X_three_test, X_four_test

    ## labels
    Y_all_train = np.concatenate((Y_two_train, Y_three_train, Y_four_train),axis=0)
    del Y_two_train, Y_three_train, Y_four_train

    Y_all_val = np.concatenate((Y_two_val, Y_three_val, Y_four_val),axis=0)
    del Y_two_val, Y_three_val, Y_four_val

    Y_all_test = np.concatenate((Y_two_test, Y_three_test, Y_four_test),axis=0)
    del Y_two_test, Y_three_test, Y_four_test

    end = time.perf_counter()

    print('所用时间为%ss' % (str(end - start)))

    print('数据总数为:%s' %(a))
    print('同步状态数量:非同步状态=%s:%s' %(a-b, b))

    print('训练集：',X_all_train.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_all_train)-int(np.sum(Y_all_train)), int(np.sum(Y_all_train))))
    print('验证集：',X_all_val.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_all_val)-int(np.sum(Y_all_val)), int(np.sum(Y_all_val))))
    print('测试集：',X_all_test.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_all_test)-int(np.sum(Y_all_test)), int(np.sum(Y_all_test))))

    print('数据分组完毕，开始训练')

    return X_all_train, X_all_val, X_all_test, Y_all_train, Y_all_val, Y_all_test, a, b

## IEEE

def load_data_one_IEEE_l(N, omega, timelength, chosedlength, TEST_SIZE, CHANNEL, interval, relative, normalize, standard, mode, move, WSZ):
    """
    no oversampling
    """
    start = time.perf_counter()
    if N == 14:
        length = 4000 # load data 4000*14
    elif N == 39:
        length = 441 # load data 441*39
    elif N == 118:
        length = 441 # load data 441*118
    mode = 12
    X_one_theta_2 = np.zeros((length, chosedlength*N))
    X_one_omega_2 = np.zeros((length, chosedlength*N))
    if N == 39:
        if interval == True:
            f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s/1.h5' %(N, omega, length), 'r')
            # f = h5py.File('/media/duguyuan/new/Swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s/1.h5' %(N, omega, length), 'r')
        else:
            f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s/1.h5' %(N, omega, mode), 'r')
            # f = h5py.File('/media/duguyuan/new/Swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s/1.h5' %(N, omega, mode), 'r')
    elif N == 118:
        if interval == True:
            f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega=%s/change_one_node_long/%s/1.h5' %(N, omega, length), 'r')
            # f = h5py.File('/media/duguyuan/new/Swing/result/IEEE/case%s/omega=%s/change_one_node_long/%s/1.h5' %(N, omega, length), 'r')
        else:
            f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega=%s/change_one_node_long/%s_no_interval/1.h5' %(N, omega, length), 'r')
            # f = h5py.File('/media/duguyuan/new/Swing/result/IEEE/case%s/omega=%s/change_one_node_long/%s_no_interval/1.h5' %(N, omega, length), 'r')
    
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

            if N == 39:
                if interval == True:
                    f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s/%s.h5' % (N, omega, length, i+1), 'r')
                    # f = h5py.File('/media/duguyuan/new/Swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s/%s.h5' % (N, omega, length, i+1), 'r')
                else:
                    f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s/%s.h5' % (N, omega, mode, i+1), 'r')
                    # f = h5py.File('/media/duguyuan/new/Swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s/%s.h5' % (N, omega, mode, i+1), 'r')
            elif N == 118:
                if interval == True:
                    f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega=%s/change_one_node_long/%s/%s.h5' % (N, omega, length, i+1), 'r')
                    # f = h5py.File('/media/duguyuan/new/Swing/result/IEEE/case%s/omega=%s/change_one_node_long/%s/%s.h5' % (N, omega, length, i+1), 'r')
                else:
                    f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega=%s/change_one_node_long/%s_no_interval/%s.h5' % (N, omega, length, i+1), 'r')
                    # f = h5py.File('/media/duguyuan/new/Swing/result/IEEE/case%s/omega=%s/change_one_node_long/%s_no_interval/%s.h5' % (N, omega, length, i+1), 'r')
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

def load_data_one_IEEE_l_oversampling(N, omega, timelength, chosedlength, TEST_SIZE, CHANNEL, interval, relative, normalize, standard, mode, move, WSZ):
    
    start = time.perf_counter()
    if N == 14:
        length = 4000
    elif N == 39:
        length = 1000
    elif N == 118:
        length = 441
    ## load data 4000*14

    X_one_theta_2 = np.zeros((length, chosedlength*N))
    X_one_omega_2 = np.zeros((length, chosedlength*N))
    if N == 39:
        if interval == True:
            f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s/1.h5' %(N, omega, length), 'r')
            # f = h5py.File('/media/duguyuan/new/Swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s/1.h5' %(N, omega, length), 'r')
        else:
            f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s_no_interval/1.h5' %(N, omega, length), 'r')
            # f = h5py.File('/media/duguyuan/new/Swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s_no_interval/1.h5' %(N, omega, length), 'r')
    elif N == 118:
        if interval == True:
            f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega=%s/change_one_node_long/%s/1.h5' %(N, omega, length), 'r')
            # f = h5py.File('/media/duguyuan/new/Swing/result/IEEE/case%s/omega=%s/change_one_node_long/%s/1.h5' %(N, omega, length), 'r')
        else:
            f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega=%s/change_one_node_long/%s_no_interval/1.h5' %(N, omega, length), 'r')
            # f = h5py.File('/media/duguyuan/new/Swing/result/IEEE/case%s/omega=%s/change_one_node_long/%s_no_interval/1.h5' %(N, omega, length), 'r')
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
        elif not os.path.exists('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s_no_interval/%s.h5' % (N, omega, length, i+1)):
            pass
        else:
            X_theta = np.zeros((length, chosedlength*N))
            X_omega = np.zeros((length, chosedlength*N))

            if N == 39:
                if interval == True:
                    f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s/%s.h5' % (N, omega, length, i+1), 'r')
                    # f = h5py.File('/media/duguyuan/new/Swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s/%s.h5' % (N, omega, length, i+1), 'r')
                else:
                    f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s_no_interval/%s.h5' % (N, omega, length, i+1), 'r')
                    # f = h5py.File('/media/duguyuan/new/Swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s_no_interval/%s.h5' % (N, omega, length, i+1), 'r')
            elif N == 118:
                if interval == True:
                    f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega=%s/change_one_node_long/%s/%s.h5' % (N, omega, length, i+1), 'r')
                    # f = h5py.File('/media/duguyuan/new/Swing/result/IEEE/case%s/omega=%s/change_one_node_long/%s/%s.h5' % (N, omega, length, i+1), 'r')
                else:
                    f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega=%s/change_one_node_long/%s_no_interval/%s.h5' % (N, omega, length, i+1), 'r')
                    # f = h5py.File('/media/duguyuan/new/Swing/result/IEEE/case%s/omega=%s/change_one_node_long/%s_no_interval/%s.h5' % (N, omega, length, i+1), 'r')
            if chosedlength != timelength:
                for ii in range(N):
                    X_theta[:, ii*chosedlength:(ii+1)*chosedlength] = f['data_theta'][()][:, ii*timelength:ii*timelength+chosedlength]
                    X_omega[:, ii*chosedlength:(ii+1)*chosedlength] = f['data_omega'][()][:, ii*timelength:ii*timelength+chosedlength]
            else:
                x_theta = f['data_theta'][()]
                x_omega = f['data_omega'][()]
            Y = f['Y'][()]
            f.close()
            del f
            X_one_theta_2 = np.vstack((X_one_theta_2, X_theta))
            X_one_omega_2 = np.vstack((X_one_omega_2, X_omega))
            Y_one_2 = np.hstack((Y_one_2, Y))
            del X_theta, X_omega, Y

    Y_1 = np.argwhere(Y_one_2 > 0)
    x_theta = np.zeros((Y_1.shape[0],N*chosedlength))
    x_omega = np.zeros((Y_1.shape[0],N*chosedlength))
    i = 0
    for j in Y_1:
        x_omega[i,:] = X_one_omega_2[j,:]
        x_theta[i,:] = X_one_theta_2[j,:]
        i += 1

    for i in range(4):
        # x_theta = X_one_theta_2[Y_1,:,:] + np.random.normal(loc=0,scale=0.4,size=(Y_1.shape[0], N, chosedlength))
        X_one_theta_2 = np.vstack((X_one_theta_2, x_theta + np.random.normal(loc=0,scale=0.4,size=(Y_1.shape[0], N*chosedlength))))
        # x_omega = X_one_omega_2[Y_1,:,:] + np.random.normal(loc=0,scale=0.2,size=(Y_1.shape[0], N, chosedlength))
        X_one_omega_2 = np.vstack((X_one_omega_2, x_omega + np.random.normal(loc=0,scale=0.2,size=(Y_1.shape[0], N*chosedlength))))

        Y_one_2 = np.hstack((Y_one_2, np.ones(Y_1.shape[0])))
    
    end = time.perf_counter()
    print('所用时间为%ss' % (str(end - start)))
    
    X_one_theta_2 = np.float32(X_one_theta_2)
    X_one_omega_2 = np.float32(X_one_omega_2)
    if normalize:
        X_one_omega_norm, X_one_theta_norm = normalization(
            N=N, x_theta=X_one_theta_2, x_omega=X_one_omega_2
        )
        del X_one_theta_2, X_one_omega_2
        X_train, X_val, X_test, Y_train, Y_val, Y_test = classify_random_4(
            x_theta=X_one_theta_norm,
            x_omega=X_one_omega_norm,
            y_data=Y_one_2,
            test_size=TEST_SIZE,
            timelength=timelength,
            chosedlength=chosedlength,
            CHANNEL=CHANNEL,
            N=N
        )
    elif standard:
        X_one_omega_std, X_one_theta_std = standardization(
            N=N, chosedlength=chosedlength, x_theta=X_one_theta_2, x_omega=X_one_omega_2, relative=relative, mode=mode
        )
        del X_one_theta_2, X_one_omega_2
        X_train, X_val, X_test, Y_train, Y_val, Y_test = classify_random_4(
            x_theta=X_one_theta_std,
            x_omega=X_one_omega_std,
            y_data=Y_one_2,
            test_size=TEST_SIZE,
            timelength=timelength,
            chosedlength=chosedlength,
            CHANNEL=CHANNEL,
            N=N
        )
    else:
        X_train, X_val, X_test, Y_train, Y_val, Y_test = classify_random_4(
            x_theta=X_one_theta_2,
            x_omega=X_one_omega_2,
            y_data=Y_one_2,
            test_size=TEST_SIZE,
            timelength=timelength,
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

    print('训练集：',X_train.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_train)-int(np.sum(Y_train)), int(np.sum(Y_train))))
    print('验证集：',X_val.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_val)-int(np.sum(Y_val)), int(np.sum(Y_val))))
    print('测试集：',X_test.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_test)-int(np.sum(Y_test)), int(np.sum(Y_test))))

    print('数据分组完毕，开始训练')
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, a, b

def load_data_one_IEEE_l_oversampling_2(N, omega, timelength, chosedlength, TEST_SIZE, CHANNEL, interval, relative, normalize, standard, mode, move, WSZ):
    
    start = time.perf_counter()
    if N == 14:
        length = 4000
    elif N == 39:
        length = 1000
    elif N == 118:
        length = 441
    ## load data 4000*14

    X_one_theta_2 = np.zeros((length, chosedlength*N))
    X_one_omega_2 = np.zeros((length, chosedlength*N))
    if N == 39:
        if interval == True:
            # f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s/1.h5' %(N, omega, length), 'r')
            f = h5py.File('/media/duguyuan/new/Swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s/1.h5' %(N, omega, length), 'r')
        else:
            # f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s_no_interval/1.h5' %(N, omega, length), 'r')
            f = h5py.File('/media/duguyuan/new/Swing/result/IEEE/case%s/omega==%s/change_one_node_long/1/1.h5' %(N, omega), 'r')
    elif N == 118:
        if interval == True:
            # f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega=%s/change_one_node_long/%s/1.h5' %(N, omega, length), 'r')
            f = h5py.File('/media/duguyuan/new/Swing/result/IEEE/case%s/omega=%s/change_one_node_long/%s/1.h5' %(N, omega, length), 'r')
        else:
            # f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega=%s/change_one_node_long/%s_no_interval/1.h5' %(N, omega, length), 'r')
            f = h5py.File('/media/duguyuan/new/Swing/result/IEEE/case%s/omega=%s/change_one_node_long/%s_no_interval/1.h5' %(N, omega, length), 'r')
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
        # elif not os.path.exists('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s_no_interval/%s.h5' % (N, omega, length, i+1)):
        elif not os.path.exists('/media/duguyuan/new/Swing/result/IEEE/case%s/omega==%s/change_one_node_long/1/%s.h5' % (N, omega, i+1)):
            pass
        else:
            X_theta = np.zeros((length, chosedlength*N))
            X_omega = np.zeros((length, chosedlength*N))
    
            if N == 39:
                if interval == True:
                    # f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s/%s.h5' % (N, omega, length, i+1), 'r')
                    f = h5py.File('/media/duguyuan/new/Swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s/%s.h5' % (N, omega, length, i+1), 'r')
                else:
                    # f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s_no_interval/%s.h5' % (N, omega, length, i+1), 'r')
                    f = h5py.File('/media/duguyuan/new/Swing/result/IEEE/case%s/omega==%s/change_one_node_long/1/%s.h5' % (N, omega, i+1), 'r')
            elif N == 118:
                if interval == True:
                    # f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega=%s/change_one_node_long/%s/%s.h5' % (N, omega, length, i+1), 'r')
                    f = h5py.File('/media/duguyuan/new/Swing/result/IEEE/case%s/omega=%s/change_one_node_long/%s/%s.h5' % (N, omega, length, i+1), 'r')
                else:
                    # f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega=%s/change_one_node_long/%s_no_interval/%s.h5' % (N, omega, length, i+1), 'r')
                    f = h5py.File('/media/duguyuan/new/Swing/result/IEEE/case%s/omega=%s/change_one_node_long/%s_no_interval/%s.h5' % (N, omega, length, i+1), 'r')
            if chosedlength != timelength:
                for ii in range(N):
                    X_theta[:, ii*chosedlength:(ii+1)*chosedlength] = f['data_theta'][()][:, ii*timelength:ii*timelength+chosedlength]
                    X_omega[:, ii*chosedlength:(ii+1)*chosedlength] = f['data_omega'][()][:, ii*timelength:ii*timelength+chosedlength]
            else:
                x_theta = f['data_theta'][()]
                x_omega = f['data_omega'][()]
            Y = f['Y'][()]
            f.close()
            del f
            X_one_theta_2 = np.vstack((X_one_theta_2, X_theta))
            X_one_omega_2 = np.vstack((X_one_omega_2, X_omega))
            Y_one_2 = np.hstack((Y_one_2, Y))
            del X_theta, X_omega, Y

    Y_1 = np.argwhere(Y_one_2 > 0)
    x_theta = np.zeros((Y_1.shape[0],N*chosedlength))
    x_omega = np.zeros((Y_1.shape[0],N*chosedlength))
    i = 0
    for j in Y_1:
        x_omega[i,:] = X_one_omega_2[j,:]
        x_theta[i,:] = X_one_theta_2[j,:]
        i += 1

    for i in range(4):
        # x_theta = X_one_theta_2[Y_1,:,:] + np.random.normal(loc=0,scale=0.4,size=(Y_1.shape[0], N, chosedlength))
        X_one_theta_2 = np.vstack((X_one_theta_2, x_theta + np.random.normal(loc=0,scale=0.4,size=(Y_1.shape[0], N*chosedlength))))
        # x_omega = X_one_omega_2[Y_1,:,:] + np.random.normal(loc=0,scale=0.2,size=(Y_1.shape[0], N, chosedlength))
        X_one_omega_2 = np.vstack((X_one_omega_2, x_omega + np.random.normal(loc=0,scale=0.2,size=(Y_1.shape[0], N*chosedlength))))

        Y_one_2 = np.hstack((Y_one_2, np.ones(Y_1.shape[0])))
    
    end = time.perf_counter()
    print('所用时间为%ss' % (str(end - start)))
    
    X_one_theta_2 = np.float32(X_one_theta_2)
    X_one_omega_2 = np.float32(X_one_omega_2)
    if normalize:
        X_one_omega_norm, X_one_theta_norm = normalization(
            N=N, x_theta=X_one_theta_2, x_omega=X_one_omega_2
        )
        del X_one_theta_2, X_one_omega_2
        X_train, X_train_theta, X_val, X_val_theta, X_test, X_test_theta, Y_train, Y_val, Y_test = classify_random_5(
            x_theta=X_one_theta_norm,
            x_omega=X_one_omega_norm,
            y_data=Y_one_2,
            test_size=TEST_SIZE,
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

    print('训练集：',X_train.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_train)-int(np.sum(Y_train)), int(np.sum(Y_train))))
    print('验证集：',X_val.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_val)-int(np.sum(Y_val)), int(np.sum(Y_val))))
    print('测试集：',X_test.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_test)-int(np.sum(Y_test)), int(np.sum(Y_test))))

    print('数据分组完毕，开始训练')
    
    return X_train, X_train_theta, X_val, X_val_theta, X_test, X_test_theta, Y_train, Y_val, Y_test, a, b

def load_data_two_IEEE_l(N, omega, timelength, chosedlength, TEST_SIZE, CHANNEL, interval, relative, normalize, standard, mode, move, WSZ):
    
    start = time.perf_counter()
    
    if   omega == 100:

        ## Load data 1600*91

        X_two_theta_1 = np.zeros(((N-1)*1600,chosedlength*N))
        X_two_omega_1 = np.zeros(((N-1)*1600,chosedlength*N))

        f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega=%s/change_two_node_long/1600/data_1.h5' %(N, omega), 'r')
        if chosedlength != timelength:
            for i in range(N):
                X_two_theta_1[:, i*chosedlength:(i+1)*chosedlength] = f['theta'][()][:, i*timelength:i*timelength+chosedlength]
                X_two_omega_1[:, i*chosedlength:(i+1)*chosedlength] = f['omega'][()][:, i*timelength:i*timelength+chosedlength]
        else:
            x_two_theta_1 = f['theta'][()]
            x_two_omega_1 = f['omega'][()]
        
        Y_two_1 = f['Y'][()]
        del f

        for i in range(1, N-1):

            X_theta = np.zeros(((N-1-i)*1600,chosedlength*N))
            X_omega = np.zeros(((N-1-i)*1600,chosedlength*N))

            f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega=%s/change_two_node_long/1600/data_%s.h5' %(N, omega, i+1), 'r')
            if chosedlength != timelength:
                for ii in range(N):
                    X_theta[:, ii*chosedlength:(ii+1)*chosedlength] = f['theta'][()][:, ii*timelength:ii*timelength+chosedlength]
                    X_omega[:, ii*chosedlength:(ii+1)*chosedlength] = f['omega'][()][:, ii*timelength:ii*timelength+chosedlength]
            else:
                x_theta = f['theta'][()]
                x_omega = f['omega'][()]
            Y = f['Y'][()]

            X_two_theta_1 = np.vstack((X_two_theta_1, X_theta))
            X_two_omega_1 = np.vstack((X_two_omega_1, X_omega))
            Y_two_1 = np.hstack((Y_two_1, Y))
            del f, X_theta, X_omega, Y

        ## load data 2000*91

        X_two_theta_2 = np.zeros((2000, chosedlength*N))
        X_two_omega_2 = np.zeros((2000, chosedlength*N))

        f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega=%s/change_two_node_long/2000/1_2.h5' %(N, omega), 'r')

        for i in range(N):
            X_two_theta_2[:, i*chosedlength:(i+1)*chosedlength] = f['data_theta'][()][:, i*timelength:i*timelength+chosedlength]
            X_two_omega_2[:, i*chosedlength:(i+1)*chosedlength] = f['data_omega'][()][:, i*timelength:i*timelength+chosedlength]
        Y_two_2 = f['Y'][()]
        
        del f

        for i in range(N-1):
            for j in range(i+1, N):
                if i == 0 and j == 1:
                    pass
                else:
                    X_theta = np.zeros((2000, chosedlength*N))
                    X_omega = np.zeros((2000, chosedlength*N))

                    f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega=%s/change_two_node_long/2000/%s_%s.h5' % (N, omega, i+1, j+1), 'r')
                    for ii in range(N):
                        X_theta[:, ii*chosedlength:(ii+1)*chosedlength] = f['data_theta'][()][:, ii*timelength:ii*timelength+chosedlength]
                        X_omega[:, ii*chosedlength:(ii+1)*chosedlength] = f['data_omega'][()][:, ii*timelength:ii*timelength+chosedlength]
                    Y = f['Y'][()]

                    X_two_theta_2 = np.vstack((X_two_theta_2, X_theta))
                    X_two_omega_2 = np.vstack((X_two_omega_2, X_omega))
                    Y_two_2 = np.hstack((Y_two_2, Y))
                    del f, X_theta, X_omega, Y

        ## merge
        X_two_theta = np.vstack((X_two_theta_1, X_two_theta_2))
        X_two_omega = np.vstack((X_two_omega_1, X_two_omega_2))
        Y_two = np.hstack((Y_two_1, Y_two_2))
    
    elif omega == 20:
        
        if N == 14:
            length = 4000
        elif N == 39:
            length = 500

        X_two_theta_2 = np.zeros((length, chosedlength*N))
        X_two_omega_2 = np.zeros((length, chosedlength*N))

        if interval == True:
            f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_two_node_long/%s/1_2.h5' %(N, omega, length), 'r')
        else:
            f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_two_node_long/%s_no_interval/1_2.h5' %(N, omega, length), 'r')
        
        if chosedlength != timelength:
            for i in range(N):
                X_two_theta_2[:, i*chosedlength:(i+1)*chosedlength] = f['data_theta'][()][:, i*timelength:i*timelength+chosedlength]
                X_two_omega_2[:, i*chosedlength:(i+1)*chosedlength] = f['data_omega'][()][:, i*timelength:i*timelength+chosedlength]
        else:
            x_two_theta_2 = f['data_theta'][()]
            x_two_omega_2 = f['data_omega'][()]

        Y_two_2 = f['Y'][()]
        f.close()
        del f

        for i in range(N-1):
            for j in range(i+1, N):
                if i == 0 and j == 1:
                    pass
                else:
                    X_theta = np.zeros((length, chosedlength*N))
                    X_omega = np.zeros((length, chosedlength*N))
                    
                    if interval == True:
                        f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_two_node_long/%s/%s_%s.h5' % (N, omega, length, i+1, j+1), 'r')
                    else:
                        f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_two_node_long/%s_no_interval/%s_%s.h5' % (N, omega, length, i+1, j+1), 'r')
                    if chosedlength != timelength:
                        for ii in range(N):
                            X_theta[:, ii*chosedlength:(ii+1)*chosedlength] = f['data_theta'][()][:, ii*timelength:ii*timelength+chosedlength]
                            X_omega[:, ii*chosedlength:(ii+1)*chosedlength] = f['data_omega'][()][:, ii*timelength:ii*timelength+chosedlength]
                    else:
                        x_theta = f['data_theta'][()]
                        x_omega = f['data_omega'][()]
                    Y = f['Y'][()]
                    
                    f.close()
                    
                    X_two_theta_2 = np.vstack((X_two_theta_2, X_theta))
                    X_two_omega_2 = np.vstack((X_two_omega_2, X_omega))
                    Y_two_2 = np.hstack((Y_two_2, Y))
                    del f, X_theta, X_omega, Y
    
    end = time.perf_counter()
    print('所用时间为%ss' % (str(end - start)))

    X_two_theta_2 = np.float32(X_two_theta_2)
    X_two_omega_2 = np.float32(X_two_omega_2)
    if normalize:
        X_two_omega_norm, X_two_theta_norm = normalization(x_theta=X_two_theta_2, x_omega=X_two_omega_2)
        del X_two_theta_2, X_two_omega_2
        X_train, X_val, X_test, Y_train, Y_val, Y_test = classify_random_4(x_theta=X_two_theta_norm,
                                                                        x_omega=X_two_omega_norm,
                                                                        y_data=Y_two_2,
                                                                        test_size=TEST_SIZE,
                                                                        chosedlength=chosedlength,
                                                                        timelength=timelength,
                                                                        CHANNEL=CHANNEL,
                                                                        N=N)
    elif standard:
        X_two_omega_std, X_two_theta_std = standardization(N=N, chosedlength=chosedlength, x_theta=X_two_theta_2, x_omega=X_two_omega_2, relative=relative, mode=mode)
        del X_two_theta_2, X_two_omega_2
        X_train, X_val, X_test, Y_train, Y_val, Y_test = classify_random_4(x_theta=X_two_theta_std,
                                                                        x_omega=X_two_omega_std,
                                                                        y_data=Y_two_2,
                                                                        test_size=TEST_SIZE,
                                                                        chosedlength=chosedlength,
                                                                        timelength=timelength,
                                                                        CHANNEL=CHANNEL,
                                                                        N=N)
    else:
        X_train, X_val, X_test, Y_train, Y_val, Y_test = classify_random_4(x_theta=X_two_theta_2,
                                                                        x_omega=X_two_omega_2,
                                                                        y_data=Y_two_2,
                                                                        test_size=TEST_SIZE,
                                                                        chosedlength=chosedlength,
                                                                        timelength=timelength,
                                                                        CHANNEL=CHANNEL,
                                                                        N=N)
        del X_two_theta_2, X_two_omega_2
    
    if move:
        for i in range(N):
            for j in range(X_train.shape[0]):
                X_train[j, 0, :, i] = smooth(a=X_train[j, 0, :, i], WSZ=WSZ)
                X_train[j, 1, :, i] = smooth(a=X_train[j, 1, :, i], WSZ=WSZ)
            for j in range(X_val.shape[0]):
                X_val[j, 0, :, i] = smooth(a=X_val[j, 0, :, i], WSZ=WSZ)
                X_val[j, 1, :, i] = smooth(a=X_val[j, 1, :, i], WSZ=WSZ)
            for j in range(X_test.shape[0]):
                X_test[j, 0, :, i] = smooth(a=X_test[j, 0, :, i], WSZ=WSZ)
                X_test[j, 1, :, i] = smooth(a=X_test[j, 1, :, i], WSZ=WSZ)
        del i, j
    
    a = len(Y_train)+len(Y_val)+len(Y_test)
    print('数据总数为:%s' %(a))
    b = int(np.sum(Y_train)+np.sum(Y_val)+np.sum(Y_test))

    print('同步状态数量:非同步状态=%s:%s' %(a-b, b))

    print('训练集：',X_train.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_train)-int(np.sum(Y_train)), int(np.sum(Y_train))))
    print('验证集：',X_val.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_val)-int(np.sum(Y_val)), int(np.sum(Y_val))))
    print('测试集：',X_test.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_test)-int(np.sum(Y_test)), int(np.sum(Y_test))))

    print('数据分组完毕，开始训练')
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, a, b

def load_data_three_IEEE_l(N, omega, timelength, chosedlength, TEST_SIZE, CHANNEL, interval, relative, normalize, standard, mode, move, WSZ):
    
    start = time.perf_counter()

    if N == 14:
        length = 1000 # load data 1000*364
    
    X_three_theta_2 = np.zeros((length, chosedlength*N))
    X_three_omega_2 = np.zeros((length, chosedlength*N))

    if interval == True:
        f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_three_node_long/%s/1_2_3.h5' %(N, omega, length), 'r')
    else:
        f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_three_node_long/%s_no_interval/1_2_3.h5' %(N, omega, length), 'r')
    
    if chosedlength != timelength:
        for i in range(N):
            X_three_theta_2[:, i*chosedlength:(i+1)*chosedlength] = f['data_theta'][()][:, i*timelength:i*timelength+chosedlength]
            X_three_omega_2[:, i*chosedlength:(i+1)*chosedlength] = f['data_omega'][()][:, i*timelength:i*timelength+chosedlength]
    else:
        X_three_theta_2 = f['data_theta'][()]
        X_three_omega_2 = f['data_omega'][()]
        
    Y_three_2 = f['Y'][()]
    f.close()
    del f

    for i in range(N-2):
        for j in range(i+1, N-1):
            for k in range(j+1, N):
                if i == 0 and j == 1 and k == 2:
                    pass
                else:
                    X_theta = np.zeros((length, chosedlength*N))
                    X_omega = np.zeros((length, chosedlength*N))

                    if interval == True:
                        f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_three_node_long/%s/%s_%s_%s.h5' % (N, omega, length, i+1, j+1, k+1), 'r')
                    else:
                        f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_three_node_long/%s_no_interval/%s_%s_%s.h5' % (N, omega, length, i+1, j+1, k+1), 'r')
                    if chosedlength != timelength:
                        for ii in range(N):
                            X_theta[:, ii*chosedlength:(ii+1)*chosedlength] = f['data_theta'][()][:, ii*timelength:ii*timelength+chosedlength]
                            X_omega[:, ii*chosedlength:(ii+1)*chosedlength] = f['data_omega'][()][:, ii*timelength:ii*timelength+chosedlength]
                    else:
                        X_theta = f['data_theta'][()]
                        X_omega = f['data_omega'][()]
                    
                    Y = f['Y'][()]

                    X_three_theta_2 = np.vstack((X_three_theta_2, X_theta))
                    X_three_omega_2 = np.vstack((X_three_omega_2, X_omega))
                    Y_three_2 = np.hstack((Y_three_2, Y))
                    del f, X_theta, X_omega, Y

    end = time.perf_counter()
    print('所用时间为%ss' % (str(end - start)))

    X_three_theta_2 = np.float32(X_three_theta_2)
    X_three_omega_2 = np.float32(X_three_omega_2)
    if normalize:
        X_three_omega_norm, X_three_theta_norm = normalization(x_theta=X_three_theta_2, x_omega=X_three_omega_2)
        del X_three_theta_2, X_three_omega_2
        X_train, X_val, X_test, Y_train, Y_val, Y_test = classify_random_4(x_theta=X_three_theta_norm,
                                                                           x_omega=X_three_omega_norm,
                                                                           y_data=Y_three_2,
                                                                           test_size=TEST_SIZE,
                                                                           timelength=timelength,
                                                                           chosedlength=chosedlength,
                                                                           CHANNEL=CHANNEL,
                                                                           N=N)
    elif standard:
        X_three_omega_std, X_three_theta_std = standardization(N=N, chosedlength=chosedlength, x_theta=X_three_theta_2, x_omega=X_three_omega_2, relative=relative, mode=mode)
        del X_three_theta_2, X_three_omega_2
        X_train, X_val, X_test, Y_train, Y_val, Y_test = classify_random_4(x_theta=X_three_theta_std,
                                                                        x_omega=X_three_omega_std,
                                                                        y_data=Y_three_2,
                                                                        test_size=TEST_SIZE,
                                                                        timelength=timelength,
                                                                        chosedlength=chosedlength,
                                                                        CHANNEL=CHANNEL,
                                                                        N=N)
    else:
        X_train, X_val, X_test, Y_train, Y_val, Y_test = classify_random_4(x_theta=X_three_theta_2,
                                                                        x_omega=X_three_omega_2,
                                                                        y_data=Y_three_2,
                                                                        test_size=TEST_SIZE,
                                                                        timelength=timelength,
                                                                        chosedlength=chosedlength,
                                                                        CHANNEL=CHANNEL,
                                                                        N=N)
        del X_three_theta_2, X_three_omega_2
    
    if move:
        for i in range(N):
            for j in range(X_train.shape[0]):
                X_train[j, 0, :, i] = smooth(a=X_train[j, 0, :, i], WSZ=WSZ)
                X_train[j, 1, :, i] = smooth(a=X_train[j, 1, :, i], WSZ=WSZ)
            for j in range(X_val.shape[0]):
                X_val[j, 0, :, i] = smooth(a=X_val[j, 0, :, i], WSZ=WSZ)
                X_val[j, 1, :, i] = smooth(a=X_val[j, 1, :, i], WSZ=WSZ)
            for j in range(X_test.shape[0]):
                X_test[j, 0, :, i] = smooth(a=X_test[j, 0, :, i], WSZ=WSZ)
                X_test[j, 1, :, i] = smooth(a=X_test[j, 1, :, i], WSZ=WSZ)
        del i, j

    a = len(Y_train)+len(Y_val)+len(Y_test)
    print('数据总数为:%s' %(a))
    b = int(np.sum(Y_train)+np.sum(Y_val)+np.sum(Y_test))

    print('同步状态数量:非同步状态=%s:%s' %(a-b, b))

    print('训练集：',X_train.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_train)-int(np.sum(Y_train)), int(np.sum(Y_train))))
    print('验证集：',X_val.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_val)-int(np.sum(Y_val)), int(np.sum(Y_val))))
    print('测试集：',X_test.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_test)-int(np.sum(Y_test)), int(np.sum(Y_test))))

    print('数据分组完毕，开始训练')

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, a, b

def load_data_four_IEEE_l(N, omega, timelength, chosedlength, TEST_SIZE, CHANNEL, interval, relative, normalize, standard, mode, move, WSZ):
    
    start = time.perf_counter()

    if N == 14:
        ## load data 400*1001
        length = 400
    
    X_four_theta_2 = np.zeros((length, chosedlength*N))
    X_four_omega_2 = np.zeros((length, chosedlength*N))
    
    if interval == True:
        f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_four_node_long/400/1_2_3_4.h5' %(N, omega), 'r')
    else:
        f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_four_node_long/400_no_interval/1_2_3_4.h5' %(N, omega), 'r')
    
    if chosedlength != timelength:
        for i in range(N):
            X_four_theta_2[:, i*chosedlength:(i+1)*chosedlength] = f['data_theta'][()][:, i*timelength:i*timelength+chosedlength]
            X_four_omega_2[:, i*chosedlength:(i+1)*chosedlength] = f['data_omega'][()][:, i*timelength:i*timelength+chosedlength]
    else:
        X_four_theta_2 = f['data_theta'][()]
        X_four_omega_2 = f['data_omega'][()]
    
    Y_four_2 = f['Y'][()]
    f.close()
    del f

    for i in range(N-3):
        for j in range(i+1, N-2):
            for k in range(j+1, N -1):
                for l in range(k+1, N):
                    if i == 0 and j == 1 and k == 2 and l == 3:
                        pass
                    else:
                        X_theta = np.zeros((length, chosedlength*N))
                        X_omega = np.zeros((length, chosedlength*N))
                        if interval == True:
                            f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_four_node_long/400/%s_%s_%s_%s.h5' % (N, omega, i+1, j+1, k+1, l+1), 'r')
                        else:
                            f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_four_node_long/400_no_interval/%s_%s_%s_%s.h5' % (N, omega, i+1, j+1, k+1, l+1), 'r')
                        if chosedlength != timelength:
                            for ii in range(N):
                                X_theta[:, ii*chosedlength:(ii+1)*chosedlength] = f['data_theta'][()][:, ii*timelength:ii*timelength+chosedlength]
                                X_omega[:, ii*chosedlength:(ii+1)*chosedlength] = f['data_omega'][()][:, ii*timelength:ii*timelength+chosedlength]
                        else:
                            X_theta = f['data_theta'][()]
                            X_omega = f['data_omega'][()]
                        Y = f['Y'][()]

                        X_four_theta_2 = np.vstack((X_four_theta_2, X_theta))
                        X_four_omega_2 = np.vstack((X_four_omega_2, X_omega))
                        Y_four_2 = np.hstack((Y_four_2, Y))
                        del f, X_theta, X_omega, Y

    end = time.perf_counter()
    print('所用时间为%ss' % (str(end - start)))

    X_four_theta_2 = np.float32(X_four_theta_2)
    X_four_omega_2 = np.float32(X_four_omega_2)
    if normalize:
        X_four_omega_norm, X_four_theta_norm = normalization(x_theta=X_four_theta_2, x_omega=X_four_omega_2)
        del X_four_theta_2, X_four_omega_2
        X_train, X_val, X_test, Y_train, Y_val, Y_test = classify_random_4(x_theta=X_four_theta_norm,
                                                                           x_omega=X_four_omega_norm,
                                                                           y_data=Y_four_2,
                                                                           test_size=TEST_SIZE,
                                                                           timelength=timelength,
                                                                           chosedlength=chosedlength,
                                                                           CHANNEL=CHANNEL,
                                                                           N=N)
    elif standard:
        X_four_omega_std, X_four_theta_std = standardization(N=N, chosedlength=chosedlength, x_theta=X_four_theta_2, x_omega=X_four_omega_2, relative=relative, mode=mode)
        del X_four_theta_2, X_four_omega_2
        X_train, X_val, X_test, Y_train, Y_val, Y_test = classify_random_4(x_theta=X_four_theta_std,
                                                                        x_omega=X_four_omega_std,
                                                                        y_data=Y_four_2,
                                                                        test_size=TEST_SIZE,
                                                                        timelength=timelength,
                                                                        chosedlength=chosedlength,
                                                                        CHANNEL=CHANNEL,
                                                                        N=N)
    else:
        X_train, X_val, X_test, Y_train, Y_val, Y_test = classify_random_4(x_theta=X_four_theta_2,
                                                                        x_omega=X_four_omega_2,
                                                                        y_data=Y_four_2,
                                                                        test_size=TEST_SIZE,
                                                                        timelength=timelength,
                                                                        chosedlength=chosedlength,
                                                                        CHANNEL=CHANNEL,
                                                                        N=N)
        del X_four_theta_2, X_four_omega_2
    
    if move:
        for i in range(N):
            for j in range(X_train.shape[0]):
                X_train[j, 0, :, i] = smooth(a=X_train[j, 0, :, i], WSZ=WSZ)
                X_train[j, 1, :, i] = smooth(a=X_train[j, 1, :, i], WSZ=WSZ)
            for j in range(X_val.shape[0]):
                X_val[j, 0, :, i] = smooth(a=X_val[j, 0, :, i], WSZ=WSZ)
                X_val[j, 1, :, i] = smooth(a=X_val[j, 1, :, i], WSZ=WSZ)
            for j in range(X_test.shape[0]):
                X_test[j, 0, :, i] = smooth(a=X_test[j, 0, :, i], WSZ=WSZ)
                X_test[j, 1, :, i] = smooth(a=X_test[j, 1, :, i], WSZ=WSZ)
        del i, j
    
    a = len(Y_train)+len(Y_val)+len(Y_test)
    print('数据总数为:%s' %(a))
    b = int(np.sum(Y_train)+np.sum(Y_val)+np.sum(Y_test))

    print('同步状态数量:非同步状态=%s:%s' %(a-b, b))

    print('训练集：',X_train.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_train)-int(np.sum(Y_train)), int(np.sum(Y_train))))
    print('验证集：',X_val.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_val)-int(np.sum(Y_val)), int(np.sum(Y_val))))
    print('测试集：',X_test.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_test)-int(np.sum(Y_test)), int(np.sum(Y_test))))

    print('数据分组完毕，开始训练')
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, a, b

def load_data_all_IEEE_l(N, omega, timelength, chosedlength, TEST_SIZE, CHANNEL, interval, relative, normalize, standard, mode, move, WSZ):

    start = time.perf_counter()

    X_two_train, X_two_val, X_two_test, Y_two_train, Y_two_val, Y_two_test, a_two, b_two = load_data_two_IEEE_l(N=N,
                                                                                                                omega=omega,
                                                                                                                timelength=timelength,
                                                                                                                chosedlength=chosedlength,
                                                                                                                TEST_SIZE=TEST_SIZE,
                                                                                                                CHANNEL=CHANNEL,
                                                                                                                interval=interval,
                                                                                                                relative=relative,
                                                                                                                normalize=normalize,
                                                                                                                standard=standard,
                                                                                                                mode=mode,
                                                                                                                move=move,
                                                                                                                WSZ=WSZ)
    
    X_three_train, X_three_val, X_three_test, Y_three_train, Y_three_val, Y_three_test, a_three, b_three = load_data_three_IEEE_l(N=N,
                                                                                                                                  omega=omega,
                                                                                                                                  timelength=timelength,
                                                                                                                                  chosedlength=chosedlength,
                                                                                                                                  TEST_SIZE=TEST_SIZE,
                                                                                                                                  CHANNEL=CHANNEL,
                                                                                                                                  interval=interval,
                                                                                                                                  relative=relative,
                                                                                                                                  normalize=normalize,
                                                                                                                                  standard=standard,
                                                                                                                                  mode=mode,
                                                                                                                                  move=move,
                                                                                                                                  WSZ=WSZ)
    
    X_four_train, X_four_val, X_four_test, Y_four_train, Y_four_val, Y_four_test, a_four, b_four = load_data_four_IEEE_l(N=N,
                                                                                                                         omega=omega,
                                                                                                                         timelength=timelength,
                                                                                                                         chosedlength=chosedlength,
                                                                                                                         TEST_SIZE=TEST_SIZE,
                                                                                                                         CHANNEL=CHANNEL,
                                                                                                                         interval=interval,
                                                                                                                         relative=relative,
                                                                                                                         normalize=normalize,
                                                                                                                         standard=standard,
                                                                                                                         mode=mode,
                                                                                                                         move=move,
                                                                                                                         WSZ=WSZ)
    
    a = a_two + a_three + a_four
    b = b_two + b_three + b_four

    ## dataset
    X_all_train = np.concatenate((X_two_train, X_three_train, X_four_train),axis=0)
    del X_two_train, X_three_train, X_four_train

    X_all_val = np.concatenate((X_two_val, X_three_val, X_four_val),axis=0)
    del X_two_val, X_three_val, X_four_val

    X_all_test = np.concatenate((X_two_test, X_three_test, X_four_test),axis=0)
    del X_two_test, X_three_test, X_four_test

    ## labels
    Y_all_train = np.concatenate((Y_two_train, Y_three_train, Y_four_train),axis=0)
    del Y_two_train, Y_three_train, Y_four_train

    Y_all_val = np.concatenate((Y_two_val, Y_three_val, Y_four_val),axis=0)
    del Y_two_val, Y_three_val, Y_four_val

    Y_all_test = np.concatenate((Y_two_test, Y_three_test, Y_four_test),axis=0)
    del Y_two_test, Y_three_test, Y_four_test

    end = time.perf_counter()

    print('所用时间为%ss' % (str(end - start)))

    print('数据总数为:%s' %(a))
    print('同步状态数量:非同步状态=%s:%s' %(a-b, b))

    print('训练集：',X_all_train.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_all_train)-int(np.sum(Y_all_train)), int(np.sum(Y_all_train))))
    print('验证集：',X_all_val.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_all_val)-int(np.sum(Y_all_val)), int(np.sum(Y_all_val))))
    print('测试集：',X_all_test.shape)
    print('同步状态数量:非同步状态=%s:%s' %(len(Y_all_test)-int(np.sum(Y_all_test)), int(np.sum(Y_all_test))))

    print('数据分组完毕，开始训练')

    return X_all_train, X_all_val, X_all_test, Y_all_train, Y_all_val, Y_all_test, a, b

if __name__ == '__main__':
    # X_train, X_val, X_test, Y_train, Y_val, Y_test, a, b = load_data_four_IEEE_l(N=14,
    #                                                                             omega=20,
    #                                                                             timelength=400,
    #                                                                             chosedlength=10,
    #                                                                             TEST_SIZE=0.2,
    #                                                                             CHANNEL=2,
    #                                                                             interval=False,
    #                                                                             relative=False,
    #                                                                             normalize=False,
    #                                                                             standard=True,
    #                                                                             mode=1,
    #                                                                             move=False)
    Y, PY = load_para_h5(
        N=39, M=60000, omega_s=100*math.pi, baseMVA=10**8, net='GCN', adj_mode=2
    )
    print(Y)
