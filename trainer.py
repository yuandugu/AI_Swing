import numpy as np
np.random.seed(1337)
import math
import time
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
SEED=20000
tf.random.set_seed(seed=SEED)
from tensorflow.keras.optimizer import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy, Accuracy, BinaryAccuracy, CategoricalAccuracy
from spektral.layers import GraphConv, GraphAttention, GlobalAvgPool
from spektral.utils import batch_iterator
from utils.load_data import load_data_one_IEEE_l, load_data_two_IEEE_l, load_data_three_IEEE_l, load_data_four_IEEE_l, load_para
from utils.load_data import load_data_one_IEEE_l_oversampling, load_data_one_IEEE_l_oversampling_2, load_data_one_IEEE_l_oversampling_3, load_data_one_IEEE_l_oversampling_4
from utils.load_data import normalization, standardization, smooth
from model.model_1 import GCN_Net, GCN_LSTM_Net, GCN_TCN_Net, GCN_TCN_Net_Logits, GCN_TCN_2_Net, GCN_2_Net, GAT_Net, GAT_3_Net
from utils.loss_fn import variant_focal_loss, focal_loss_2, focal_loss_3, class_balanced_sigmoid_cross_entropy
import os

#####################  set parameters  ####################

N             = 39                     # number of node
omega_s       = 100 * math.pi          # synchronous angular frequency
baseMVA       = 10**8                  # power reference value
if N == 14:
    M         = 6800                   # mass moments of inertia, 6800 for 14, 12000 for 118
elif N == 39:
    M         = 50000
elif N == 118:
    M         = 12000
alpha         = 0.1                    # damping
theta         = math.pi                # range of theta_0
omega         = 20                    # range of omega_0
exp_num = 81

early_stop = True
interval = False

relative = False

normalize = False

standard = False
mode = 1

move = False
WSZ = 11
oversample = True
n_critical = 10000

if interval == True:
    if N == 14:
        timelength = 100
    elif N == 39:
        timelength = 50
    elif N == 118:
        timelength = 100

else:
    if N == 14:
        timelength = 400
    elif N == 39:
        timelength = 100           # 原始数据的时间长度
    elif N == 118:
        timelength = 100
net = 'GCN'
data_set = 'one'
adj_mode      = 2                       # 邻接矩阵模式：1、adj=Y
                                        #              2、adj=diag(P)+Y
                                        #              3、adj=P'+Y',P'=P·(1+ω_0/ω_s),Y'=Y_ij·sin(θ_i-θ_j)
print('adj_mode=%s' %(adj_mode))
chosedlength  =  20                     # length used to train
TEST_SIZE     =  0.2                    # train:val_test = 6:2:2
CHANNEL       =  1                      # only use omega data to train

if net == 'GCN' or 'RGCN' or 'RGCN-TCN' or 'RGCN-TCN_2':
    learning_rate = 1e-3                    # Learning rate for Adam
elif net == 'GAT' or 'RGAT':
    learning_rate = 5e-3
BATCH_SIZE    = 256                     # Batch size
epochs        = 1000                    # Number of training epochs
patience      = 200                       # Patience for early stopping

#####################  load data & processing  ####################

def gen_train(N, x_omega, y_data, chosedlength, CHANNEL):
    """
    randomly classify data to train group

    (N, chosedlength)
    """
    X_train, Y_train = shuffle(x_omega, y_data)
    X_train = np.reshape(X_train, (x_omega.shape[0], N, chosedlength))
    
    return X_train, Y_train

def gen_val_test(N, x_omega, y_data, test_size, chosedlength, timelength, CHANNEL):
    """
    randomly classify data to val and test group
    
    (N, chosedlength)
    """
    x_train, X_test, y_train, Y_test = train_test_split(x_omega, y_data, test_size=test_size, random_state=0) # 随机分组
    if chosedlength == timelength:
        x_train = np.float32(x_train)
        x_test = np.float32(x_test)
    X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=0) # 随机分组
    if chosedlength == timelength:
        X_train = np.float32(X_train)
        X_val = np.float32(X_val)
    del y_data, x_train, y_train, X_train, Y_train
    X_val = np.reshape(X_val, (X_val.shape[0], N, chosedlength))
    # X_val = np.swapaxes(X_val,1,3)
    X_test = np.reshape(X_test, (X_test.shape[0], N, chosedlength))
    # X_test = np.swapaxes(X_test,1,3)
    
    return X_val, X_test, Y_val, Y_test

def load_one_train(N, omega, timelength, chosedlength, TEST_SIZE, CHANNEL, interval, relative, normalize, standard, mode, move, WSZ, oversample):
    
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
    if interval == True:
        f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s/1.h5' %(N, omega, length), 'r')
        # f = h5py.File('/media/duguyuan/new/Swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s/1.h5' %(N, omega, length), 'r')
    else:
        f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s_no_interval/1.h5' %(N, omega, length), 'r')
        # f = h5py.File('/media/duguyuan/new/Swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s_no_interval/1.h5' %(N, omega, length), 'r')
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
                    
            if interval == True:
                f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s/%s.h5' %(N, omega, length, i+1), 'r')
                # f = h5py.File('/media/duguyuan/new/Swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s/%s_%s.h5' % (N, omega, length, i+1, j+1), 'r')
            else:
                f = h5py.File('/public/home/spy2018/swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s_no_interval/%s.h5' %(N, omega, length, i+1), 'r')
                # f = h5py.File('/media/duguyuan/new/Swing/result/IEEE/case%s/omega==%s/change_one_node_long/%s_no_interval/%s_%s.h5' % (N, omega, length, i+1, j+1), 'r')
            if chosedlength != timelength:
                for ii in range(N):
                    X_theta[:, ii*chosedlength:(ii+1)*chosedlength] = f['data_theta'][()][:, ii*timelength:ii*timelength+chosedlength]
                    X_omega[:, ii*chosedlength:(ii+1)*chosedlength] = f['data_omega'][()][:, ii*timelength:ii*timelength+chosedlength]
            else:
                X_theta = f['data_theta'][()]
                X_omega = f['data_omega'][()]

            Y = f['Y'][()]

            X_one_theta_2 = np.vstack((X_one_theta_2, X_theta))
            X_one_omega_2 = np.vstack((X_one_omega_2, X_omega))
            Y_one_2 = np.hstack((Y_one_2, Y))
            f.close()
            del f, X_theta, X_omega, Y
    
    if oversample:
        Y_1 = np.argwhere(Y_one_2 > 0)
        x_theta = np.zeros((Y_1.shape[0],N*chosedlength))
        x_omega = np.zeros((Y_1.shape[0],N*chosedlength))
        i = 0
        for j in Y_1:
            x_omega[i,:] = X_one_omega_2[j,:]
            x_theta[i,:] = X_one_theta_2[j,:]
            i += 1

        for i in range(4):
            X_one_theta_2 = np.vstack((X_one_theta_2, x_theta + np.random.normal(loc=0,scale=0.4,size=(Y_1.shape[0], N*chosedlength))))
            X_one_omega_2 = np.vstack((X_one_omega_2, x_omega + np.random.normal(loc=0,scale=0.2,size=(Y_1.shape[0], N*chosedlength))))
            Y_one_2 = np.hstack((Y_one_2, np.ones(Y_1.shape[0])))
    
    end = time.perf_counter()
    print('所用时间为%ss' % (str(end - start)))

    if normalize:
        X_one_omega_norm, X_one_theta_norm = normalization(
            N=N, x_theta=X_one_theta_2, x_omega=X_one_omega_2
        )
        del X_one_theta_2, X_one_omega_2
        X_train, Y_train = gen_train(
            N=N,
            # x_theta=X_one_theta_norm,
            x_omega=X_one_omega_norm,
            y_data=Y_one_2,
            chosedlength=chosedlength,
            CHANNEL=CHANNEL
        )
        del X_one_theta_norm, X_one_omega_norm
    elif standard:
        X_one_omega_std, X_one_theta_std = standardization(
            N=N, chosedlength=chosedlength, x_theta=X_one_theta_2, x_omega=X_one_omega_2, relative=relative, mode=mode
        )
        del X_one_theta_2, X_one_omega_2
        X_train, Y_train = gen_train(
            N=N,
            # x_theta=X_one_theta_std,
            x_omega=X_one_omega_std,
            y_data=Y_one_2,
            chosedlength=chosedlength,
            CHANNEL=CHANNEL
        )
        del X_one_theta_std, X_one_omega_std
    else:
        X_train, Y_train = gen_train(
            N=N,
            # x_theta=X_one_theta_2,
            x_omega=X_one_omega_2,
            y_data=Y_one_2,
            chosedlength=chosedlength,
            CHANNEL=CHANNEL
        )
        del X_one_theta_2, X_one_omega_2
    
    if move:
        for i in range(N):
            for j in range(X_train.shape[0]):
                X_train[j, 0, :, i] = smooth(a=X_train[j, 0, :, i], WSZ=WSZ)
                if CHANNEL == 2:
                    X_train[j, 1, :, i] = smooth(a=X_train[j, 1, :, i], WSZ=WSZ)
                else:
                    pass
        del i, j

    print('训练集：', X_train.shape)
    print('同步状态数量:非同步状态=%s:%s' % (len(Y_train)-int(np.sum(Y_train)), int(np.sum(Y_train))))
    
    return X_train, Y_train

def gen_adj(x_omega, x_theta, PY):
    
    init_theta = x_theta[:,:,0]
    Y = np.abs(np.sin(
            np.repeat(
                a=np.expand_dims(init_theta, axis=2),
                repeats=N,
                axis=2
            )-
            np.repeat(
                a=np.expand_dims(init_theta, axis=1),
                repeats=N,
                axis=1)
        ) * np.repeat(
            a=np.expand_dims(PY[1:,:], axis=0),
            repeats=x_theta.shape[0],
            axis=0
        ))
    init_omega = x_omega[:,:,0]
    P = np.abs(np.repeat(
            a = np.expand_dims(
                    a=PY[0,:],
                    axis=0
                ),
            repeats=x_omega.shape[0],
            axis=0
        )) * (1 + 1 / omega_s * init_omega)
    P = np.array([np.diag(P[i,:]) for i in range(P.shape[0])])

    adj = np.array(P + Y)

    return adj

A, PY = load_para(
    N=N, M=M, baseMVA=baseMVA, omega_s=omega_s, net=net, adj_mode=adj_mode
)

if data_set == 'one':

    X_train, X_train_theta, X_val, X_val_theta, X_test, X_test_theta, Y_train, Y_val, Y_test, a, b = load_data_one_IEEE_l_oversampling_4(
        N=N,
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
        WSZ=WSZ
    )

elif data_set == 'two':
    
    X_train, X_val, X_test, Y_train, Y_val, Y_test, a, b = load_data_two_IEEE_l(
        N=N,
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
        WSZ=WSZ
    )

F = chosedlength
n_out = 1

## process adj matrix
if net == 'GCN' or 'RGCN' or 'GAT':
    # Create filter for GCN and convert to sparse tensor
    if adj_mode != 3:
        # (N, N)
        adj = GraphConv.preprocess(A=A)
        del X_train_theta, X_val_theta, X_test_theta
        adj_train = np.repeat(
            a=np.expand_dims(adj, axis=0),
            repeats=X_train.shape[0],
            axis=0
        )
        adj_val = np.repeat(
            a=np.expand_dims(adj, axis=0),
            repeats=X_val.shape[0],
            axis=0
        )
        adj_test = np.repeat(
            a=np.expand_dims(adj, axis=0),
            repeats=X_test.shape[0],
            axis=0
        )
    else:
        # (length, N, N)
        adj_train = gen_adj(x_omega=X_train, x_theta=X_train_theta, PY=PY)
        adj_val = gen_adj(x_omega=X_val, x_theta=X_val_theta, PY=PY)
        adj_test = gen_adj(x_omega=X_test, x_theta=X_test_theta, PY=PY)

        adj_train = np.array([GraphConv.preprocess(adj_train[i,:,:]) for i in range(Y_train.shape[0])])
        adj_val = np.array([GraphConv.preprocess(adj_val[i,:,:]) for i in range(Y_val.shape[0])])
        adj_test = np.array([GraphConv.preprocess(adj_test[i,:,:]) for i in range(Y_test.shape[0])])
    # fltr = sp_matrix_to_sp_tensor(fltr)
elif net == 'GAT' or 'RGAT':
    # Add self-loop
    fltr = A + np.eye(A.shape[0])

#####################  Network setup  ####################

if net == 'GCN':
    model = GCN_2_Net()
elif net == 'RGCN':
    model = GCN_LSTM_Net()
elif net == 'GAT':
    model = GAT_3_Net()

optimizer = Adam(lr=learning_rate)
loss_fn = BinaryCrossentropy()
acc_fn = BinaryAccuracy()

#####################  Functions  ####################

# Training step
@tf.function
def train(x, fltr, y):
    with tf.GradientTape() as tape:
        predictions = model([x, fltr], training=True)
        loss = loss_fn(y, predictions)
        loss += sum(model.losses)
    acc = acc_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, acc

# Evaluation step
@tf.function
def evaluate(x, fltr, y):
    predictions = model([x, fltr], training=False)
    loss = loss_fn(y, predictions)
    loss += sum(model.losses)
    acc = acc_fn(y, predictions)

    return loss, acc

# Testing step
@tf.function
def test(x, fltr, y):
    predictions = model([x, fltr], training=False)
    loss = loss_fn(y, predictions)
    loss += sum(model.losses)
    acc = acc_fn(y, predictions)

    return loss, acc, predictions

# Setup training
best_val_loss = 99999
current_patience = patience
curent_batch = 0
batches_in_epoch = int(np.ceil(X_train.shape[0] / BATCH_SIZE))
batches_tr = batch_iterator([X_train, adj_train, Y_train], batch_size=BATCH_SIZE, epochs=epochs)
# Training loop
loss_train = []
acc_train  = []

loss_val   = []
acc_val    = []

loss_test  = []
acc_test   = []

results_tr = []
results_te = []
n = 0
print('\nTraining ------------')
for batch in batches_tr:
    curent_batch += 1
    # Training step
    loss, acc = train(*batch)
    results_tr.append((loss, acc))

    if curent_batch == batches_in_epoch:
        n = n + 1
        if n >= n_critical:
            loss_fn = variant_focal_loss(alpha=0.90)
        else:
            pass
        if n == n_critical:
            print('Loss function=Focal Loss')
        else:
            pass

        batches_va = batch_iterator([X_val, adj_val, Y_val], batch_size=BATCH_SIZE)
        results_va = [evaluate(*batch) for batch in batches_va]
        results_va = np.array(results_va)
        loss_va, acc_va = results_va.mean(0)
        if loss_va < best_val_loss:
            best_val_loss = loss_va
            current_patience = patience
            # Test
            batches_te = batch_iterator([X_test, adj_test, Y_test], batch_size=BATCH_SIZE)
            results_te = [evaluate(*batch) for batch in batches_te]
            results_te = np.array(results_te)
        else:
            current_patience -= 1
            if current_patience == 0:
                print('Early stopping')
                break

        # Print results
        results_tr = np.array(results_tr)
        print('Epochs: {:.0f} | '
              'Train loss: {:.4f}, acc: {:.4f} | '
              'Valid loss: {:.4f}, acc: {:.4f} | '
              'Test loss: {:.4f}, acc: {:.4f}'
              .format(n,
                      *results_tr.mean(0),
                      *results_va.mean(0),
                      *results_te.mean(0)))
        loss, acc = results_tr.mean(0)
        loss_train.append(loss)
        acc_train.append(acc)

        loss, acc = results_va.mean(0)
        loss_val.append(loss)
        acc_val.append(acc)

        loss, acc = results_te.mean(0)
        loss_test.append(loss)
        acc_test.append(acc)
        # Reset epoch
        results_tr = []
        curent_batch = 0

loss_train = np.array(loss_train)
acc_train  = np.array(acc_train)
loss_val   = np.array(loss_val)
acc_val    = np.array(acc_val)
loss_test  = np.array(loss_test)
acc_test   = np.array(acc_test)

EPOCHS = loss_train.shape[0]

HISTORY = np.zeros((4, EPOCHS))
HISTORY[0, :] = acc_train
HISTORY[1, :] = acc_val
HISTORY[2, :] = loss_train
HISTORY[3, :] = loss_val

#####################  Testing  ####################

print('\nTesting ------------')
loss_fn = BinaryCrossentropy()
loss, accuracy, Y_predict = test(X_test, adj_test, Y_test)
print('model test loss: ', loss)
print('model test accuracy: ', accuracy)

## 获取预测值序列
# Y_predict = model.predict(X_test)
Y_predict_int = np.rint(Y_predict)

## 矩阵
from sklearn.metrics import confusion_matrix

con_mat = confusion_matrix(Y_test, Y_predict_int)
print(con_mat)
con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]  # 归一化
con_mat_norm = np.around(con_mat_norm, decimals=2)

## AUC
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
fpr, tpr, thresholds_keras = roc_curve(Y_test.astype(int), Y_predict)   
auc = auc(fpr, tpr)
print("AUC : ", auc)

"""
保存数据
"""
f = h5py.File('histroy.h5', 'w')
f.create_dataset('train_history', data=HISTORY)
f.create_dataset('test_loss', data=loss)
f.create_dataset('test_accuracy', data=accuracy)
f.create_dataset('test_matrix', data=con_mat)
f.create_dataset('test_fpr', data=fpr)
f.create_dataset('test_tpr', data=tpr)
f.create_dataset('test_AUC', data=auc)
f.create_dataset('pre', data=Y_predict)
f.close()

model.save('my_model.h5')
